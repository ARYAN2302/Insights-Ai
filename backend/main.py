from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
import os
import shutil
import time
from datetime import datetime, timedelta
import uuid
import pandas as pd
import numpy as np

# Local imports
from database import get_db, create_tables
from models import User, Dataset, Query, Report, Alert
import schemas
import crud
from auth import create_access_token, verify_token
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'ai'))
from data_processor import AdvancedDataProcessor
from llm_service import GroqLLMService

app = FastAPI(
    title="Insights AI API",
    description="GenAI-powered business insights platform",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize services
data_processor = AdvancedDataProcessor()
llm_service = GroqLLMService()

# Create tables on startup
@app.on_event("startup")
async def startup_event():
    create_tables()

# Dependency to get current user
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), 
                          db: Session = Depends(get_db)):
    token = credentials.credentials
    payload = verify_token(token)
    email = payload.get("sub")
    
    user = crud.get_user_by_email(db, email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    return user

# Auth endpoints
@app.post("/api/auth/register", response_model=schemas.User)
async def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = crud.get_user_by_email(db, user.email)
    if db_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    db_user = crud.create_user(db, user)
    return db_user

@app.post("/api/auth/login", response_model=schemas.Token)
async def login(email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = crud.authenticate_user(db, email, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/api/auth/me", response_model=schemas.User)
async def get_me(current_user: User = Depends(get_current_user)):
    return current_user

# Dataset endpoints
@app.post("/api/datasets/upload", response_model=schemas.FileUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(""),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate file type
    if not file.filename.endswith(('.csv', '.xlsx', '.xls', '.json')):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload CSV, Excel, or JSON files."
        )
    
    # Determine file type
    file_type = 'csv' if file.filename.endswith('.csv') else \
               'excel' if file.filename.endswith(('.xlsx', '.xls')) else 'json'
    
    # Create unique filename
    file_extension = file.filename.split('.')[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = os.path.join("../storage", unique_filename)
    
    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    try:
        # Process file
        columns_info, row_count = data_processor.process_file(file_path, file_type)
        file_size = os.path.getsize(file_path)
        
        # Create dataset record
        dataset_create = schemas.DatasetCreate(name=name, description=description)
        dataset = crud.create_dataset(
            db, dataset_create, current_user.id, file_path, file_type, 
            file_size, columns_info, row_count
        )
        
        # Embed dataset for RAG
        llm_service.embed_dataset(dataset.id, file_path, file_type, columns_info)
        
        return {
            "dataset_id": dataset.id,
            "message": "Dataset uploaded successfully",
            "file_info": {
                "name": name,
                "size": file_size,
                "rows": row_count,
                "columns": len(columns_info.get("columns", {}))
            }
        }
    
    except Exception as e:
        # Clean up file on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/datasets", response_model=List[schemas.Dataset])
async def get_datasets(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_datasets(db, current_user.id, skip, limit)

@app.get("/api/datasets/{dataset_id}", response_model=schemas.Dataset)
async def get_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset

@app.get("/api/datasets/{dataset_id}/sample")
async def get_dataset_sample(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        sample_data = data_processor.get_sample_data(dataset.file_path, dataset.file_type)
        return sample_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/datasets/{dataset_id}")
async def delete_dataset(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Delete file
    if os.path.exists(dataset.file_path):
        os.remove(dataset.file_path)
    
    # Delete dataset record
    crud.delete_dataset(db, dataset_id, current_user.id)
    
    return {"message": "Dataset deleted successfully"}

# Chat endpoints
@app.post("/api/chat/query", response_model=schemas.ChatResponse)
async def chat_query(
    message: schemas.ChatMessage,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    try:
        if message.dataset_id:
            # Get dataset
            dataset = crud.get_dataset(db, message.dataset_id, current_user.id)
            if not dataset:
                raise HTTPException(status_code=404, detail="Dataset not found")
            
            # Query dataset using LLM
            result = llm_service.query_dataset(
                message.dataset_id, 
                message.message, 
                dataset.columns_info or {}
            )
            
            if not result["success"]:
                raise HTTPException(status_code=500, detail=result["response"])
            
            response_text = result["response"]
        else:
            # General query without dataset
            response_text = "I can help you analyze your data! Please upload a dataset first or specify which dataset you'd like to analyze."
        
        execution_time = int((time.time() - start_time) * 1000)
        
        # Store query
        query_create = schemas.QueryCreate(
            question=message.message,
            dataset_id=message.dataset_id
        )
        crud.create_query(db, query_create, current_user.id, response_text, execution_time)
        
        return {
            "response": response_text,
            "execution_time": execution_time
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chat/history", response_model=List[schemas.Query])
async def get_chat_history(
    dataset_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if dataset_id:
        return crud.get_dataset_queries(db, dataset_id, current_user.id, skip, limit)
    else:
        return crud.get_user_queries(db, current_user.id, skip, limit)

# Report endpoints
@app.post("/api/reports", response_model=schemas.Report)
async def create_report(
    report: schemas.ReportCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate dataset
    dataset = crud.get_dataset(db, report.dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Create report record
    db_report = crud.create_report(db, report, current_user.id)
    
    # Generate report content asynchronously (in production, use background tasks)
    try:
        result = llm_service.generate_report(
            report.dataset_id,
            report.report_type,
            dataset.columns_info or {}
        )
        
        if result["success"]:
            # Update report with generated content
            crud.update_report(
                db, db_report.id, current_user.id,
                result["content"],
                result.get("charts_suggestions", []),
                "completed"
            )
        else:
            crud.update_report(
                db, db_report.id, current_user.id,
                result["content"],
                {},
                "failed"
            )
    except Exception as e:
        crud.update_report(
            db, db_report.id, current_user.id,
            f"Error generating report: {str(e)}",
            {},
            "failed"
        )
    
    return crud.get_report(db, db_report.id, current_user.id)

@app.get("/api/reports", response_model=List[schemas.Report])
async def get_reports(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_user_reports(db, current_user.id, skip, limit)

@app.get("/api/reports/{report_id}", response_model=schemas.Report)
async def get_report(
    report_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = crud.get_report(db, report_id, current_user.id)
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report

# Alert endpoints
@app.get("/api/alerts", response_model=List[schemas.Alert])
async def get_alerts(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_user_alerts(db, current_user.id, skip, limit)

@app.get("/api/alerts/unread", response_model=List[schemas.Alert])
async def get_unread_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return crud.get_unread_alerts(db, current_user.id)

@app.post("/api/alerts/{alert_id}/mark-read")
async def mark_alert_read(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    alert = crud.mark_alert_read(db, alert_id, current_user.id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert marked as read"}

# Generate insights endpoint
@app.post("/api/datasets/{dataset_id}/insights")
async def generate_insights(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Generate insights using LLM
        insights = llm_service.detect_insights(dataset_id, dataset.columns_info or {})
        
        # Create alerts for high-severity insights
        for insight in insights:
            if insight.get("severity") in ["medium", "high"]:
                alert_create = schemas.AlertCreate(
                    title=insight["title"],
                    message=insight["description"],
                    alert_type=insight["type"],
                    severity=insight["severity"],
                    dataset_id=dataset_id
                )
                crud.create_alert(db, alert_create, current_user.id, insight)
        
        return {"insights": insights}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Analytics Endpoints

@app.post("/api/datasets/{dataset_id}/clustering")
async def perform_clustering(
    dataset_id: int,
    n_clusters: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform clustering analysis on the dataset."""
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        clustering_results = data_processor.perform_clustering(
            dataset.file_path, 
            dataset.file_type, 
            n_clusters
        )
        
        # Save clustering results as a report
        if "error" not in clustering_results:
            report_create = schemas.ReportCreate(
                title=f"Clustering Analysis - {dataset.name}",
                report_type="clustering",
                content=clustering_results,
                dataset_id=dataset_id
            )
            crud.create_report(db, report_create, current_user.id)
        
        return clustering_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clustering analysis failed: {str(e)}")

@app.post("/api/datasets/{dataset_id}/anomaly-detection")
async def detect_anomalies(
    dataset_id: int,
    contamination: float = 0.1,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform anomaly detection on the dataset."""
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        anomaly_results = data_processor.detect_anomalies(
            dataset.file_path, 
            dataset.file_type,
            contamination
        )
        
        # Create alerts for significant anomalies
        if "error" not in anomaly_results:
            total_anomalies = 0
            for method, results in anomaly_results.items():
                if isinstance(results, dict) and "count" in results:
                    total_anomalies += results["count"]
            
            if total_anomalies > 0:
                alert_create = schemas.AlertCreate(
                    title=f"Anomalies Detected - {dataset.name}",
                    message=f"Found {total_anomalies} potential anomalies in the dataset",
                    alert_type="anomaly",
                    severity="medium" if total_anomalies > len(dataset.columns_info) * 0.05 else "low",
                    dataset_id=dataset_id
                )
                crud.create_alert(db, alert_create, current_user.id, anomaly_results)
        
        return anomaly_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly detection failed: {str(e)}")

@app.post("/api/datasets/{dataset_id}/comprehensive-insights")
async def generate_comprehensive_insights(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate comprehensive business insights for the dataset."""
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Generate comprehensive insights
        insights = data_processor.generate_insights(dataset.file_path, dataset.file_type)
        
        if "error" not in insights:
            # Save insights as a report
            report_create = schemas.ReportCreate(
                title=f"Comprehensive Insights - {dataset.name}",
                report_type="insights",
                content=insights,
                dataset_id=dataset_id
            )
            crud.create_report(db, report_create, current_user.id)
            
            # Create alerts for risk factors
            if "risk_factors" in insights and insights["risk_factors"]:
                for risk in insights["risk_factors"]:
                    alert_create = schemas.AlertCreate(
                        title=f"Risk Factor Identified - {dataset.name}",
                        message=risk,
                        alert_type="risk",
                        severity="medium",
                        dataset_id=dataset_id
                    )
                    crud.create_alert(db, alert_create, current_user.id, {"risk_factor": risk})
        
        return insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insight generation failed: {str(e)}")

@app.post("/api/datasets/{dataset_id}/data-profiling")
async def perform_data_profiling(
    dataset_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Perform comprehensive data profiling."""
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Get comprehensive metadata (already generated during upload)
        metadata = dataset.columns_info
        
        # Generate additional profiling insights
        profiling_insights = {
            "data_quality_score": metadata.get("data_quality", {}).get("quality_score", 0),
            "recommendations": metadata.get("data_quality", {}).get("recommendations", []),
            "patterns": metadata.get("patterns", {}),
            "statistical_summary": metadata.get("statistical_summary", {}),
            "correlations": metadata.get("correlations", {})
        }
        
        return profiling_insights
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data profiling failed: {str(e)}")

@app.post("/api/datasets/{dataset_id}/predictive-modeling")
async def generate_predictive_model(
    dataset_id: int,
    target_column: str,
    model_type: str = "linear_regression",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Generate a simple predictive model for the dataset."""
    dataset = crud.get_dataset(db, dataset_id, current_user.id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    try:
        # Load data
        if dataset.file_type == 'csv':
            df = pd.read_csv(dataset.file_path)
        elif dataset.file_type == 'excel':
            df = pd.read_excel(dataset.file_path)
        elif dataset.file_type == 'json':
            df = pd.read_json(dataset.file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        # Verify target column exists
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found")
        
        # Basic predictive modeling
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_column not in numeric_cols:
            raise HTTPException(status_code=400, detail="Target column must be numeric")
        
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if len(feature_cols) == 0:
            raise HTTPException(status_code=400, detail="No numeric features found for modeling")
        
        # Prepare data
        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df[target_column].fillna(df[target_column].mean())
        
        # Train simple model
        if model_type == "linear_regression":
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error, r2_score
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            feature_importance = {
                feature: coef for feature, coef in zip(feature_cols, model.coef_)
            }
            
            model_results = {
                "model_type": "linear_regression",
                "target_column": target_column,
                "features": feature_cols,
                "performance": {
                    "mse": mse,
                    "r2_score": r2,
                    "rmse": np.sqrt(mse)
                },
                "feature_importance": feature_importance,
                "predictions_sample": y_pred[:10].tolist()
            }
            
            # Save model results as a report
            report_create = schemas.ReportCreate(
                title=f"Predictive Model - {dataset.name}",
                report_type="modeling",
                content=model_results,
                dataset_id=dataset_id
            )
            crud.create_report(db, report_create, current_user.id)
            
            return model_results
        
        else:
            raise HTTPException(status_code=400, detail="Unsupported model type")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Predictive modeling failed: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
