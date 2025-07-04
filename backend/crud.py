from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import models
import schemas
from auth import get_password_hash, verify_password

# User CRUD operations
def get_user(db: Session, user_id: int) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.id == user_id).first()

def get_user_by_email(db: Session, email: str) -> Optional[models.User]:
    return db.query(models.User).filter(models.User.email == email).first()

def create_user(db: Session, user: schemas.UserCreate) -> models.User:
    hashed_password = get_password_hash(user.password)
    db_user = models.User(
        email=user.email,
        hashed_password=hashed_password,
        full_name=user.full_name
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str) -> Optional[models.User]:
    user = get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user

def update_user(db: Session, user_id: int, user_update: schemas.UserUpdate) -> Optional[models.User]:
    db_user = get_user(db, user_id)
    if not db_user:
        return None
    
    update_data = user_update.dict(exclude_unset=True)
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    for field, value in update_data.items():
        setattr(db_user, field, value)
    
    db_user.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_user)
    return db_user

# Dataset CRUD operations
def get_datasets(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Dataset]:
    return db.query(models.Dataset).filter(
        models.Dataset.owner_id == user_id
    ).offset(skip).limit(limit).all()

def get_dataset(db: Session, dataset_id: int, user_id: int) -> Optional[models.Dataset]:
    return db.query(models.Dataset).filter(
        and_(models.Dataset.id == dataset_id, models.Dataset.owner_id == user_id)
    ).first()

def create_dataset(db: Session, dataset: schemas.DatasetCreate, user_id: int, 
                  file_path: str, file_type: str, file_size: int, 
                  columns_info: Dict[str, Any], row_count: int) -> models.Dataset:
    db_dataset = models.Dataset(
        name=dataset.name,
        description=dataset.description,
        file_path=file_path,
        file_type=file_type,
        file_size=file_size,
        columns_info=columns_info,
        row_count=row_count,
        owner_id=user_id
    )
    db.add(db_dataset)
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def update_dataset(db: Session, dataset_id: int, user_id: int, 
                  dataset_update: schemas.DatasetUpdate) -> Optional[models.Dataset]:
    db_dataset = get_dataset(db, dataset_id, user_id)
    if not db_dataset:
        return None
    
    update_data = dataset_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_dataset, field, value)
    
    db_dataset.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_dataset)
    return db_dataset

def delete_dataset(db: Session, dataset_id: int, user_id: int) -> bool:
    db_dataset = get_dataset(db, dataset_id, user_id)
    if not db_dataset:
        return False
    
    db.delete(db_dataset)
    db.commit()
    return True

# Query CRUD operations
def create_query(db: Session, query: schemas.QueryCreate, user_id: int, 
                answer: str, execution_time: int) -> models.Query:
    db_query = models.Query(
        question=query.question,
        answer=answer,
        query_type="chat",
        execution_time=execution_time,
        user_id=user_id,
        dataset_id=query.dataset_id
    )
    db.add(db_query)
    db.commit()
    db.refresh(db_query)
    return db_query

def get_user_queries(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Query]:
    return db.query(models.Query).filter(
        models.Query.user_id == user_id
    ).order_by(desc(models.Query.created_at)).offset(skip).limit(limit).all()

def get_dataset_queries(db: Session, dataset_id: int, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Query]:
    return db.query(models.Query).filter(
        and_(models.Query.dataset_id == dataset_id, models.Query.user_id == user_id)
    ).order_by(desc(models.Query.created_at)).offset(skip).limit(limit).all()

# Report CRUD operations
def create_report(db: Session, report: schemas.ReportCreate, user_id: int) -> models.Report:
    db_report = models.Report(
        title=report.title,
        description=report.description,
        report_type=report.report_type,
        scheduled=report.scheduled,
        schedule_frequency=report.schedule_frequency,
        user_id=user_id,
        dataset_id=report.dataset_id
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)
    return db_report

def get_user_reports(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Report]:
    return db.query(models.Report).filter(
        models.Report.user_id == user_id
    ).order_by(desc(models.Report.created_at)).offset(skip).limit(limit).all()

def get_report(db: Session, report_id: int, user_id: int) -> Optional[models.Report]:
    return db.query(models.Report).filter(
        and_(models.Report.id == report_id, models.Report.user_id == user_id)
    ).first()

def update_report(db: Session, report_id: int, user_id: int, 
                 content: str, charts_data: Dict[str, Any], status: str) -> Optional[models.Report]:
    db_report = get_report(db, report_id, user_id)
    if not db_report:
        return None
    
    db_report.content = content
    db_report.charts_data = charts_data
    db_report.status = status
    db_report.updated_at = datetime.utcnow()
    
    db.commit()
    db.refresh(db_report)
    return db_report

# Alert CRUD operations
def create_alert(db: Session, alert: schemas.AlertCreate, user_id: int, 
                metadata: Optional[Dict[str, Any]] = None) -> models.Alert:
    db_alert = models.Alert(
        title=alert.title,
        message=alert.message,
        alert_type=alert.alert_type,
        severity=alert.severity,
        metadata=metadata,
        user_id=user_id,
        dataset_id=alert.dataset_id
    )
    db.add(db_alert)
    db.commit()
    db.refresh(db_alert)
    return db_alert

def get_user_alerts(db: Session, user_id: int, skip: int = 0, limit: int = 100) -> List[models.Alert]:
    return db.query(models.Alert).filter(
        models.Alert.user_id == user_id
    ).order_by(desc(models.Alert.created_at)).offset(skip).limit(limit).all()

def get_unread_alerts(db: Session, user_id: int) -> List[models.Alert]:
    return db.query(models.Alert).filter(
        and_(models.Alert.user_id == user_id, models.Alert.is_read == False)
    ).order_by(desc(models.Alert.created_at)).all()

def mark_alert_read(db: Session, alert_id: int, user_id: int) -> Optional[models.Alert]:
    db_alert = db.query(models.Alert).filter(
        and_(models.Alert.id == alert_id, models.Alert.user_id == user_id)
    ).first()
    
    if not db_alert:
        return None
    
    db_alert.is_read = True
    db.commit()
    db.refresh(db_alert)
    return db_alert

# User Preferences CRUD operations
def get_user_preferences(db: Session, user_id: int) -> Optional[models.UserPreferences]:
    return db.query(models.UserPreferences).filter(
        models.UserPreferences.user_id == user_id
    ).first()

def create_user_preferences(db: Session, preferences: schemas.UserPreferencesCreate, 
                           user_id: int) -> models.UserPreferences:
    db_preferences = models.UserPreferences(
        email_notifications=preferences.email_notifications,
        alert_frequency=preferences.alert_frequency,
        preferred_chart_types=preferences.preferred_chart_types,
        theme=preferences.theme,
        user_id=user_id
    )
    db.add(db_preferences)
    db.commit()
    db.refresh(db_preferences)
    return db_preferences

def update_user_preferences(db: Session, user_id: int, 
                           preferences_update: schemas.UserPreferencesUpdate) -> Optional[models.UserPreferences]:
    db_preferences = get_user_preferences(db, user_id)
    if not db_preferences:
        return None
    
    update_data = preferences_update.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_preferences, field, value)
    
    db_preferences.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_preferences)
    return db_preferences
