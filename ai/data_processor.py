import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class AdvancedDataProcessor:
    """Advanced data processing with comprehensive analytics capabilities."""
    
    def __init__(self, storage_path: str = "../storage"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.scaler = StandardScaler()
    
    def process_file(self, file_path: str, file_type: str) -> Tuple[Dict[str, Any], int]:
        """Process uploaded file and return comprehensive metadata and row count."""
        try:
            # Load data based on file type
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Generate comprehensive metadata
            metadata = self._generate_comprehensive_metadata(df)
            row_count = len(df)
            
            return metadata, row_count
            
        except Exception as e:
            raise Exception(f"Error processing file: {str(e)}")
    
    def _generate_comprehensive_metadata(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive metadata with advanced analytics."""
        metadata = {
            "basic_info": {
                "columns": list(df.columns),
                "shape": df.shape,
                "memory_usage": df.memory_usage(deep=True).sum(),
                "null_counts": df.isnull().sum().to_dict(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "duplicate_rows": df.duplicated().sum()
            },
            "column_details": {},
            "correlations": {},
            "statistical_summary": {},
            "data_quality": {},
            "patterns": {}
        }
        
        # Detailed column analysis
        for column in df.columns:
            col_info = self._analyze_column(df, column)
            metadata["column_details"][column] = col_info
        
        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            correlation_matrix = df[numeric_cols].corr()
            metadata["correlations"] = {
                "matrix": correlation_matrix.round(3).to_dict(),
                "strong_correlations": self._find_strong_correlations(correlation_matrix)
            }
        
        # Statistical summary
        metadata["statistical_summary"] = self._generate_statistical_summary(df)
        
        # Data quality assessment
        metadata["data_quality"] = self._assess_data_quality(df)
        
        # Pattern detection
        metadata["patterns"] = self._detect_patterns(df)
        
        return metadata
    
    def _analyze_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Comprehensive column analysis."""
        col_data = df[column]
        
        analysis = {
            "dtype": str(col_data.dtype),
            "null_count": col_data.isnull().sum(),
            "null_percentage": (col_data.isnull().sum() / len(col_data)) * 100,
            "unique_count": col_data.nunique(),
            "unique_percentage": (col_data.nunique() / len(col_data)) * 100,
            "is_numeric": pd.api.types.is_numeric_dtype(col_data),
            "is_categorical": pd.api.types.is_categorical_dtype(col_data),
            "is_datetime": pd.api.types.is_datetime64_any_dtype(col_data),
            "is_constant": col_data.nunique() <= 1,
            "has_outliers": False
        }
        
        # Numeric column analysis
        if analysis["is_numeric"]:
            analysis["statistics"] = {
                "mean": col_data.mean(),
                "median": col_data.median(),
                "std": col_data.std(),
                "min": col_data.min(),
                "max": col_data.max(),
                "q25": col_data.quantile(0.25),
                "q75": col_data.quantile(0.75),
                "skewness": col_data.skew(),
                "kurtosis": col_data.kurtosis()
            }
            
            # Outlier detection using IQR method
            Q1 = col_data.quantile(0.25)
            Q3 = col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
            analysis["outliers"] = {
                "count": len(outliers),
                "percentage": (len(outliers) / len(col_data)) * 100,
                "values": outliers.head(10).tolist()
            }
            analysis["has_outliers"] = len(outliers) > 0
        
        # Categorical/text column analysis
        if analysis["unique_count"] < 50:
            value_counts = col_data.value_counts()
            analysis["value_counts"] = value_counts.head(20).to_dict()
            analysis["most_frequent"] = value_counts.index[0] if not value_counts.empty else None
            analysis["frequency_distribution"] = {
                "entropy": self._calculate_entropy(value_counts),
                "concentration": value_counts.iloc[0] / len(col_data) if not value_counts.empty else 0
            }
        
        # String column analysis
        if col_data.dtype == 'object':
            string_lengths = col_data.astype(str).str.len()
            analysis["string_analysis"] = {
                "avg_length": string_lengths.mean(),
                "min_length": string_lengths.min(),
                "max_length": string_lengths.max(),
                "has_numbers": col_data.astype(str).str.contains(r'\d').any(),
                "has_special_chars": col_data.astype(str).str.contains(r'[^a-zA-Z0-9\s]').any()
            }
        
        return analysis
    
    def _calculate_entropy(self, value_counts: pd.Series) -> float:
        """Calculate entropy of a categorical distribution."""
        proportions = value_counts / value_counts.sum()
        return -np.sum(proportions * np.log2(proportions + 1e-10))
    
    def _find_strong_correlations(self, correlation_matrix: pd.DataFrame, threshold: float = 0.7) -> List[Dict]:
        """Find strong correlations between variables."""
        strong_correlations = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    strong_correlations.append({
                        "variable_1": correlation_matrix.columns[i],
                        "variable_2": correlation_matrix.columns[j],
                        "correlation": round(corr_value, 3),
                        "strength": "very strong" if abs(corr_value) >= 0.9 else "strong"
                    })
        
        return strong_correlations
    
    def _generate_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive statistical summary."""
        numeric_df = df.select_dtypes(include=[np.number])
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(numeric_df.columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime']).columns),
            "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            "duplicate_rows": df.duplicated().sum(),
            "unique_rows": len(df.drop_duplicates())
        }
        
        if len(numeric_df.columns) > 0:
            summary["numeric_summary"] = {
                "mean_values": numeric_df.mean().to_dict(),
                "median_values": numeric_df.median().to_dict(),
                "std_values": numeric_df.std().to_dict(),
                "range_values": (numeric_df.max() - numeric_df.min()).to_dict()
            }
        
        return summary
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality."""
        quality_score = 100
        issues = []
        
        # Check for missing data
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 10:
            quality_score -= 20
            issues.append(f"High missing data: {missing_percentage:.1f}%")
        elif missing_percentage > 5:
            quality_score -= 10
            issues.append(f"Moderate missing data: {missing_percentage:.1f}%")
        
        # Check for duplicates
        duplicate_percentage = (df.duplicated().sum() / len(df)) * 100
        if duplicate_percentage > 5:
            quality_score -= 15
            issues.append(f"High duplicate rows: {duplicate_percentage:.1f}%")
        
        # Check for constant columns
        constant_columns = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_columns:
            quality_score -= 10
            issues.append(f"Constant columns: {len(constant_columns)}")
        
        # Check for high cardinality columns
        high_cardinality = [col for col in df.columns if df[col].nunique() > len(df) * 0.9]
        if high_cardinality:
            quality_score -= 5
            issues.append(f"High cardinality columns: {len(high_cardinality)}")
        
        return {
            "quality_score": max(0, quality_score),
            "issues": issues,
            "recommendations": self._generate_quality_recommendations(df, issues)
        }
    
    def _generate_quality_recommendations(self, df: pd.DataFrame, issues: List[str]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        if "missing data" in " ".join(issues).lower():
            recommendations.append("Consider imputing missing values or removing rows/columns with high missing data")
        
        if "duplicate" in " ".join(issues).lower():
            recommendations.append("Remove duplicate rows to improve data quality")
        
        if "constant" in " ".join(issues).lower():
            recommendations.append("Remove constant columns as they provide no predictive value")
        
        if "cardinality" in " ".join(issues).lower():
            recommendations.append("Consider grouping high cardinality categorical variables")
        
        return recommendations
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect interesting patterns in the data."""
        patterns = {
            "trends": [],
            "seasonality": [],
            "anomalies": [],
            "business_insights": []
        }
        
        # Time series patterns
        datetime_cols = df.select_dtypes(include=['datetime']).columns
        if len(datetime_cols) > 0:
            patterns["time_patterns"] = self._analyze_time_patterns(df, datetime_cols[0])
        
        # Business patterns
        patterns["business_insights"] = self._extract_business_insights(df)
        
        return patterns
    
    def _analyze_time_patterns(self, df: pd.DataFrame, datetime_col: str) -> Dict[str, Any]:
        """Analyze time-based patterns."""
        df_time = df.copy()
        df_time[datetime_col] = pd.to_datetime(df_time[datetime_col])
        
        # Extract time components
        df_time['year'] = df_time[datetime_col].dt.year
        df_time['month'] = df_time[datetime_col].dt.month
        df_time['day_of_week'] = df_time[datetime_col].dt.dayofweek
        df_time['hour'] = df_time[datetime_col].dt.hour
        
        patterns = {
            "date_range": {
                "start": df_time[datetime_col].min().isoformat(),
                "end": df_time[datetime_col].max().isoformat(),
                "span_days": (df_time[datetime_col].max() - df_time[datetime_col].min()).days
            },
            "temporal_distribution": {
                "by_month": df_time['month'].value_counts().to_dict(),
                "by_day_of_week": df_time['day_of_week'].value_counts().to_dict(),
                "by_hour": df_time['hour'].value_counts().to_dict() if 'hour' in df_time.columns else {}
            }
        }
        
        return patterns
    
    def _extract_business_insights(self, df: pd.DataFrame) -> List[str]:
        """Extract business-relevant insights."""
        insights = []
        
        # Revenue/sales patterns
        revenue_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount', 'price', 'cost'])]
        if revenue_cols:
            for col in revenue_cols:
                if pd.api.types.is_numeric_dtype(df[col]):
                    total = df[col].sum()
                    mean = df[col].mean()
                    insights.append(f"Total {col}: ${total:,.2f}, Average: ${mean:,.2f}")
        
        # Customer/user patterns
        customer_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['customer', 'user', 'client'])]
        if customer_cols:
            for col in customer_cols:
                unique_count = df[col].nunique()
                insights.append(f"Unique {col}: {unique_count:,}")
        
        # Product patterns
        product_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['product', 'item', 'category'])]
        if product_cols:
            for col in product_cols:
                if df[col].dtype == 'object':
                    top_product = df[col].value_counts().index[0]
                    count = df[col].value_counts().iloc[0]
                    insights.append(f"Top {col}: {top_product} ({count:,} occurrences)")
        
        return insights
    
    def perform_clustering(self, file_path: str, file_type: str, n_clusters: int = None) -> Dict[str, Any]:
        """Perform clustering analysis on the dataset."""
        try:
            # Load data
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Prepare data for clustering
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df.columns) < 2:
                return {"error": "Not enough numeric columns for clustering"}
            
            # Scale the data
            X_scaled = self.scaler.fit_transform(numeric_df)
            
            # Determine optimal number of clusters if not provided
            if n_clusters is None:
                n_clusters = self._find_optimal_clusters(X_scaled)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Calculate silhouette score
            silhouette_avg = silhouette_score(X_scaled, clusters)
            
            # Analyze clusters
            cluster_analysis = self._analyze_clusters(df, clusters, numeric_df.columns)
            
            return {
                "n_clusters": n_clusters,
                "silhouette_score": silhouette_avg,
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "cluster_labels": clusters.tolist(),
                "cluster_analysis": cluster_analysis,
                "feature_importance": self._calculate_cluster_feature_importance(X_scaled, clusters)
            }
            
        except Exception as e:
            return {"error": f"Clustering analysis failed: {str(e)}"}
    
    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using elbow method."""
        if len(X) < 10:
            return 2
        
        max_clusters = min(max_clusters, len(X) // 2)
        inertias = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Find elbow point
        if len(inertias) >= 2:
            # Simple elbow detection
            diffs = np.diff(inertias)
            diff_diffs = np.diff(diffs)
            if len(diff_diffs) > 0:
                elbow_idx = np.argmax(diff_diffs) + 2
                return min(elbow_idx, max_clusters)
        
        return 3  # Default
    
    def _analyze_clusters(self, df: pd.DataFrame, clusters: np.ndarray, numeric_cols: List[str]) -> Dict[str, Any]:
        """Analyze cluster characteristics."""
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = clusters
        
        analysis = {}
        
        for cluster_id in range(len(np.unique(clusters))):
            cluster_data = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            analysis[f"cluster_{cluster_id}"] = {
                "size": len(cluster_data),
                "percentage": (len(cluster_data) / len(df)) * 100,
                "numeric_means": cluster_data[numeric_cols].mean().to_dict(),
                "characteristics": self._describe_cluster(cluster_data, numeric_cols)
            }
        
        return analysis
    
    def _describe_cluster(self, cluster_data: pd.DataFrame, numeric_cols: List[str]) -> str:
        """Generate a text description of cluster characteristics."""
        characteristics = []
        
        for col in numeric_cols:
            mean_val = cluster_data[col].mean()
            characteristics.append(f"{col}: {mean_val:.2f}")
        
        return ", ".join(characteristics)
    
    def _calculate_cluster_feature_importance(self, X: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance for clustering."""
        # Use PCA to understand feature importance
        pca = PCA()
        pca.fit(X)
        
        # Get the importance of each feature for the first two components
        feature_importance = {}
        for i in range(min(X.shape[1], 2)):
            component = pca.components_[i]
            for j, importance in enumerate(component):
                feature_name = f"feature_{j}"
                if feature_name not in feature_importance:
                    feature_importance[feature_name] = 0
                feature_importance[feature_name] += abs(importance)
        
        return feature_importance
    
    def detect_anomalies(self, file_path: str, file_type: str, contamination: float = 0.1) -> Dict[str, Any]:
        """Detect anomalies in the dataset using multiple methods."""
        try:
            # Load data
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            numeric_df = df.select_dtypes(include=[np.number]).dropna()
            
            if len(numeric_df.columns) == 0:
                return {"error": "No numeric columns found for anomaly detection"}
            
            # Isolation Forest
            isolation_forest = IsolationForest(contamination=contamination, random_state=42)
            anomaly_labels = isolation_forest.fit_predict(numeric_df)
            
            # Statistical outliers (Z-score method)
            z_scores = np.abs(stats.zscore(numeric_df))
            z_outliers = (z_scores > 3).any(axis=1)
            
            # IQR method
            iqr_outliers = np.zeros(len(numeric_df), dtype=bool)
            for col in numeric_df.columns:
                Q1 = numeric_df[col].quantile(0.25)
                Q3 = numeric_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                iqr_outliers |= (numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)
            
            # Combine results
            anomaly_indices = np.where(anomaly_labels == -1)[0]
            z_outlier_indices = np.where(z_outliers)[0]
            iqr_outlier_indices = np.where(iqr_outliers)[0]
            
            return {
                "isolation_forest_anomalies": {
                    "count": len(anomaly_indices),
                    "indices": anomaly_indices.tolist(),
                    "percentage": (len(anomaly_indices) / len(df)) * 100
                },
                "statistical_outliers": {
                    "count": len(z_outlier_indices),
                    "indices": z_outlier_indices.tolist(),
                    "percentage": (len(z_outlier_indices) / len(df)) * 100
                },
                "iqr_outliers": {
                    "count": len(iqr_outlier_indices),
                    "indices": iqr_outlier_indices.tolist(),
                    "percentage": (len(iqr_outlier_indices) / len(df)) * 100
                },
                "summary": self._summarize_anomalies(df, anomaly_indices, z_outlier_indices, iqr_outlier_indices)
            }
            
        except Exception as e:
            return {"error": f"Anomaly detection failed: {str(e)}"}
    
    def _summarize_anomalies(self, df: pd.DataFrame, *outlier_indices) -> Dict[str, Any]:
        """Summarize anomaly detection results."""
        all_outliers = set()
        for indices in outlier_indices:
            all_outliers.update(indices)
        
        if not all_outliers:
            return {"message": "No significant anomalies detected"}
        
        outlier_data = df.iloc[list(all_outliers)]
        
        return {
            "total_unique_outliers": len(all_outliers),
            "percentage_of_data": (len(all_outliers) / len(df)) * 100,
            "outlier_characteristics": self._describe_outliers(outlier_data)
        }
    
    def _describe_outliers(self, outlier_data: pd.DataFrame) -> Dict[str, Any]:
        """Describe characteristics of detected outliers."""
        numeric_cols = outlier_data.select_dtypes(include=[np.number]).columns
        
        characteristics = {}
        for col in numeric_cols:
            characteristics[col] = {
                "mean": outlier_data[col].mean(),
                "std": outlier_data[col].std(),
                "min": outlier_data[col].min(),
                "max": outlier_data[col].max()
            }
        
        return characteristics
    
    def generate_insights(self, file_path: str, file_type: str) -> Dict[str, Any]:
        """Generate comprehensive business insights."""
        try:
            # Load data
            if file_type == 'csv':
                df = pd.read_csv(file_path)
            elif file_type == 'excel':
                df = pd.read_excel(file_path)
            elif file_type == 'json':
                df = pd.read_json(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            insights = {
                "key_metrics": self._calculate_key_metrics(df),
                "trends": self._identify_trends(df),
                "recommendations": self._generate_recommendations(df),
                "risk_factors": self._identify_risk_factors(df),
                "opportunities": self._identify_opportunities(df)
            }
            
            return insights
            
        except Exception as e:
            return {"error": f"Insight generation failed: {str(e)}"}
    
    def _calculate_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate key business metrics."""
        metrics = {}
        
        # Revenue metrics
        revenue_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'amount'])]
        for col in revenue_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                metrics[f"total_{col}"] = df[col].sum()
                metrics[f"avg_{col}"] = df[col].mean()
                metrics[f"growth_rate_{col}"] = self._calculate_growth_rate(df[col])
        
        # Volume metrics
        metrics["total_records"] = len(df)
        metrics["data_completeness"] = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        
        return metrics
    
    def _calculate_growth_rate(self, series: pd.Series) -> float:
        """Calculate growth rate for a numeric series."""
        if len(series) < 2:
            return 0
        
        first_val = series.iloc[0]
        last_val = series.iloc[-1]
        
        if first_val == 0:
            return 0
        
        return ((last_val - first_val) / first_val) * 100
    
    def _identify_trends(self, df: pd.DataFrame) -> List[str]:
        """Identify trends in the data."""
        trends = []
        
        # Numeric trend analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if len(df[col].dropna()) > 1:
                # Simple trend detection
                values = df[col].dropna().values
                if len(values) >= 10:
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    if slope > 0:
                        trends.append(f"{col} shows an upward trend")
                    elif slope < 0:
                        trends.append(f"{col} shows a downward trend")
        
        return trends
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        # Data quality recommendations
        missing_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        if missing_percentage > 10:
            recommendations.append("Address data quality issues by reducing missing values")
        
        # Performance recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.lower() in ['revenue', 'sales', 'profit']:
                mean_val = df[col].mean()
                recommendations.append(f"Focus on improving {col} performance (current average: {mean_val:.2f})")
        
        return recommendations
    
    def _identify_risk_factors(self, df: pd.DataFrame) -> List[str]:
        """Identify potential risk factors."""
        risks = []
        
        # High variability in key metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col.lower() in ['revenue', 'sales', 'profit']:
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                if cv > 1:
                    risks.append(f"High variability in {col} (CV: {cv:.2f})")
        
        # Data quality risks
        if df.duplicated().sum() > len(df) * 0.1:
            risks.append("High number of duplicate records")
        
        return risks
    
    def _identify_opportunities(self, df: pd.DataFrame) -> List[str]:
        """Identify opportunities for improvement."""
        opportunities = []
        
        # Correlation opportunities
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = df[numeric_cols].corr()
            strong_corr = self._find_strong_correlations(corr_matrix, threshold=0.8)
            if strong_corr:
                opportunities.append(f"Leverage strong correlations between variables for predictive modeling")
        
        # Missing value opportunities
        high_missing_cols = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.5]
        if high_missing_cols:
            opportunities.append("Consider feature engineering to handle missing values")
        
        return opportunities
    
    def get_sample_data(self, file_path: str, file_type: str, n_rows: int = 5) -> Dict[str, Any]:
        """Get sample data for preview."""
        try:
            if file_type == 'csv':
                df = pd.read_csv(file_path).head(n_rows)
            elif file_type == 'excel':
                df = pd.read_excel(file_path).head(n_rows)
            elif file_type == 'json':
                df = pd.read_json(file_path).head(n_rows)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            return {
                "columns": df.columns.tolist(),
                "data": df.to_dict('records'),
                "dtypes": df.dtypes.astype(str).to_dict()
            }
            
        except Exception as e:
            raise Exception(f"Error getting sample data: {str(e)}")

# Maintain backward compatibility
class DataProcessor(AdvancedDataProcessor):
    """Backward compatible wrapper for AdvancedDataProcessor."""
    pass
