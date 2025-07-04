from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    password: Optional[str] = None

class User(UserBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True

# Dataset Schemas
class DatasetBase(BaseModel):
    name: str
    description: Optional[str] = None

class DatasetCreate(DatasetBase):
    pass

class DatasetUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None

class Dataset(DatasetBase):
    id: int
    file_path: str
    file_type: str
    file_size: int
    columns_info: Optional[Dict[str, Any]] = None
    row_count: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    owner_id: int
    
    class Config:
        from_attributes = True

# Query Schemas
class QueryBase(BaseModel):
    question: str
    dataset_id: Optional[int] = None

class QueryCreate(QueryBase):
    pass

class Query(QueryBase):
    id: int
    answer: Optional[str] = None
    query_type: str
    execution_time: Optional[int] = None
    created_at: datetime
    user_id: int
    
    class Config:
        from_attributes = True

# Report Schemas
class ReportBase(BaseModel):
    title: str
    description: Optional[str] = None
    report_type: str
    scheduled: bool = False
    schedule_frequency: Optional[str] = None

class ReportCreate(ReportBase):
    dataset_id: int

class ReportUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    scheduled: Optional[bool] = None
    schedule_frequency: Optional[str] = None

class Report(ReportBase):
    id: int
    content: Optional[str] = None
    charts_data: Optional[Dict[str, Any]] = None
    status: str
    created_at: datetime
    updated_at: datetime
    user_id: int
    dataset_id: int
    
    class Config:
        from_attributes = True

# Alert Schemas
class AlertBase(BaseModel):
    title: str
    message: str
    alert_type: str
    severity: str = "medium"

class AlertCreate(AlertBase):
    dataset_id: int

class Alert(AlertBase):
    id: int
    is_read: bool
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    user_id: int
    dataset_id: int
    
    class Config:
        from_attributes = True

# User Preferences Schemas
class UserPreferencesBase(BaseModel):
    email_notifications: bool = True
    alert_frequency: str = "immediate"
    preferred_chart_types: Optional[List[str]] = None
    theme: str = "dark"

class UserPreferencesCreate(UserPreferencesBase):
    pass

class UserPreferencesUpdate(BaseModel):
    email_notifications: Optional[bool] = None
    alert_frequency: Optional[str] = None
    preferred_chart_types: Optional[List[str]] = None
    theme: Optional[str] = None

class UserPreferences(UserPreferencesBase):
    id: int
    created_at: datetime
    updated_at: datetime
    user_id: int
    
    class Config:
        from_attributes = True

# Auth Schemas
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# Chat Schemas
class ChatMessage(BaseModel):
    message: str
    dataset_id: Optional[int] = None

class ChatResponse(BaseModel):
    response: str
    charts: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None
    execution_time: Optional[int] = None

# File Upload Schemas
class FileUploadResponse(BaseModel):
    dataset_id: int
    message: str
    file_info: Dict[str, Any]
