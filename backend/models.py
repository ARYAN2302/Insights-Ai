from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    datasets = relationship("Dataset", back_populates="owner")
    queries = relationship("Query", back_populates="user")
    reports = relationship("Report", back_populates="user")
    alerts = relationship("Alert", back_populates="user")

class Dataset(Base):
    __tablename__ = "datasets"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)  # csv, excel, json
    file_size = Column(Integer, nullable=False)
    columns_info = Column(JSON, nullable=True)  # Column names, types, stats
    row_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    owner_id = Column(Integer, ForeignKey("users.id"))
    
    # Relationships
    owner = relationship("User", back_populates="datasets")
    queries = relationship("Query", back_populates="dataset")
    reports = relationship("Report", back_populates="dataset")
    alerts = relationship("Alert", back_populates="dataset")

class Query(Base):
    __tablename__ = "queries"
    
    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=True)
    query_type = Column(String(50), default="chat")  # chat, analysis, report
    execution_time = Column(Integer, nullable=True)  # in milliseconds
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="queries")
    dataset = relationship("Dataset", back_populates="queries")

class Report(Base):
    __tablename__ = "reports"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    content = Column(Text, nullable=True)  # Generated report content
    charts_data = Column(JSON, nullable=True)  # Chart configurations
    report_type = Column(String(50), nullable=False)  # summary, trend, custom
    status = Column(String(50), default="pending")  # pending, completed, failed
    scheduled = Column(Boolean, default=False)
    schedule_frequency = Column(String(50), nullable=True)  # daily, weekly, monthly
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relationships
    user = relationship("User", back_populates="reports")
    dataset = relationship("Dataset", back_populates="reports")

class Alert(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    message = Column(Text, nullable=False)
    alert_type = Column(String(50), nullable=False)  # anomaly, trend, insight
    severity = Column(String(50), default="medium")  # low, medium, high
    is_read = Column(Boolean, default=False)
    metadata_json = Column(JSON, nullable=True)  # Additional alert data
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"))
    dataset_id = Column(Integer, ForeignKey("datasets.id"))
    
    # Relationships
    user = relationship("User", back_populates="alerts")
    dataset = relationship("Dataset", back_populates="alerts")

class UserPreferences(Base):
    __tablename__ = "user_preferences"
    
    id = Column(Integer, primary_key=True, index=True)
    email_notifications = Column(Boolean, default=True)
    alert_frequency = Column(String(50), default="immediate")  # immediate, daily, weekly
    preferred_chart_types = Column(JSON, nullable=True)
    theme = Column(String(50), default="dark")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Foreign Keys
    user_id = Column(Integer, ForeignKey("users.id"))
