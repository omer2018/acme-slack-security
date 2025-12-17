import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "sqlite:///./acme_bot.db"

Base = declarative_base()

class Thread(Base):
    __tablename__ = "threads"
    thread_id = Column(String, primary_key=True, index=True)
    slack_thread_ts = Column(String, unique=True, index=True)
    status = Column(String, default="intake")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    role = Column(String)
    text = Column(Text)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class Decision(Base):
    __tablename__ = "decisions"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, unique=True, index=True)
    request_type = Column(String, nullable=True)
    request_summary = Column(Text, nullable=True)
    risk_score = Column(Integer, nullable=True)
    extracted_fields = Column(JSON, nullable=True)
    missing_fields = Column(JSON, nullable=True)
    mandatory_fields = Column(JSON, nullable=True)
    outcome = Column(String, nullable=True)
    rationale = Column(Text, nullable=True)
    confidence_score = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True, index=True)
    thread_id = Column(String, index=True)
    action = Column(String)
    input_data = Column(JSON)
    output_data = Column(JSON)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

class HistoricalTicket(Base):
    __tablename__ = "historical_tickets"
    ticket_id = Column(String, primary_key=True, index=True)
    request_type = Column(String, index=True)
    request_summary = Column(Text)
    details = Column(Text)
    mandatory_fields = Column(String)
    fields_provided = Column(String)
    outcome = Column(String, index=True)
    security_risk_score = Column(Integer)
    embedding = Column(JSON)
    created_at = Column(DateTime)
    requester_department = Column(String)
    requester_title = Column(String)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def initialize_database():
    Base.metadata.create_all(bind=engine)