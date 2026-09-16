import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

Base = declarative_base()

class Violation(Base):
    __tablename__ = 'violations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    camera_id = Column(String(50))
    worker_id = Column(String(50))
    violation_type = Column(String(100))
    confidence = Column(Float)
    risk_score = Column(Float)
    risk_label = Column(String(20))
    bocw_section = Column(String(50))
    penalty = Column(Text)
    zone = Column(String(50))
    screenshot_path = Column(String(255))
    report_text = Column(Text)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ShiftReport(Base):
    __tablename__ = 'shift_reports'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    shift_start = Column(DateTime)
    shift_end = Column(DateTime)
    camera_id = Column(String(50))
    total_violations = Column(Integer)
    compliance_score = Column(Float)
    report_pdf_path = Column(String(255))
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

def init_db(db_url: str) -> sessionmaker:
    engine = create_engine(db_url)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)

def save_violation(session: Session, violation_data: dict) -> Violation:
    violation = Violation(**violation_data)
    session.add(violation)
    session.commit()
    session.refresh(violation)
    return violation

def get_violations(session: Session, camera_id: str, start_time: datetime.datetime, end_time: datetime.datetime) -> List[Violation]:
    return session.query(Violation).filter(
        Violation.camera_id == camera_id,
        Violation.timestamp >= start_time,
        Violation.timestamp <= end_time
    ).all()

def get_compliance_score(session: Session, camera_id: str, shift_start: datetime.datetime) -> float:
    violations_count = session.query(Violation).filter(
        Violation.camera_id == camera_id,
        Violation.timestamp >= shift_start
    ).count()
    
    if violations_count == 0:
        return 100.0
    
    # Simulating a total detection count as violations + 100 for calculation
    estimated_detections = violations_count + 100
    compliance = 100.0 - (violations_count / estimated_detections * 100.0)
    return max(0.0, compliance)
