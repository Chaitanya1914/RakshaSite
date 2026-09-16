import datetime
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from .models import Violation, ShiftReport

def create_violation(session: Session, data: dict) -> Violation:
    violation = Violation(**data)
    session.add(violation)
    session.commit()
    session.refresh(violation)
    return violation

def get_violations_by_timerange(session: Session, start: datetime.datetime, end: datetime.datetime) -> List[Violation]:
    return session.query(Violation).filter(
        Violation.timestamp >= start,
        Violation.timestamp <= end
    ).all()

def get_violation_stats(session: Session) -> Dict[str, Any]:
    total_violations = session.query(Violation).count()
    
    type_counts_raw = session.query(
        Violation.violation_type, func.count(Violation.id)
    ).group_by(Violation.violation_type).all()
    type_counts = {t[0]: t[1] for t in type_counts_raw}
    
    avg_risk_score = session.query(func.avg(Violation.risk_score)).scalar() or 0.0
    
    estimated_detections = total_violations + 500
    compliance_score = max(0.0, 100.0 - (total_violations / estimated_detections * 100)) if estimated_detections > 0 else 100.0
    
    return {
        "total_violations": total_violations,
        "counts_by_type": type_counts,
        "average_risk_score": float(avg_risk_score),
        "compliance_score": compliance_score
    }

def get_repeat_offenders(session: Session, min_count: int = 3) -> List[Dict[str, Any]]:
    offenders_raw = session.query(
        Violation.worker_id, func.count(Violation.id).label('count')
    ).group_by(Violation.worker_id).having(func.count(Violation.id) >= min_count).all()
    
    return [{"worker_id": r[0], "violation_count": r[1]} for r in offenders_raw]
