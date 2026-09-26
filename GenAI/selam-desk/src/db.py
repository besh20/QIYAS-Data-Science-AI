"""SQLite storage for registrations. On Streamlit Cloud this file resets on
redeploy/restart -- acceptable for a demo, note it in the presentation.
For a persistent live deployment, swap DB_URL for a hosted free Postgres
(e.g. Supabase) connection string.
"""
import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "selam_desk.db")
DB_URL = f"sqlite:///{DB_PATH}"

Base = declarative_base()
engine = create_engine(DB_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)


class Registration(Base):
    __tablename__ = "registrations"
    id = Column(Integer, primary_key=True)
    full_name = Column(String, nullable=False)
    phone = Column(String, nullable=False)
    program = Column(String, nullable=False)
    date_of_birth = Column(String, nullable=True)  # stores "EC_YYYY-MM-DD | GC_YYYY-MM-DD"
    id_number = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Complaint(Base):
    __tablename__ = "complaints"
    id = Column(Integer, primary_key=True)
    summary = Column(String, nullable=False)
    student_name = Column(String, nullable=True)
    contact = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Escalation(Base):
    __tablename__ = "escalations"
    id = Column(Integer, primary_key=True)
    reason = Column(String, nullable=False)
    contact = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(engine)


def save_registration(data: dict):
    init_db()
    session = SessionLocal()
    reg = Registration(**data)
    session.add(reg)
    session.commit()
    session.refresh(reg)
    session.close()
    return reg.id


def save_complaint(summary: str, student_name: str = "", contact: str = ""):
    init_db()
    session = SessionLocal()
    c = Complaint(summary=summary, student_name=student_name or None, contact=contact or None)
    session.add(c)
    session.commit()
    session.refresh(c)
    session.close()
    return c.id


def save_escalation(reason: str, contact: str = ""):
    init_db()
    session = SessionLocal()
    e = Escalation(reason=reason, contact=contact or None)
    session.add(e)
    session.commit()
    session.refresh(e)
    session.close()
    return e.id


if __name__ == "__main__":
    init_db()
    print("DB initialized at", DB_PATH)
