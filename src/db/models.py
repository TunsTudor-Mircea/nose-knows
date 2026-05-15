from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, ForeignKey, Integer, SmallInteger, Text, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Session(Base):
    __tablename__ = "sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=_now, nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), default=_now, nullable=False)

    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan")
    feedback = relationship("Feedback", back_populates="session", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(Text, nullable=False)       # 'user' | 'assistant'
    content = Column(Text, nullable=False)
    intent = Column(Text, nullable=True)
    hyde_doc = Column(Text, nullable=True)
    perfumes = Column(JSONB, nullable=True)   # list[PerfumeCard] as dicts
    latency_ms = Column(Integer, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=_now, nullable=False)

    session = relationship("Session", back_populates="messages")
    feedback = relationship("Feedback", back_populates="message", cascade="all, delete-orphan")


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
    )
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )
    score = Column(SmallInteger, nullable=False)  # -1 (thumbs down) or 1 (thumbs up)
    comment = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=_now, nullable=False)

    message = relationship("Message", back_populates="feedback")
    session = relationship("Session", back_populates="feedback")


class IngestJob(Base):
    __tablename__ = "ingest_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(Text, nullable=False)
    status = Column(Text, nullable=False, default="pending")  # pending|running|done|error
    total_rows = Column(Integer, nullable=True)
    processed_rows = Column(Integer, nullable=True)
    message = Column(Text, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=_now, nullable=False)
    completed_at = Column(TIMESTAMP(timezone=True), nullable=True)
