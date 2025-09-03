from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import func

from sqlalchemy.orm import Session
from database import get_db, History, User, Banknote

router = APIRouter(prefix="/admin", tags=["admin"])


class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    knn_pred: str
    rf_pred: str
    svm_pred: str
    user_id: int | None

    class Config:
        from_attributes = True


@router.get("/")
def read_root():
    return {"message": "Hello from Admin API!"}


@router.get("/history/all", response_model=list[HistoryItem])
def get_all_history(db: Session = Depends(get_db)):
    history = db.query(History).order_by(History.timestamp.desc()).all()
    return history


@router.get("/user/{username}")
def get_user_id_by_name(username: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{username}' not found")

    return {"user_id": user.id, "username": user.username}


def check_database_health(db: Session) -> str:
    """
    Sprawdza status zdrowia bazy danych.
    """
    try:

        # Sprawdzenie czy główne tabele istnieją i są dostępne
        db.query(User).first()
        db.query(History).first()
        db.query(Banknote).first()

        return "on"
    except Exception as e:
        print(f"Database health check failed: {e}")
        return "off"


@router.get("/dashboard")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """
    Dashboard dla admina – zwraca statystyki zgodne z wymaganym formatem JSON.
    """
    try:
        # Sprawdzenie statusu serwera bazy danych
        server_status = check_database_health(db)

        # Jeśli baza jest niedostępna, zwróć podstawowe informacje
        if server_status == "off":
            return {
                "history_count": 0,
                "users_count": 0,
                "countries": 0,
                "server_status": server_status
            }

        # Liczba rekordów w tabeli History
        history_count = db.query(History).count()

        # Liczba rekordów w tabeli User
        users_count = db.query(User).count()

        # Liczba unikalnych krajów w tabeli Banknote
        countries_count = db.query(func.count(func.distinct(Banknote.country))).scalar()

        return {
            "history_count": history_count,
            "users_count": users_count,
            "countries": countries_count,
            "server_status": server_status
        }

    except Exception as e:
        # W przypadku krytycznego błędu
        return {
            "history_count": 0,
            "users_count": 0,
            "countries": 0,
            "server_status": "off"
        }