from xmlrpc.client import SERVER_ERROR

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from datetime import datetime
from sqlalchemy import func

import traceback

from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from database import get_db, History, User, Banknote

STATUS_ERROR = -1
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


class AdminUpdateRequest(BaseModel):
    admin: bool


class CreateUserRequest(BaseModel):
    username: str
    password: str
    admin: bool = False


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

@router.get("/users")
def get_users(db: Session = Depends(get_db)):
    """
    zwraca listę wszystkich użytkowników w bazie
    """
    try:
        server_status = check_database_health(db)
        if server_status == "off":
            return {
                "server_status": server_status,
            }
        return db.query(User).all()
    except Exception as e:
        return {
            "server_status": STATUS_ERROR
        }


@router.delete("/users/{userID}")
def delete_user(userID: int, db: Session = Depends(get_db)):
    try:
        print(f"Attempting to delete user with ID: {userID}")

        # Sprawdź czy użytkownik istnieje
        user = db.query(User).filter(User.id == userID).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with id {userID} not found"
            )

        print(f"Found user: {user.username}")

        # Ustaw user_id na NULL w tabeli history (zamiast usuwania rekordów)
        updated_count = db.query(History).filter(History.user_id == userID).update(
            {"user_id": None},
            synchronize_session=False  # Dodaj to dla lepszej wydajności
        )
        print(f"Updated {updated_count} history records to remove user reference")

        # Teraz usuń użytkownika
        deleted_count = db.query(User).filter(User.id == userID).delete()
        print(f"Deleted {deleted_count} user records")

        db.commit()
        print("User deletion committed successfully")

        return {
            "success": True,
            "message": f"User '{user.username}' has been successfully deleted",
            "deleted_user_id": userID,
            "updated_history_records": updated_count
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error deleting user: {error_msg}")

        # Sprawdź czy to nadal problem z foreign key
        if "ForeignKeyViolation" in error_msg or "klucz obcy" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete user due to database constraints. User may have additional dependencies."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the user"
        )


# Opcjonalnie: Endpoint do pobrania szczegółów użytkownika
@router.get("/users/{userID}")
def get_user_details(userID: int, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == userID).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with id {userID} not found"
            )

        return {
            "id": user.id,
            "username": user.username,
            "admin": user.admin,
            "created_at": user.created_at
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching user details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching user details"
        )


@router.post("/users")
def create_user(user_data: CreateUserRequest, db: Session = Depends(get_db)):
    try:
        print(f"Attempting to create user: {user_data.username}")
        print(f"Admin privileges: {user_data.admin}")

        # Walidacja danych
        if len(user_data.username.strip()) < 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username must be at least 3 characters long"
            )

        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )

        # Sprawdź czy użytkownik już istnieje
        existing_user = db.query(User).filter(User.username == user_data.username.strip()).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with username '{user_data.username}' already exists"
            )

        # Utwórz nowego użytkownika
        new_user = User(
            username=user_data.username.strip(),
            password=user_data.password,  # W produkcji: zahashuj hasło!
            admin=user_data.admin
        )

        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        print(f"User '{new_user.username}' created successfully with ID: {new_user.id}")

        return {
            "id": new_user.id,
            "username": new_user.username,
            "password": new_user.password,  # Zwracane tylko dla spójności z istniejącym API
            "admin": new_user.admin,
            "created_at": new_user.created_at,
            "message": f"User '{new_user.username}' created successfully"
        }

    except HTTPException:
        raise
    except IntegrityError as e:
        db.rollback()
        print(f"IntegrityError creating user: {str(e)}")

        # Sprawdź czy to problem z duplikacją nazwy użytkownika
        if "unique" in str(e).lower() or "duplicate" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with username '{user_data.username}' already exists"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid user data"
            )
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error creating user: {error_msg}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the user"
        )


@router.patch("/users/{userID}/admin")
def update_user_admin_privileges(
        userID: int,
        admin_data: AdminUpdateRequest,
        db: Session = Depends(get_db)
):
    try:
        print(f"Attempting to update admin privileges for user ID: {userID}")
        print(f"New admin status: {admin_data.admin}")

        # Sprawdź czy użytkownik istnieje
        user = db.query(User).filter(User.id == userID).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with id {userID} not found"
            )

        print(f"Found user: {user.username}, current admin status: {user.admin}")

        # Sprawdź czy zmiana jest rzeczywiście potrzebna
        if user.admin == admin_data.admin:
            return {
                "success": True,
                "message": f"User '{user.username}' already has admin status: {admin_data.admin}",
                "user_id": userID,
                "username": user.username,
                "admin": user.admin
            }

        # Zaktualizuj status administratora
        user.admin = admin_data.admin
        db.commit()
        db.refresh(user)  # Odśwież obiekt żeby mieć aktualne dane

        action = "granted" if admin_data.admin else "revoked"
        print(f"Admin privileges {action} for user {user.username}")

        return {
            "success": True,
            "message": f"Admin privileges successfully {action} for user '{user.username}'",
            "user_id": userID,
            "username": user.username,
            "admin": user.admin,
            "previous_admin": not admin_data.admin
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error updating admin privileges: {error_msg}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating admin privileges"
        )


# Opcjonalnie: Endpoint do pobrania szczegółów użytkownika
@router.get("/users/{userID}")
def get_user_details(userID: int, db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.id == userID).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User with id {userID} not found"
            )

        return {
            "id": user.id,
            "username": user.username,
            "admin": user.admin,
            "created_at": user.created_at
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error fetching user details: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while fetching user details"
        )