import sys
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect, ForeignKey, Boolean
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# --- KONFIGURACJA BAZY ---
DATABASE_URL = "postgresql://postgres:admin@3.71.11.3:8542/NajsHajs"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- MODELE ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    admin = Column(Boolean)

    # Relacja z historią
    history = relationship("History", back_populates="user")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    knn_pred = Column(String)
    rf_pred = Column(String)
    svm_pred = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)

    # Relacja do User
    user = relationship("User", back_populates="history")


# --- FUNKCJE NARZĘDZIOWE ---
def get_db():
    """
    Dependency do użycia w FastAPI – zwraca sesję bazy danych.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_and_prepare_database():
    """
    Sprawdza połączenie z bazą danych, wylistowuje istniejące tabele
    i tworzy brakujące.
    """
    print("Sprawdzanie połączenia z bazą danych...")
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        print("✅ Połączenie z bazą danych udane!")
        print(f"   Znalezione tabele: {existing_tables}")

        required_tables = [User.__tablename__, History.__tablename__]
        missing_tables = [t for t in required_tables if t not in existing_tables]

        if missing_tables:
            print(f"   Brakuje tabel: {missing_tables}. Tworzenie...")
            Base.metadata.create_all(bind=engine)
            print("   Brakujące tabele zostały utworzone.")
        else:
            print("   Wszystkie wymagane tabele istnieją. Gotowe ✅")

    except OperationalError as e:
        print("❌ BŁĄD: Nie można połączyć się z bazą danych!", file=sys.stderr)
        print("   Upewnij się, że kontener Docker z PostgreSQL jest uruchomiony.", file=sys.stderr)
        print(f"   Szczegóły błędu: {e}", file=sys.stderr)
        raise
