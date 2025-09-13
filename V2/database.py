import sys
from datetime import datetime
import pytz

from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect, ForeignKey, Boolean, Text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, declarative_base, relationship

# --- KONFIGURACJA BAZY ---
DATABASE_URL = "postgresql://postgres:admin@localhost:5432/NajsHajs"

# Funkcja do pobierania czasu polskiego
def get_polish_time():

    poland_tz = pytz.timezone('Europe/Warsaw')
    return datetime.now(poland_tz)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- MODELE ---
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=get_polish_time)

    # Relacja z historią
    history = relationship("History", back_populates="user")


class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=get_polish_time)
    knn_pred = Column(String)
    rf_pred = Column(String)
    svm_pred = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    image = Column(String, nullable=True)

    # Relacja do User
    user = relationship("User", back_populates="history")


class Banknote(Base):
    __tablename__ = "banknotes"

    id = Column(Integer, primary_key=True, index=True)
    country = Column(String(100), nullable=False)
    currency = Column(String(100), nullable=False)
    denomination = Column(String(50), nullable=False)
    effigy = Column(String(255))
    dimensions = Column(String(50))
    description = Column(Text)
    image_avers = Column(String(255), nullable=False)
    image_rewers = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=get_polish_time)


# --- FUNKCJE NARZĘDZIOWE ---
def get_db():
   
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def check_and_prepare_database():
    
    print("Sprawdzanie połączenia z bazą danych...")
    try:
        inspector = inspect(engine)
        existing_tables = inspector.get_table_names()

        print("✅ Połączenie z bazą danych udane!")
        print(f"   Znalezione tabele: {existing_tables}")

        required_tables = [User.__tablename__, History.__tablename__, Banknote.__tablename__]
        missing_tables = [t for t in required_tables if t not in existing_tables]

        if missing_tables:
            print(f"   Brakuje tabel: {missing_tables}. Tworzenie...")
            Base.metadata.create_all(bind=engine)
            print("   Brakujące tabele zostały utworzone.")
        else:
            print("   Wszystkie wymagane tabele istnieją. Gotowe ✅")

    except OperationalError as e:
        print(" BŁĄD: Nie można połączyć się z bazą danych!", file=sys.stderr)
        print("   Upewnij się, że kontener Docker z PostgreSQL jest uruchomiony.", file=sys.stderr)
        print(f"   Szczegóły błędu: {e}", file=sys.stderr)
        raise