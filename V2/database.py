from sqlalchemy import create_engine, Column, Integer, String, DateTime, inspect
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import OperationalError
from datetime import datetime
import sys
# 1. Zmiana adresu URL na PostgreSQL
# Format: postgresql://<użytkownik>:<hasło>@<host>:<port>/<nazwa_bazy>
DATABASE_URL = "postgresql://user:password@localhost:5432/najs_hajs_db"

# Tworzymy "silnik" bazy danych - usuwamy connect_args, które było dla SQLite
engine = create_engine(DATABASE_URL)

# Reszta pliku pozostaje bez zmian
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class ProcessedImage(Base):
    __tablename__ = "processed_images"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    knn_pred = Column(String)
    rf_pred = Column(String)
    svm_pred = Column(String)


# --- NOWA, BARDZIEJ ZAAWANSOWANA FUNKCJA ---
def check_and_prepare_database():
    """
    Sprawdza połączenie z bazą danych, wylistowuje istniejące tabele
    i tworzy wymaganą tabelę, jeśli jej brakuje.
    """
    print("Sprawdzanie połączenia z bazą danych...")
    try:
        # Utwórz inspektora, który pozwoli nam zajrzeć do bazy danych
        inspector = inspect(engine)

        # Pobierz nazwy tabel - ta operacja jednocześnie testuje połączenie
        existing_tables = inspector.get_table_names()

        print("✅ Połączenie z bazą danych udane!")
        print(f"   Znalezione tabele: {existing_tables}")

        # Sprawdź, czy nasza tabela już istnieje
        if ProcessedImage.__tablename__ not in existing_tables:
            print(f"   Tabela '{ProcessedImage.__tablename__}' nie istnieje. Tworzenie...")
            # Utwórz tylko tabele zdefiniowane w tym pliku (dziedziczące po Base)
            Base.metadata.create_all(bind=engine)
            print(f"   Tabela '{ProcessedImage.__tablename__}' została utworzona.")
        else:
            print(f"   Tabela '{ProcessedImage.__tablename__}' już istnieje. Wszystko gotowe.")

    except OperationalError as e:
        print("❌ BŁĄD: Nie można połączyć się z bazą danych!", file=sys.stderr)
        print("   Upewnij się, że kontener Docker z PostgreSQL jest uruchomiony.", file=sys.stderr)
        print(f"   Szczegóły błędu: {e}", file=sys.stderr)
        # Rzucamy błąd dalej, aby zatrzymać start aplikacji, co jest pożądanym zachowaniem
        raise