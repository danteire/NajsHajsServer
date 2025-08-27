from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import io
import base64
from datetime import datetime

# --- NOWE IMPORTY ---
from sqlalchemy.orm import Session
import database as db  # Importujemy nasz plik database.py
from database import ProcessedImage  # Importujemy model tabeli

# Importujemy poprawione funkcje
from classify import procesIMG, load_models

# --- Aplikacja i Startup Event ---
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    load_models()
    db.check_and_prepare_database()  # Tworzymy bazę i tabele przy starcie


# --- Konfiguracja CORS (bez zmian) ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Zależność (Dependency) do obsługi sesji bazy danych ---
def get_db():
    database = db.SessionLocal()
    try:
        yield database
    finally:
        database.close()


# --- Modele Pydantic ---
class ImageData(BaseModel):
    image: str  # base64 string


# NOWY model Pydantic do zwracania danych z historii
class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    knn_pred: str
    rf_pred: str
    svm_pred: str

    class Config:
        from_attributes = True  # Pozwala na konwersję z obiektu SQLAlchemy


# --- Endpointy API ---
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Models are loaded and ready."}


# ZMODYFIKOWANY endpoint upload
@app.post("/api/upload")
async def upload_image(data: ImageData, db: Session = Depends(get_db)):
    try:
        # Dekodowanie obrazu (bez zmian)
        image_data = data.image.split(',')[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Przetwarzanie obrazu (bez zmian)
        classification_results = procesIMG(image)

        # --- NOWA CZĘŚĆ: ZAPIS DO BAZY DANYCH ---
        db_record = ProcessedImage(
            knn_pred=classification_results['knn']['pred'],
            rf_pred=classification_results['rf']['pred'],
            svm_pred=classification_results['svm']['pred']
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)
        # --- KONIEC NOWEJ CZĘŚCI ---

        # Zwróć wyniki klasyfikacji jak poprzednio
        return classification_results

    except Exception as e:
        return {"error": str(e)}, 500


# --- NOWY ENDPOINT DO POBIERANIA HISTORII ---
@app.get("/api/history", response_model=list[HistoryItem])
def get_history(db: Session = Depends(get_db)):
    """
    Pobiera historię wszystkich przetworzonych obrazów.
    """
    history = db.query(ProcessedImage).order_by(ProcessedImage.timestamp.desc()).all()
    return history