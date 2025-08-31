from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import base64
from datetime import datetime

# --- IMPORTY BAZY ---
from sqlalchemy.orm import Session
from database import get_db, check_and_prepare_database, User, History
import database as db

# Importujemy funkcje ML
from classify import procesIMG, load_models


# --- Aplikacja i Startup Event ---
app = FastAPI()


@app.on_event("startup")
async def startup_event():
    load_models()
    check_and_prepare_database()  # Tworzymy bazę i tabele przy starcie


# --- Konfiguracja CORS ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Modele Pydantic ---
class ImageData(BaseModel):
    image: str  # base64 string


class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    knn_pred: str
    rf_pred: str
    svm_pred: str
    user_id: int | None   # <-- może być None

    class Config:
        from_attributes = True


# --- Endpointy API ---
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Models are loaded and ready."}


@app.post("/api/upload")
async def upload_image(data: ImageData, db: Session = Depends(get_db)):
    try:
        # --- Dekodowanie obrazu ---
        image_data = data.image.split(',')[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- Klasyfikacja ---
        try:
            classification_results = procesIMG(image)
        except Exception as e:
            raise HTTPException(status_code=520, detail=str(e))

        # --- Zapis do bazy z user_id=None ---
        try:
            db_record = History(
                knn_pred=classification_results["knn"]["pred"],
                rf_pred=classification_results["rf"]["pred"],
                svm_pred=classification_results["svm"]["pred"],
                user_id=None
            )
        except Exception as e:
            raise HTTPException(status_code=530, detail=str(e))
        try:
            db.add(db_record)
            db.commit()
            db.refresh(db_record)
        except Exception as e:
            raise HTTPException(status_code=540, detail=str(e))

        return classification_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/history", response_model=list[HistoryItem])
def get_history(db: Session = Depends(get_db)):
    """
    Pobiera historię wszystkich przetworzonych obrazów.
    """
    history = db.query(History).order_by(History.timestamp.desc()).all()
    return history
