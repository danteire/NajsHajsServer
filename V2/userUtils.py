from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import base64
from datetime import datetime

from sqlalchemy.orm import Session
from database import get_db, History
from classify import procesIMG

router = APIRouter(prefix="/user", tags=["user"])


class ImageData(BaseModel):
    image: str  # base64 string


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
    return {"message": "Hello from User API!"}


@router.post("/upload")
async def upload_image(data: ImageData, db: Session = Depends(get_db)):
    try:
        image_data = data.image.split(',')[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        classification_results = procesIMG(image)

        db_record = History(
            knn_pred=classification_results["knn"]["pred"],
            rf_pred=classification_results["rf"]["pred"],
            svm_pred=classification_results["svm"]["pred"],
            user_id=None
        )
        db.add(db_record)
        db.commit()
        db.refresh(db_record)

        return classification_results

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history", response_model=list[HistoryItem])
def get_history(db: Session = Depends(get_db)):
    history = db.query(History).order_by(History.timestamp.desc()).all()
    return history
