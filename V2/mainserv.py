# mainserv.py

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import io
import base64

# Importujemy poprawione funkcje
from classify import procesIMG, load_models

app = FastAPI()


# --- Ładowanie modeli przy starcie aplikacji ---
@app.on_event("startup")
async def startup_event():
    load_models()


# --- Konfiguracja CORS (bez zmian) ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Definicja modelu danych wejściowych (bez zmian) ---
class ImageData(BaseModel):
    image: str  # base64 string


# --- Endpointy API ---
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Models are loaded and ready."}


@app.post("/api/upload")
async def upload_image(data: ImageData):
    try:
        # Odczytaj dane base64 i przekonwertuj na obraz PIL
        image_data = data.image.split(',')[-1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Wywołaj funkcję przetwarzającą obraz
        classification_results = procesIMG(image)

        # Zwróć wyniki klasyfikacji jako odpowiedź JSON
        return classification_results

    except Exception as e:
        return {"error": str(e)}, 500