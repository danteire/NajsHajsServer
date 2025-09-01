from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from PIL import Image
import io
import base64
from datetime import datetime, timedelta
from typing import Optional
import jwt
from passlib.context import CryptContext
import hashlib

# --- IMPORTY BAZY ---
from sqlalchemy.orm import Session
from database import get_db, check_and_prepare_database, History
import database as db

# Importujemy funkcje ML
from classify import procesIMG, load_models

# --- Konfiguracja JWT i haszowania ---
SECRET_KEY = "your-secret-key-change-this-in-production"  # ZMIEŃ TO W PRODUKCJI!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()


# --- Modele Pydantic ---
class ImageData(BaseModel):
    image: str  # base64 string


class UserRegister(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    username: str


class UserResponse(BaseModel):
    user_id: int
    username: str
    admin: bool


# --- Aplikacja i Startup Event ---
app = FastAPI()

# --- Konfiguracja CORS ---
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://3.71.11.3:2115",
    "http://3.71.11.3:8543",
    "http://najshajs.mywire.org:8543",
    "http://najshajs.mywire.org",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    load_models()
    check_and_prepare_database()


class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    knn_pred: str
    rf_pred: str
    svm_pred: str
    user_id: Optional[int] = None
    predicted: str

    class Config:
        from_attributes = True


# --- Funkcje pomocnicze JWT ---
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), db: Session = Depends(get_db)):
    """
    Dekoduje token JWT i zwraca aktualnego użytkownika.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(db.User).filter(db.User.username == username).first()
    if user is None:
        raise credentials_exception
    return user


def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        db: Session = Depends(get_db)):
    """
    Opcjonalna autentykacja - zwraca użytkownika jeśli token jest poprawny, None w przeciwnym razie.
    """
    if credentials is None:
        return None

    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
    except jwt.PyJWTError:
        return None

    user = db.query(db.User).filter(db.User.username == username).first()
    return user


# --- Endpointy API ---
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Models are loaded and ready."}


@app.post("/api/register", response_model=Token)
async def register_user(user_data: UserRegister, db: Session = Depends(get_db)):
    """
    Rejestracja nowego użytkownika.
    """
    # Sprawdź czy użytkownik już istnieje
    existing_user = db.query(db.User).filter(db.User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Username already registered"
        )

    # Stwórz nowego użytkownika
    hashed_password = get_password_hash(user_data.password)
    db_user = db.User(
        username=user_data.username,
        password=hashed_password,
        admin=False
    )

    try:
        db.add(db_user)
        db.commit()
        db.refresh(db_user)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Stwórz token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": db_user.id,
        "username": db_user.username
    }


@app.post("/api/login", response_model=Token)
async def login_user(user_data: UserLogin, db: Session = Depends(get_db)):
    """
    Logowanie użytkownika.
    """
    user = db.query(db.User).filter(db.User.username == user_data.username).first()

    if not user or not verify_password(user_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }


@app.get("/api/me", response_model=UserResponse)
async def get_current_user_info(current_user: db.User = Depends(get_current_user)):
    """
    Pobiera informacje o aktualnie zalogowanym użytkowniku.
    """
    return {
        "user_id": current_user.id,
        "username": current_user.username,
        "admin": current_user.admin
    }


@app.post("/api/upload")
async def upload_image(data: ImageData, current_user: Optional[db.User] = Depends(get_current_user_optional),
                       db: Session = Depends(get_db)):
    """
    Upload obrazu - obsługuje zarówno zalogowanych użytkowników jak i gości.
    """
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

        # --- Zapis do bazy z user_id z tokenu lub None dla gościa ---
        try:
            predicted_class = classification_results["rf"]["pred"]

            db_record = History(
                knn_pred=classification_results["knn"]["pred"],
                rf_pred=classification_results["rf"]["pred"],
                svm_pred=classification_results["svm"]["pred"],
                user_id=current_user.id if current_user else None,
                predicted=predicted_class
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
def get_history(current_user: db.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Pobiera historię dla zalogowanego użytkownika.
    """
    history = db.query(History).filter(History.user_id == current_user.id).order_by(History.timestamp.desc()).all()
    return history


@app.get("/api/history/all", response_model=list[HistoryItem])
def get_all_history(current_user: db.User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Pobiera całą historię - tylko dla adminów.
    """
    if not current_user.admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    history = db.query(History).order_by(History.timestamp.desc()).all()
    return history


@app.get("/api/user/{username}")
def get_user_id_by_name(username: str, db: Session = Depends(get_db)):
    """
    Pobiera informacje o użytkowniku na podstawie nazwy (bez wrażliwych danych).
    """
    try:
        user = db.query(db.User).filter(db.User.username == username).first()

        if not user:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")

        return {"user_id": user.id, "username": user.username}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))