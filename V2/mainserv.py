from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import io
import base64
import os
import uuid
from datetime import datetime, timedelta
import pytz
from typing import List
from sqlalchemy import func, desc

# --- IMPORTY BAZY ---
from sqlalchemy.orm import Session

# Funkcja do pobierania czasu polskiego
def get_polish_time():
    """Zwraca aktualny czas w strefie czasowej Polski"""
    poland_tz = pytz.timezone('Europe/Warsaw')
    return datetime.now(poland_tz)
from database import get_db, check_and_prepare_database, User, History, Banknote
import database as db

# Importujemy funkcje ML
from classify import procesIMG, load_models

# Importujemy funkcje autoryzacji
import auth

# --- Modele Pydantic ---
class ImageData(BaseModel):
    image: str  # base64 string

class Token(BaseModel):
    access_token: str
    token_type: str

class UserCreate(BaseModel):
    username: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    admin: bool | None
    
    class Config:
        from_attributes = True

class BanknoteCreate(BaseModel):
    country: str
    currency: str
    denomination: str
    effigy: str | None = None
    dimensions: str | None = None
    description: str | None = None
    image_avers: str
    image_rewers: str

class BanknoteUpdate(BaseModel):
    country: str | None = None
    currency: str | None = None
    denomination: str | None = None
    effigy: str | None = None
    dimensions: str | None = None
    description: str | None = None
    image_avers: str | None = None
    image_rewers: str | None = None

class BanknoteResponse(BaseModel):
    id: int
    country: str
    currency: str
    denomination: str
    effigy: str | None
    dimensions: str | None
    description: str | None
    image_avers: str
    image_rewers: str
    created_at: datetime
    
    class Config:
        from_attributes = True

# --- FUNKCJE POMOCNICZE ---
def save_base64_image(base64_string, folder_path="C:/xampp/htdocs/react_front_app/resources"):
    """Zapisuje obraz base64 do pliku i zwraca ścieżkę"""
    try:
        # Utwórz folder jeśli nie istnieje
        os.makedirs(folder_path, exist_ok=True)
        
        # Wygeneruj unikalną nazwę pliku
        filename = f"{uuid.uuid4()}.jpg"
        file_path = os.path.join(folder_path, filename)
        
        # Dekoduj base64
        if base64_string.startswith('data:image'):
            # Usuń prefix data:image/...;base64,
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        
        # Otwórz obraz i zapisz jako JPEG
        image = Image.open(io.BytesIO(image_data))
        # Konwertuj do RGB jeśli to PNG z przezroczystością
        if image.mode in ('RGBA', 'LA', 'P'):
            image = image.convert('RGB')
        image.save(file_path, 'JPEG', quality=85)
        
        # Zwróć względną ścieżkę dla bazy danych
        return f"resources/{filename}"
        
    except Exception as e:
        print(f"Error saving image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving image: {str(e)}"
        )

def delete_image_file(image_path):
    """Usuwa plik obrazu"""
    try:
        if image_path and image_path.startswith('/resources/'):
            filename = image_path.split('/')[-1]
            full_path = os.path.join("C:/xampp/htdocs/react_front_app/resources", filename)
            if os.path.exists(full_path):
                os.remove(full_path)
    except Exception as e:
        print(f"Error deleting image: {e}")

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
    "https://najshajs.mywire.org",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
#    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount dla statycznych plików (obrazy banknotów i użytkowników)
app.mount("/resources", StaticFiles(directory="C:/xampp/htdocs/react_front_app/resources"), name="resources")

def create_test_user():
    """Tworzy testowego użytkownika i administratora jeśli nie istnieją."""
    db_session = next(get_db())
    try:
        # Sprawdź czy testowy użytkownik już istnieje
        existing_user = db_session.query(User).filter(User.username == "test").first()
        if not existing_user:
            # Utwórz testowego użytkownika
            hashed_password = auth.get_password_hash("test123")
            test_user = User(
                username="test",
                password=hashed_password,
                admin=False
            )
            db_session.add(test_user)
            db_session.commit()
            print("✅ Utworzono testowego użytkownika: test/test123")
        else:
            print("✅ Testowy użytkownik już istnieje")
            
        # Sprawdź czy administrator już istnieje
        existing_admin = db_session.query(User).filter(User.username == "admin").first()
        if not existing_admin:
            # Utwórz administratora
            admin_password = auth.get_password_hash("admin123")
            admin_user = User(
                username="admin",
                password=admin_password,
                admin=True
            )
            db_session.add(admin_user)
            db_session.commit()
            print("✅ Utworzono administratora: admin/admin123")
        else:
            print("✅ Administrator już istnieje")
            
    except Exception as e:
        print(f"❌ Błąd przy tworzeniu użytkowników: {e}")
    finally:
        db_session.close()

@app.on_event("startup")
async def startup_event():
    load_models()
    check_and_prepare_database()  # Tworzymy bazę i tabele przy starcie
    create_test_user()  # Tworzymy testowego użytkownika





class HistoryItem(BaseModel):
    id: int
    timestamp: datetime
    knn_pred: str
    rf_pred: str
    svm_pred: str
    user_id: int | None
    image: str | None   # <-- może być None

    class Config:
        from_attributes = True

# Modele dla panelu administratora
class AdminStats(BaseModel):
    total_users: int
    total_history: int
    recent_activity: int
    total_banknotes: int

class AdminUser(BaseModel):
    id: int
    username: str
    admin: bool
    created_at: datetime

    class Config:
        from_attributes = True

class AdminHistoryEntry(BaseModel):
    id: int
    user_id: int
    username: str
    prediction: str
    timestamp: datetime

    class Config:
        from_attributes = True

# Modele dla oryginalnego API admin_panel
class AdminUpdateRequest(BaseModel):
    admin: bool

class CreateUserRequest(BaseModel):
    username: str
    password: str
    admin: bool = False


# --- Funkcje pomocnicze ---
def get_admin_user(current_user: User = Depends(auth.get_current_user)):
    """Sprawdza czy użytkownik ma uprawnienia administratora"""
    if not current_user.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Brak uprawnień administratora"
        )
    return current_user

# --- Endpointy API ---
@app.get("/")
def read_root():
    return {"message": "Hello, FastAPI! Models are loaded and ready."}

# --- Endpointy autoryzacji ---
@app.post("/api/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Endpoint do logowania - zwraca JWT token."""
    user = auth.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nieprawidłowa nazwa użytkownika lub hasło",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/register", response_model=UserResponse)
async def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Endpoint do rejestracji nowego użytkownika."""
    # Sprawdź czy użytkownik już istnieje
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Użytkownik o tej nazwie już istnieje"
        )
    
    # Utwórz nowego użytkownika
    hashed_password = auth.get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        password=hashed_password,
        admin=False
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

@app.get("/api/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(auth.get_current_user)):
    """Endpoint do pobierania informacji o zalogowanym użytkowniku."""
    return current_user



@app.post("/api/upload")
async def upload_image(data: ImageData, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
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

        # --- Zapis obrazu użytkownika ---
        try:
            user_image_path = save_base64_image(
                data.image, 
                folder_path="C:/xampp/htdocs/react_front_app/resources"
            )
        except Exception as e:
            print(f"Błąd zapisywania obrazu użytkownika: {e}")
            user_image_path = None

        # --- Zapis do bazy z user_id zalogowanego użytkownika ---
        try:
            db_record = History(
                knn_pred=classification_results["knn"]["pred"],
                rf_pred=classification_results["rf"]["pred"],
                svm_pred=classification_results["svm"]["pred"],
                user_id=current_user.id,
                image=user_image_path
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
def get_history(db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    """
    Pobiera historię przetworzonych obrazów zalogowanego użytkownika.
    """
    history = db.query(History).filter(History.user_id == current_user.id).order_by(History.timestamp.desc()).all()
    return history

@app.delete("/api/history/{history_id}")
def delete_history_item(history_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    """
    Usuwa pojedynczy wpis z historii zalogowanego użytkownika.
    """
    history_item = db.query(History).filter(
        History.id == history_id,
        History.user_id == current_user.id
    ).first()
    
    if not history_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nie znaleziono wpisu w historii"
        )
    
    db.delete(history_item)
    db.commit()
    
    return {"message": "Wpis został usunięty z historii"}

@app.get("/api/history/{history_id}/details")
def get_history_item_details(history_id: int, db: Session = Depends(get_db), current_user: User = Depends(auth.get_current_user)):
    """
    Pobiera szczegóły pojedynczego wpisu z historii zalogowanego użytkownika.
    """
    history_item = db.query(History).filter(
        History.id == history_id,
        History.user_id == current_user.id
    ).first()
    
    if not history_item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nie znaleziono wpisu w historii"
        )
    
    return {
        "id": history_item.id,
        "timestamp": history_item.timestamp,
        "knn_pred": history_item.knn_pred,
        "rf_pred": history_item.rf_pred,
        "svm_pred": history_item.svm_pred,
        "user_id": history_item.user_id,
        "image": history_item.image
    }

# --- Endpointy gościa ---
@app.post("/api/guest/upload")
async def guest_upload(file: UploadFile = File(...)):
    """Endpoint dla gości - analiza obrazu bez autoryzacji"""
    try:
        # Sprawdź typ pliku
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plik musi być obrazem"
            )
        
        # Sprawdź rozmiar pliku (max 10MB)
        if file.size and file.size > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Plik jest za duży. Maksymalny rozmiar to 10MB"
            )
        
        # Odczytaj zawartość pliku
        contents = await file.read()
        
        # Konwertuj na base64
        import base64
        base64_string = base64.b64encode(contents).decode('utf-8')
        
        # --- Dekodowanie obrazu --- (dokładnie jak w głównym endpoincie)
        image_data = base64_string
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # --- Klasyfikacja --- (dokładnie jak w głównym endpoincie)
        try:
            classification_results = procesIMG(image)
        except Exception as e:
            raise HTTPException(status_code=520, detail=str(e))

        # Zwróć wyniki bezpośrednio (dokładnie jak w głównym endpoincie)
        return classification_results
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Błąd w guest upload: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Błąd podczas analizy obrazu: {str(e)}"
        )

# --- Endpointy administratora ---
@app.get("/api/admin/stats", response_model=AdminStats)
def get_admin_stats(db: Session = Depends(get_db), admin_user: User = Depends(get_admin_user)):
    """Pobiera statystyki dla panelu administratora"""
    total_users = db.query(User).count()
    total_history = db.query(History).count()
    total_banknotes = db.query(Banknote).count()
    
    # Aktywność z ostatnich 24 godzin
    yesterday = get_polish_time() - timedelta(days=1)
    recent_activity = db.query(History).filter(History.timestamp >= yesterday).count()
    
    return AdminStats(
        total_users=total_users,
        total_history=total_history,
        recent_activity=recent_activity,
        total_banknotes=total_banknotes
    )

# Dodatkowy endpoint zgodny z oryginalnym API admin_panel
@app.get("/admin/dashboard")
def get_dashboard_stats(db: Session = Depends(get_db)):
    """Dashboard dla admina – zwraca statystyki zgodne z wymaganym formatem JSON."""
    try:
        # Liczba rekordów w tabeli History
        history_count = db.query(History).count()
        
        # Liczba rekordów w tabeli User
        users_count = db.query(User).count()
        
        # Liczba banknotów w systemie (26 różnych typów)
        countries_count = 26
        
        return {
            "history_count": history_count,
            "users_count": users_count,
            "countries": countries_count,
            "server_status": "on"
        }
    except Exception as e:
        # W przypadku krytycznego błędu
        return {
            "history_count": 0,
            "users_count": 0,
            "countries": 0,
            "server_status": "off"
        }

@app.get("/api/admin/users", response_model=List[AdminUser])
def get_admin_users(db: Session = Depends(get_db), admin_user: User = Depends(get_admin_user)):
    """Pobiera listę wszystkich użytkowników"""
    users = db.query(User).order_by(User.id).all()
    return users

@app.get("/api/admin/history", response_model=List[AdminHistoryEntry])
def get_admin_history(db: Session = Depends(get_db), admin_user: User = Depends(get_admin_user)):
    """Pobiera historię wszystkich użytkowników"""
    history_entries = db.query(
        History.id,
        History.user_id,
        User.username,
        History.knn_pred,
        History.rf_pred,
        History.svm_pred,
        History.timestamp
    ).join(User, History.user_id == User.id).order_by(desc(History.timestamp)).limit(100).all()
    
    # Przekształć dane na format AdminHistoryEntry
    result = []
    for entry in history_entries:
        # Wybierz najlepsze przewidywanie
        predictions = [entry.knn_pred, entry.rf_pred, entry.svm_pred]
        prediction = max(set(predictions), key=predictions.count) if predictions else "Nie rozpoznano"
        
        result.append(AdminHistoryEntry(
            id=entry.id,
            user_id=entry.user_id,
            username=entry.username,
            prediction=prediction,
            timestamp=entry.timestamp
        ))
    
    return result

@app.delete("/api/admin/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db), admin_user: User = Depends(get_admin_user)):
    """Usuwa użytkownika (tylko administrator)"""
    if user_id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nie można usunąć własnego konta"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nie znaleziono użytkownika"
        )
    
    # Usuń historię użytkownika
    db.query(History).filter(History.user_id == user_id).delete()
    
    # Usuń użytkownika
    db.delete(user)
    db.commit()
    
    return {"message": f"Użytkownik {user.username} został usunięty"}

@app.put("/api/admin/users/{user_id}/admin")
def toggle_admin_status(user_id: int, db: Session = Depends(get_db), admin_user: User = Depends(get_admin_user)):
    """Przełącza status administratora użytkownika"""
    if user_id == admin_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Nie można zmienić własnych uprawnień"
        )
    
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nie znaleziono użytkownika"
        )
    
    user.admin = not user.admin
    db.commit()
    
    return {"message": f"Status administratora dla {user.username} zmieniony na {user.admin}"}

# Dodatkowe endpointy zgodne z oryginalnym API admin_panel
@app.get("/admin/users")
def get_users_admin(db: Session = Depends(get_db)):
    """Zwraca listę wszystkich użytkowników w bazie (bez autoryzacji dla kompatybilności)"""
    try:
        return db.query(User).all()
    except Exception as e:
        return {
            "server_status": -1
        }

@app.get("/admin/history/all", response_model=List[HistoryItem])
def get_all_history_admin(db: Session = Depends(get_db)):
    """Zwraca całą historię (bez autoryzacji dla kompatybilności)"""
    history = db.query(History).order_by(History.timestamp.desc()).all()
    return history

@app.delete("/admin/users/{userID}")
def delete_user_admin(userID: int, db: Session = Depends(get_db)):
    """Usuwa użytkownika (bez autoryzacji dla kompatybilności)"""
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
            synchronize_session=False
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

        if "ForeignKeyViolation" in error_msg or "klucz obcy" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot delete user due to database constraints. User may have additional dependencies."
            )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the user"
        )

@app.patch("/admin/users/{userID}/admin")
def update_user_admin_privileges_admin(
        userID: int,
        admin_data: AdminUpdateRequest,
        db: Session = Depends(get_db)
):
    """Przełącza status administratora użytkownika (bez autoryzacji dla kompatybilności)"""
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
        db.refresh(user)

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

@app.post("/admin/users")
def create_user_admin(user_data: CreateUserRequest, db: Session = Depends(get_db)):
    """Tworzy nowego użytkownika (bez autoryzacji dla kompatybilności)"""
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
            "password": new_user.password,
            "admin": new_user.admin,
            "created_at": new_user.created_at,
            "message": f"User '{new_user.username}' created successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error creating user: {error_msg}")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while creating the user"
        )

# --- ENDPOINTY BANKNOTÓW ---

@app.get("/api/banknotes", response_model=List[BanknoteResponse])
def get_banknotes(db: Session = Depends(get_db)):
    """Pobiera listę wszystkich banknotów"""
    banknotes = db.query(Banknote).order_by(Banknote.denomination).all()
    return banknotes

@app.get("/api/banknotes/{banknote_id}", response_model=BanknoteResponse)
def get_banknote(banknote_id: int, db: Session = Depends(get_db)):
    """Pobiera konkretny banknot po ID"""
    banknote = db.query(Banknote).filter(Banknote.id == banknote_id).first()
    if not banknote:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Banknote not found"
        )
    return banknote

@app.post("/api/banknotes", response_model=BanknoteResponse)
def create_banknote(banknote_data: BanknoteCreate, db: Session = Depends(get_db)):
    """Tworzy nowy banknot"""
    try:
        # Zapisz obrazy base64 do plików
        avers_path = None
        rewers_path = None
        
        if banknote_data.image_avers:
            avers_path = save_base64_image(banknote_data.image_avers)
        
        if banknote_data.image_rewers:
            rewers_path = save_base64_image(banknote_data.image_rewers)
        
        new_banknote = Banknote(
            country=banknote_data.country,
            currency=banknote_data.currency,
            denomination=banknote_data.denomination,
            effigy=banknote_data.effigy,
            dimensions=banknote_data.dimensions,
            description=banknote_data.description,
            image_avers=avers_path or "",
            image_rewers=rewers_path or ""
        )
        
        db.add(new_banknote)
        db.commit()
        db.refresh(new_banknote)
        
        return new_banknote
    except Exception as e:
        db.rollback()
        # Usuń zapisane pliki w przypadku błędu
        if 'avers_path' in locals() and avers_path:
            delete_image_file(avers_path)
        if 'rewers_path' in locals() and rewers_path:
            delete_image_file(rewers_path)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating banknote: {str(e)}"
        )

@app.put("/api/banknotes/{banknote_id}", response_model=BanknoteResponse)
def update_banknote(banknote_id: int, banknote_data: BanknoteUpdate, db: Session = Depends(get_db)):
    """Aktualizuje banknot"""
    try:
        banknote = db.query(Banknote).filter(Banknote.id == banknote_id).first()
        if not banknote:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Banknote not found"
            )
        
        # Zapisz stare ścieżki obrazów
        old_avers_path = banknote.image_avers
        old_rewers_path = banknote.image_rewers
        
        # Aktualizuj tylko podane pola
        update_data = banknote_data.dict(exclude_unset=True)
        
        # Obsłuż obrazy base64
        if 'image_avers' in update_data and update_data['image_avers']:
            if update_data['image_avers'].startswith('data:image'):
                # Nowy obraz base64 - zapisz do pliku
                new_avers_path = save_base64_image(update_data['image_avers'])
                update_data['image_avers'] = new_avers_path
                # Usuń stary plik
                if old_avers_path:
                    delete_image_file(old_avers_path)
        
        if 'image_rewers' in update_data and update_data['image_rewers']:
            if update_data['image_rewers'].startswith('data:image'):
                # Nowy obraz base64 - zapisz do pliku
                new_rewers_path = save_base64_image(update_data['image_rewers'])
                update_data['image_rewers'] = new_rewers_path
                # Usuń stary plik
                if old_rewers_path:
                    delete_image_file(old_rewers_path)
        
        for field, value in update_data.items():
            setattr(banknote, field, value)
        
        db.commit()
        db.refresh(banknote)
        
        return banknote
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating banknote: {str(e)}"
        )

@app.delete("/api/banknotes/{banknote_id}")
def delete_banknote(banknote_id: int, db: Session = Depends(get_db)):
    """Usuwa banknot"""
    try:
        banknote = db.query(Banknote).filter(Banknote.id == banknote_id).first()
        if not banknote:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Banknote not found"
            )
        
        # Usuń pliki obrazów
        if banknote.image_avers:
            delete_image_file(banknote.image_avers)
        if banknote.image_rewers:
            delete_image_file(banknote.image_rewers)
        
        db.delete(banknote)
        db.commit()
        
        return {"message": "Banknote deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting banknote: {str(e)}"
        )