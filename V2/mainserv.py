from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from database import check_and_prepare_database
from classify import load_models

# Import routerów
from userUtils import router as user_router
from adminUtils import router as admin_router

app = FastAPI(title="Unified FastAPI App")

# --- Startup event ---
@app.on_event("startup")
async def startup_event():
    load_models()
    check_and_prepare_database()


# --- Konfiguracja CORS ---
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Rejestracja routerów ---
app.include_router(user_router)
app.include_router(admin_router)


@app.get("/")
def root():
    return {"message": "Main API is running! Check /user or /admin"}
