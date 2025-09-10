@echo off
echo Uruchamianie serwera NajsHajs...

echo.
echo 1. Uruchamianie bazy danych PostgreSQL...
cd DB
docker-compose up -d
cd ..

echo.
echo 2. Instalowanie zależności...
pip install -r requirements.txt

echo.
echo 3. Uruchamianie serwera FastAPI...
uvicorn mainserv:app --host 0.0.0.0 --port 8000 --reload

pause

