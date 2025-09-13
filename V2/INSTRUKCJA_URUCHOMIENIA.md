gi# Instrukcja uruchomienia serwera NajsHajs z JWT

## Wymagania
- Python 3.8+
- Docker i Docker Compose
- pip

## Kroki uruchomienia

### 1. Uruchomienie bazy danych PostgreSQL
```bash
cd Backend/V2/DB
docker-compose up -d
```

### 2. Instalacja zależności Python
```bash
cd Backend/V2
pip install -r requirements.txt
```

### 3. Uruchomienie serwera FastAPI
```bash
uvicorn mainserv:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Uruchomienie frontendu
```bash
cd Frontend
npm install
npm start
```

## Użytkownicy
Po uruchomieniu serwera zostaną automatycznie utworzeni użytkownicy:
- **Testowy użytkownik:** `test` / `test123` (zwykły użytkownik)
- **Administrator:** `admin` / `admin123` (z uprawnieniami administratora)

## Endpointy API

### Autoryzacja
- `POST /api/token` - Logowanie (zwraca JWT token)
- `POST /api/register` - Rejestracja nowego użytkownika
- `GET /api/me` - Informacje o zalogowanym użytkowniku

### Główne funkcje
- `POST /api/upload` - Upload obrazu banknotu (wymaga autoryzacji)
- `GET /api/history` - Historia użytkownika (wymaga autoryzacji)
- `GET /api/history/{id}/details` - Szczegóły wpisu z historii (wymaga autoryzacji)
- `DELETE /api/history/{id}` - Usuń wpis z historii (wymaga autoryzacji)

### Panel administratora
- `GET /api/admin/stats` - Statystyki systemu (wymaga uprawnień administratora)
- `GET /api/admin/users` - Lista wszystkich użytkowników (wymaga uprawnień administratora)
- `GET /api/admin/history` - Historia wszystkich użytkowników (wymaga uprawnień administratora)
- `DELETE /api/admin/users/{id}` - Usuń użytkownika (wymaga uprawnień administratora)
- `PUT /api/admin/users/{id}/admin` - Przełącz status administratora (wymaga uprawnień administratora)

### Oryginalne API admin_panel (bez autoryzacji)
- `GET /admin/dashboard` - Dashboard ze statystykami
- `GET /admin/users` - Lista wszystkich użytkowników
- `GET /admin/history/all` - Cała historia
- `DELETE /admin/users/{userID}` - Usuń użytkownika
- `PATCH /admin/users/{userID}/admin` - Przełącz status administratora
- `POST /admin/users` - Utwórz nowego użytkownika

## Trasy frontendu

### Główne trasy
- `/` - Strona logowania
- `/user` - Panel użytkownika (wymaga logowania)
- `/results` - Strona wyników (wymaga logowania)

### Trasy administratora (wymagają uprawnień administratora)
- `/admin` - Dashboard administratora
- `/admin/userlist` - Lista użytkowników
- `/admin/banknotelist` - Lista banknotów w systemie
- `/admin/history` - Historia wszystkich użytkowników

## Konfiguracja bazy danych
- **Host:** localhost:5432 (lokalnie) / najshajs.mywire.org (produkcja)
- **Baza:** NajsHajs
- **Użytkownik:** postgres
- **Hasło:** admin

## Adresy API
- **Lokalny serwer:** http://localhost:8000
- **Produkcyjny serwer:** https://najshajs.mywire.org

## Konfiguracja frontendu

### Przełączanie między środowiskami
W pliku `Frontend/src/config/api.js` możesz przełączać między:
- **Lokalnym:** `environment: 'local'` (http://localhost:8000)
- **Produkcyjnym:** `environment: 'production'` (https://najshajs.mywire.org)

## Wdrożenie frontendu

### Problem z routingiem
Po wdrożeniu na serwer, odświeżenie strony na trasie `/user` powoduje błąd 404. To dlatego, że serwer nie wie o trasach React Router.

### Rozwiązanie
1. **Zbuduj aplikację:** `npm run build` w folderze Frontend
2. **Skopiuj pliki konfiguracyjne** z folderu `build/` na serwer
3. **Włącz mod_rewrite** w Apache (dla pliku `.htaccess`)

### Pliki konfiguracyjne
- `.htaccess` - dla Apache
- `_redirects` - dla Netlify  
- `web.config` - dla IIS

## Uwagi
- Serwer automatycznie tworzy tabele w bazie danych przy starcie
- Testowy użytkownik jest tworzony automatycznie jeśli nie istnieje
- JWT tokeny wygasają po 30 minutach
- Wszystkie endpointy wymagające autoryzacji sprawdzają JWT token
- Frontend domyślnie skonfigurowany do pracy z https://najshajs.mywire.org
- **WAŻNE:** Po wdrożeniu frontendu upewnij się, że pliki konfiguracyjne są na serwerze
