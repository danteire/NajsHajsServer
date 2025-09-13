# Instrukcja uruchomienia serwera NajsHajs z JWT

## Wymagania
- Python 3.8+
- Instalacja zależności Python z pliku requirements.txt

## Uwagi
- Frontend domyślnie skonfigurowany do pracy z https://najshajs.mywire.org
- JWT tokeny wygasają po 30 minutach
- Wszystkie endpointy wymagające autoryzacji sprawdzają JWT token

## Kroki uruchomienia

### 1. Uruchomienie bazy danych PostgreSQL

### 2. Uruchomienie serwera FastAPI skryptem uvicorn_run.bat

### 3. Uruchomienie Apache

### 4. Uruchomienie FRP (Fast Reverse Proxy)

### 5. Uruchomienie frontendu
### npm run build

### 6. Dystrybucja plików frontendu przez Apache 

## Użytkownicy:
- **Testowy użytkownik:** `test` / `test`
- **Administrator:** `admin` / `admin`

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

### Tryb gościa (bez autoryzacji)
- `POST /api/upload` - Upload obrazu banknotu dla gościa (user_id=null)
- `GET /api/banknotes` - Lista wszystkich banknotów w systemie
- `GET /api/banknotes/{id}` - Szczegóły konkretnego banknotu
- `GET /api/banknotes/{id}/image` - Obraz banknotu


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
- **Host:** localhost:5432 / najshajs.mywire.org
- **Baza:** NajsHajs
- **Użytkownik:** postgres
- **Hasło:** admin

## Adresy API
- **Serwer:** https://najshajs.mywire.org


### Pliki konfiguracyjne dla Apache
- `.htaccess`