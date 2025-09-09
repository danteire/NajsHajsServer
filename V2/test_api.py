#!/usr/bin/env python3
"""
Skrypt do testowania API NajsHajs z JWT
"""

import requests
import json

BASE_URL = "https://najshajs.mywire.org"

def test_register():
    """Test rejestracji nowego użytkownika"""
    print("=== Test rejestracji ===")
    url = f"{BASE_URL}/api/register"
    data = {
        "username": "testuser",
        "password": "testpass123"
    }
    
    try:
        response = requests.post(url, json=data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Rejestracja udana!")
            print(f"Odpowiedź: {response.json()}")
        else:
            print(f"❌ Błąd rejestracji: {response.text}")
    except Exception as e:
        print(f"❌ Błąd połączenia: {e}")

def test_login():
    """Test logowania"""
    print("\n=== Test logowania ===")
    url = f"{BASE_URL}/api/token"
    data = {
        "username": "test",
        "password": "test123"
    }
    
    try:
        response = requests.post(url, data=data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Logowanie udane!")
            token_data = response.json()
            print(f"Token: {token_data['access_token'][:50]}...")
            return token_data['access_token']
        else:
            print(f"❌ Błąd logowania: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Błąd połączenia: {e}")
        return None

def test_me(token):
    """Test pobierania informacji o użytkowniku"""
    print("\n=== Test /me ===")
    url = f"{BASE_URL}/api/me"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Pobieranie danych użytkownika udane!")
            print(f"Dane: {response.json()}")
        else:
            print(f"❌ Błąd: {response.text}")
    except Exception as e:
        print(f"❌ Błąd połączenia: {e}")

def test_history(token):
    """Test pobierania historii"""
    print("\n=== Test /api/history ===")
    url = f"{BASE_URL}/api/history"
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Pobieranie historii udane!")
            history = response.json()
            print(f"Liczba wpisów w historii: {len(history)}")
        else:
            print(f"❌ Błąd: {response.text}")
    except Exception as e:
        print(f"❌ Błąd połączenia: {e}")

def main():
    print("🧪 Testowanie API NajsHajs z JWT")
    print("=" * 50)
    
    # Test rejestracji
    test_register()
    
    # Test logowania
    token = test_login()
    
    if token:
        # Test pobierania danych użytkownika
        test_me(token)
        
        # Test pobierania historii
        test_history(token)
    
    print("\n" + "=" * 50)
    print("✅ Testy zakończone!")

if __name__ == "__main__":
    main()
