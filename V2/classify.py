# classify.py

import json
from pathlib import Path
from typing import Dict, Any

import torch
from joblib import load
from PIL import Image, ImageOps
from torchvision.transforms import v2 as transforms

from embedder.embedding_model import MobileNetV3Embedder

# --- Zmienne globalne na modele ---
MODELS = {}


# --- Transformacje (bez zmian) ---
def pad_to_square(img):
    w, h = img.size
    m = max(w, h)
    pad = ((m - w) // 2, (m - h) // 2, (m - w + 1) // 2, (m - h + 1) // 2)
    return ImageOps.expand(img, pad, fill=(0, 0, 0))


transform_mobilenet = transforms.Compose([
    transforms.Lambda(pad_to_square),
    transforms.Resize((224, 224)),
    transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True)
    ]),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# --- Funkcje pomocnicze (z drobnymi zmianami) ---
def extract_embeddings(pil_img: Image.Image) -> tuple:
    """Przetwarza obraz PIL i zwraca logits oraz embeddingi."""
    img_t = transform_mobilenet(pil_img).unsqueeze(0)
    with torch.no_grad():
        logits, embeddings = MODELS['embedder'](img_t)
    return logits.squeeze().cpu().numpy(), embeddings.squeeze().cpu().numpy()


def proba_dict(proba, labels) -> dict:
    return {labels[i]: float(proba[i]) for i in range(len(labels))}


def classify(embeddings) -> dict:
    """Używa załadowanych klasyfikatorów do predykcji na podstawie embeddingów."""
    emb_scaled = MODELS['scaler'].transform(embeddings.reshape(1, -1))[0]

    # KNN
    pred_k, proba_vec_k = MODELS['knn'].predict(emb_scaled.reshape(1, -1))[0], \
    MODELS['knn'].predict_proba(emb_scaled.reshape(1, -1))[0]
    # Random Forest
    pred_r, proba_vec_r = MODELS['rf'].predict(emb_scaled.reshape(1, -1))[0], \
    MODELS['rf'].predict_proba(emb_scaled.reshape(1, -1))[0]
    # SVM
    pred_s, proba_vec_s = MODELS['svm'].predict(emb_scaled.reshape(1, -1))[0], \
    MODELS['svm'].predict_proba(emb_scaled.reshape(1, -1))[0]

    labels = MODELS['labels']
    return {
        'knn': {'pred': labels[pred_k], 'proba': proba_dict(proba_vec_k, labels)},
        'rf': {'pred': labels[pred_r], 'proba': proba_dict(proba_vec_r, labels)},
        'svm': {'pred': labels[pred_s], 'proba': proba_dict(proba_vec_s, labels)},
    }


# --- GŁÓWNA FUNKCJA PRZETWARZAJĄCA ---
def procesIMG(pil_img: Image.Image) -> Dict[str, Any]:
    """
    Przetwarza pojedynczy obraz PIL i zwraca słownik z wynikami klasyfikacji.
    """
    if not MODELS:
        raise RuntimeError("Modele nie zostały załadowane. Użyj funkcji load_models().")

    # Krok 1: Ekstrakcja cech (embeddingów) z obrazu
    _logits, emb = extract_embeddings(pil_img)

    # Krok 2: Klasyfikacja na podstawie embeddingów
    results = classify(emb)

    return results


# --- Funkcja do jednorazowego ładowania modeli ---
def load_models():
    """Ładuje wszystkie modele do pamięci."""
    print("Ładowanie modeli...")
    models_base_path = 'models/'
    embedder_path = 'embedder/mobilenetv3_embedder_statedict.pth'

    # ---- ZMIEŃ 25 NA 26 W TEJ LINII ----
    MODELS['embedder'] = MobileNetV3Embedder(weights=None, embedding_dim=512, num_classes=26)

    # Krok 2: Załaduj wagi (state_dict) do tej instancji
    MODELS['embedder'].load_state_dict(torch.load(embedder_path, map_location='cpu'))

    # Ustaw model w tryb ewaluacji
    MODELS['embedder'].eval()

    MODELS['scaler'] = load(f"{models_base_path}scaler.joblib")
    MODELS['knn'] = load(f"{models_base_path}kNN_k5.joblib")
    MODELS['rf'] = load(f"{models_base_path}RandomForest.joblib")
    MODELS['svm'] = load(f"{models_base_path}SVM_linear.joblib")

    with open(f"{models_base_path}labels.txt", 'r') as f:
        MODELS['labels'] = [l.strip() for l in f]
    print("Modele załadowane pomyślnie. ✅")