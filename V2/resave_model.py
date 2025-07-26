# resave_model.py
import torch
import sys
from embedder.embedding_model import MobileNetV3Embedder # Upewnij się, że import działa

# Ścieżka do Twojego obecnego, problematycznego modelu
old_model_path = 'embedder/mobilenetv3_embedder_full.pth'

# Ścieżka, gdzie zapiszemy nowy, poprawny plik z wagami
new_model_path = 'embedder/mobilenetv3_embedder_statedict.pth'

print(f"Ładowanie starego modelu z: {old_model_path}")

# To jest potrzebne, aby załadować model zapisany w __main__
sys.modules['__main__'].MobileNetV3Embedder = MobileNetV3Embedder

# ---- KLUCZOWA ZMIANA JEST TUTAJ ----
# Używamy weights_only=False, aby wczytać cały obiekt modelu (zgodnie z sugestią błędu)
model = torch.load(old_model_path, map_location='cpu', weights_only=False)

# Zapisujemy sam state_dict - to jest poprawny sposób
torch.save(model.state_dict(), new_model_path)

print(f"State_dict modelu został poprawnie zapisany w: {new_model_path} ✅")