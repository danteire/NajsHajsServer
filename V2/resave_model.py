# resave_model.py
import torch
import sys
from embedder.embedding_model import MobileNetV3Embedder


old_model_path = 'embedder/mobilenetv3_embedder_full.pth'


new_model_path = 'embedder/mobilenetv3_embedder_statedict.pth'

print(f"Ładowanie starego modelu z: {old_model_path}")


sys.modules['__main__'].MobileNetV3Embedder = MobileNetV3Embedder


model = torch.load(old_model_path, map_location='cpu', weights_only=False)

torch.save(model.state_dict(), new_model_path)

print(f"State_dict modelu został poprawnie zapisany w: {new_model_path} ✅")