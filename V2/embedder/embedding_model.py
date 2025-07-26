import torch.nn as nn
from torchvision.models import mobilenet_v3_large

class MobileNetV3Embedder(nn.Module):
    def __init__(self, weights, freeze_backbone=False, embedding_dim=512, num_classes=25):
        super().__init__()
        self.backbone = mobilenet_v3_large(weights=weights)
        if freeze_backbone:
            for p in self.backbone.parameters(): p.requires_grad = False
        self.backbone.classifier = nn.Identity()
        self.projector = nn.Sequential(
            nn.Linear(960, embedding_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(embedding_dim)
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        features   = self.backbone(x)
        embeddings = self.projector(features)
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        logits     = self.classifier(embeddings)
        return logits, embeddings
