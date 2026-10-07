"""Contoh submission: ResNet-18 dengan head 5 kelas.

Ganti isi file ini dengan arsitektur kelompok Anda. Aturan:
  - get_model() hanya MEMBANGUN arsitektur (jangan mengunduh bobot pretrained di sini,
    jangan memuat model.pth di sini — evaluate.py yang memuatnya).
  - get_transform() harus SAMA dengan transform validasi saat Anda melatih model.
  - Output model: logits [batch, 5], urutan kelas:
      0 botol_plastik, 1 gelas_plastik, 2 kertas_kardus, 3 saset_kemasan, 4 styrofoam
    (urutan alfabetis — sama dengan torchvision.datasets.ImageFolder).
"""
import torch.nn as nn
from torchvision import models, transforms

NUM_CLASSES = 5


def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
