"""Model: ambil arsitektur dari folder architectures/, lalu siapkan sesuai strategi fine-tuning.

Menambah arsitektur baru = buat architectures/<nama>.py berisi fungsi build(...), lalu daftarkan di ARCHITECTURES.
(Foldernya tidak bernama models/ karena akan bentrok dengan file models.py ini saat di-import.)
"""
import torch.nn as nn

from architectures import convnext_atto, efficientnet_b0, mobilenetv3, resnet18, vit_tiny

ARCHITECTURES = {
    "resnet18": resnet18.build,
    "mobilenetv3": mobilenetv3.build,
    "efficientnet_b0": efficientnet_b0.build,
    "convnext_atto": convnext_atto.build,
    "vit_tiny": vit_tiny.build,
}


def build_model(c, num_classes):
    """c["finetune"]: "scratch" (bobot acak) | "linear_probe" (backbone beku) | "full_ft" (semua dilatih)."""
    if c["model"] not in ARCHITECTURES:
        raise ValueError(f"model tidak dikenal: {c['model']} (pilihan: {list(ARCHITECTURES)})")
    if c["finetune"] not in ("scratch", "linear_probe", "full_ft"):
        raise ValueError(f"finetune tidak dikenal: {c['finetune']}")

    pretrained = c["finetune"] != "scratch"
    model = ARCHITECTURES[c["model"]](num_classes, pretrained, c["dropout"], c["drop_path"], c["img_size"])

    if c["finetune"] == "linear_probe":  # hanya classifier head (lapisan terakhir) yang dilatih
        for p in model.parameters():
            p.requires_grad = False
        for p in model.get_classifier().parameters():
            p.requires_grad = True
    return model


def parameter_groups(model, c):
    """Head memakai lr, backbone memakai lr * backbone_lr_mult (kecuali scratch: semua lr yang sama)."""
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_ids]
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    mult = 1.0 if c["finetune"] == "scratch" else c["backbone_lr_mult"]
    groups = [{"params": head, "lr": c["lr"]}]
    if backbone:  # kosong saat linear_probe
        groups.append({"params": backbone, "lr": c["lr"] * mult})
    return groups


def set_train_mode(model, c):
    model.train()
    if c["finetune"] == "linear_probe":  # backbone beku -> statistik BatchNorm juga jangan ikut berubah
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
