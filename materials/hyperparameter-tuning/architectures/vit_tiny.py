"""ViT-Tiny/16 (Dosovitskiy et al., 2021): Vision Transformer, citra dipotong jadi patch 16x16. 5,5 M parameter.

Pretrained ImageNet-21k lalu fine-tune ImageNet-1k. Berbeda dengan CNN, ViT perlu tahu img_size
karena jumlah patch (dan positional embedding-nya) bergantung pada ukuran citra.
"""
import timm


def build(num_classes, pretrained, dropout, drop_path, img_size):
    return timm.create_model("vit_tiny_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained,
                             num_classes=num_classes, drop_rate=dropout, drop_path_rate=drop_path,
                             img_size=img_size)
