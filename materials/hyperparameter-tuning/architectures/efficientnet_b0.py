"""EfficientNet-B0 (Tan & Le, 2019): compound scaling lebar, kedalaman, dan resolusi. 4,0 M parameter."""
import timm


def build(num_classes, pretrained, dropout, drop_path, img_size):
    return timm.create_model("efficientnet_b0.ra_in1k", pretrained=pretrained, num_classes=num_classes,
                             drop_rate=dropout, drop_path_rate=drop_path)
