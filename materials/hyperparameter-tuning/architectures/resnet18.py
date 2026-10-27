"""ResNet-18 (He et al., 2016): CNN klasik dengan residual connection. 11,2 M parameter."""
import timm


def build(num_classes, pretrained, dropout, drop_path, img_size):
    return timm.create_model("resnet18.a1_in1k", pretrained=pretrained, num_classes=num_classes,
                             drop_rate=dropout, drop_path_rate=drop_path)
