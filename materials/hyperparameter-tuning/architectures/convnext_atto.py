"""ConvNeXt-Atto (Liu et al., 2022): CNN "modern" dengan resep desain ala Transformer. 3,4 M parameter."""
import timm


def build(num_classes, pretrained, dropout, drop_path, img_size):
    return timm.create_model("convnext_atto.d2_in1k", pretrained=pretrained, num_classes=num_classes,
                             drop_rate=dropout, drop_path_rate=drop_path)
