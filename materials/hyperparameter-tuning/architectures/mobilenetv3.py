"""MobileNetV3-Small (Howard et al., 2019): depthwise separable conv + squeeze-excitation. 1,5 M parameter."""
import timm


def build(num_classes, pretrained, dropout, drop_path, img_size):
    return timm.create_model("mobilenetv3_small_100.lamb_in1k", pretrained=pretrained, num_classes=num_classes,
                             drop_rate=dropout, drop_path_rate=drop_path)
