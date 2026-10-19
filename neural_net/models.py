from torch import nn

import segmentation_models_pytorch as smp


class _SmpWrapper(nn.Module):
    """Thin wrapper around a segmentation_models_pytorch architecture.

    forward(x) returns full-resolution logits of shape (B, n_classes, H, W).
    `pretrained` tells CrossValidator to skip its Xavier re-initialization,
    which would otherwise overwrite the pretrained encoder.
    """

    arch = None
    default_encoder = None

    def __init__(self, n_channels, n_classes, encoder_name=None, encoder_weights=None):
        super().__init__()
        self.n_classes = n_classes
        self.encoder_name = encoder_name or self.default_encoder
        self.pretrained = encoder_weights is not None
        self.model = self.arch(
            encoder_name=self.encoder_name,
            encoder_weights=encoder_weights,
            in_channels=n_channels,
            classes=n_classes,
        )

    def forward(self, x):
        return self.model(x)


class ResNetUnet(_SmpWrapper):
    """U-Net with a ResNet-34 encoder."""
    arch = smp.Unet
    default_encoder = 'resnet34'


class SegFormer(_SmpWrapper):
    """SegFormer (MiT-B2 encoder, all-MLP decoder)."""
    arch = smp.Segformer
    default_encoder = 'mit_b2'


MODELS = {
    'resnet_unet': ResNetUnet,
    'segformer': SegFormer,
}
