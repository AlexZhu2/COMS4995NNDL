import torch.nn as nn
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    EfficientNet_V2_S_Weights
)

class CNNEncoder(nn.Module):
    def __init__(self, model_name="resnet50", embed_dim=512):
        """
        Args:
            model_name (str): Name of the backbone model. Options: "resnet50", "efficientnetv2_s"
            embed_dim (int): Dimension of the output feature vectors
        """
        super(CNNEncoder, self).__init__()
        
        self.model_name = model_name.lower()
        
        if self.model_name == "resnet50":
            backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            out_channels = 2048
            # Remove final fully connected + avgpool layers
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])
            self.pool = nn.AdaptiveAvgPool2d((7, 7))
        
        elif self.model_name == "efficientnetv2_s":
            backbone = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            out_channels = backbone.classifier[1].in_features  # Final feature dim before classifier
            self.backbone = backbone.features  # Only use the feature extraction layers
            self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Same output spatial size
        
        else:
            raise ValueError(f"Unsupported model name: {model_name}")
        
        # Projection layer to map backbone output to embed_dim
        self.conv_proj = nn.Conv2d(out_channels, embed_dim, kernel_size=1)
        
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch_size, 3, H, W)
        Returns:
            features: Tensor of shape (batch_size, num_patches, embed_dim)
        """
        features = self.backbone(images)  # shape: (B, C, H_feat, W_feat)
        features = self.pool(features)    # shape: (B, C, 7, 7)
        features = self.conv_proj(features)  # shape: (B, embed_dim, 7, 7)
        
        # Flatten spatial dimensions
        B, embed_dim, H, W = features.shape
        features = features.view(B, embed_dim, H * W)  # (B, embed_dim, 49)
        features = features.permute(0, 2, 1)           # (B, 49, embed_dim)
        
        return features
