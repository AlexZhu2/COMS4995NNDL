import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        """
        Args:
            embed_dim (int): Dimension of the output feature vectors (matches Transformer d_model)
            pretrained (bool): Whether to use a pretrained CNN backbone
        """
        super(CNNEncoder, self).__init__()
        
        # Load a pretrained ResNet50 model
        resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        
        # Remove the final fully connected layer (we only want the feature maps)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # up to the last conv layer
        
        # Adaptive pooling to get a fixed size output (like 7x7 feature map)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Project CNN output to Transformer embedding dimension
        self.conv_proj = nn.Conv2d(2048, embed_dim, kernel_size=1)
        
    def forward(self, images):
        """
        Args:
            images: tensor of shape (batch_size, 3, 224, 224)
            
        Returns:
            features: tensor of shape (batch_size, num_patches, embed_dim)
        """
        # Extract feature maps
        features = self.backbone(images)        # (batch_size, 2048, H/32, W/32), typically (batch_size, 2048, 7, 7)
        
        features = self.adaptive_pool(features) # (batch_size, 2048, 7, 7) if not already
        features = self.conv_proj(features)     # (batch_size, embed_dim, 7, 7)
        
        # Flatten spatial dimensions
        batch_size, embed_dim, h, w = features.size()
        features = features.view(batch_size, embed_dim, h * w)    # (batch_size, embed_dim, 49)
        features = features.permute(0, 2, 1)                      # (batch_size, 49, embed_dim)
        
        return features
