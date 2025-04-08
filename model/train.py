import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader
from model.vocab import Vocabulary
from model.dataset import CocoCaptionDataset
from model.models import CNNEncoder, TransformerDecoder, generate_square_subsequent_mask

import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and build vocabulary from COCO annotations
with open('data/annotations/captions_train2017.json', 'r') as f:
    coco_data = json.load(f)
captions = [ann["caption"] for ann in coco_data["annotations"]]

vocab = Vocabulary(freq_threshold=5)
vocab.build_vocab(captions)
vocab_size = len(vocab)

# Create dataset and data loader
dataset = CocoCaptionDataset(
    img_folder='data/train2017',
    ann_path='annotations/captions_train2017.json',
    vocab=vocab,
    max_len=20
)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Load pretrained ResNet-50 and modify for feature extraction
resnet50 = models.resnet50(pretrained=True)
modules = list(resnet50.children())[:-2]  # Remove avgpool and fc layers
cnn_encoder_base = nn.Sequential(*modules).to(device)

# Optionally freeze early layers except the last block
for name, param in cnn_encoder_base.named_parameters():
    param.requires_grad = False
    if 'layer4' in name:
        param.requires_grad = True

encoder = CNNEncoder(cnn_encoder_base).to(device)

# Create Transformer decoder
decoder = TransformerDecoder(vocab_size, d_model=512, nhead=8, num_layers=6,
                             dim_feedforward=2048, dropout=0.1, max_len=100).to(device)

# Loss, optimizer, and hyperparameters
criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx["<pad>"])
optimizer = optim.Adam([
    {'params': encoder.parameters(), 'lr': 1e-5},
    {'params': decoder.parameters(), 'lr': 1e-4}
])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    epoch_loss = 0.0
    
    for images, captions in data_loader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()

        # Get image features from the CNN encoder
        features = encoder(images)  # shape: [batch, 49, 2048]

        # Create a target mask for Transformer decoder
        tgt_mask = generate_square_subsequent_mask(captions.shape[1]).to(device)

        # Use teacher forcing: input captions[:, :-1]; target captions[:, 1:]
        outputs = decoder(captions[:, :-1], features, tgt_mask=tgt_mask)
        
        loss = criterion(outputs.reshape(-1, vocab_size), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(data_loader):.4f}")
