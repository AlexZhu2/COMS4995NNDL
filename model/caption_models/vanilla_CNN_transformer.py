import torch
import torch.nn as nn
from model.decoder.transformer_decoder_layer import TransformerDecoder
from model.encoder.CNN_encoder import CNNEncoder

class CNNTransformerCaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, max_len=50, dropout=0.1, pad_idx=0):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.decoder = TransformerDecoder(vocab_size, embed_dim, num_heads, num_layers, max_len, dropout)
        self.max_len = max_len
        self.pad_idx = pad_idx

    def forward(self, images, captions):
        B, T = captions.size()
        memory = self.encoder(images)  # (B, 49, embed_dim)
        tgt_mask = self.generate_subsequent_mask(T, captions.device)
        logits = self.decoder(captions, memory, tgt_mask=tgt_mask)
        return logits

    def generate_subsequent_mask(self, size, device):
        mask = torch.triu(torch.ones((size, size), device=device) == 1, diagonal=1)
        return mask.float().masked_fill(mask == 1, float('-inf'))

