import torch
import torch.nn as nn
import math

# CNN Encoder using ResNet-50 (excluding the final layers)
class CNNEncoder(nn.Module):
    def __init__(self, cnn):
        super(CNNEncoder, self).__init__()
        self.cnn = cnn

    def forward(self, images):
        features = self.cnn(images)  # [batch, 2048, 7, 7]
        features = features.view(features.size(0), 2048, -1)  # [batch, 2048, 49]
        features = features.permute(0, 2, 1)  # [batch, 49, 2048]
        return features

# Positional Encoding Module
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# Transformer Decoder for Caption Generation
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, 
                 dim_feedforward=2048, dropout=0.1, max_len=100):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        # Optionally tie weights: self.fc_out.weight = self.embedding.weight

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # tgt: [batch, seq_len]
        tgt = tgt.transpose(0, 1)  # [seq_len, batch]
        embedded = self.embedding(tgt) * math.sqrt(self.d_model)
        embedded = self.pos_encoder(embedded)
        memory = memory.transpose(0, 1)  # [num_tokens, batch, d_model]
        output = self.transformer_decoder(embedded, memory, tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_key_padding_mask)
        output = self.fc_out(output)
        output = output.transpose(0, 1)  # Back to [batch, seq_len, vocab_size]
        return output

# Utility: Generate Subsequent Mask for Transformer Decoder
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask
