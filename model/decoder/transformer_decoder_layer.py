import torch
import torch.nn as nn
from positional_encoding import SinusoidalPositionalEncoding

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, num_layers=6, max_len=50, dropout=0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_len = max_len
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Positional encoding
        self.positional_encoding = SinusoidalPositionalEncoding(embed_dim, max_len)
        
        # Transformer Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Final output projection to vocab size
        self.output_proj = nn.Linear(embed_dim, vocab_size)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Args:
            tgt: (batch_size, tgt_seq_len) -- token ids
            memory: (batch_size, src_seq_len, embed_dim) -- image features
            tgt_mask: (tgt_seq_len, tgt_seq_len) -- to prevent attending to future tokens
            memory_mask: (batch_size, tgt_seq_len, src_seq_len) -- optional
        Returns:
            output: (batch_size, tgt_seq_len, vocab_size)
        """
        
        # Embed tokens and add positional embeddings
        tgt_embeddings = self.token_embed(tgt)
        tgt_embeddings = self.positional_encoding(tgt_embeddings)
        # Transformer decoding
        output = self.transformer_decoder(tgt_embeddings, memory, tgt_mask=tgt_mask, memory_key_padding_mask=None)
        
        # Project to vocabulary
        output = self.output_proj(output)
        
        return output

    def generate_square_subsequent_mask(self, sz):
        """Generate a causal mask for decoder self-attention"""
        mask = torch.triu(torch.ones((sz, sz)) == 1, diagonal=1)
        mask = mask.float().masked_fill(mask == 1, float('-inf'))
        return mask
