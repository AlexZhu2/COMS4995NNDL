# cnn_bart_captioning_model.py

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from model.encoder.CNN_encoder import CNNEncoder
from model.decoder.transformer_decoder_layer import TransformerDecoder

class CNNBARTCaptioningModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "facebook/bart-base",
        cnn_model_name: str = "efficientnetv2_s",
        embed_dim: int = 512,
        freeze_encoder: bool = True
    ):
        """
        Combines a CNNEncoder with a pretrained BART decoder.
        
        Args:
          pretrained_model_name: HuggingFace checkpoint for BART
          embed_dim:            output dim of your CNNEncoder; must match memory_dim
          freeze_encoder:       if True, will freeze BART’s encoder and embeddings
        """
        super().__init__()
        # 1) Visual encoder
        self.encoder = CNNEncoder(embed_dim=embed_dim, model_name=cnn_model_name)
        
        # 2) Language decoder (BART under the hood)
        #    memory_dim must equal embed_dim
        self.decoder = TransformerDecoder(
            pretrained_model_name=pretrained_model_name,
            memory_dim=embed_dim,
            freeze_encoder=freeze_encoder
        )

    def forward(
        self,
        images: torch.Tensor,
        captions: torch.LongTensor,
        decoder_attention_mask=None,
        memory_attention_mask: torch.BoolTensor = None
    ) -> torch.FloatTensor:
        """
        Training‐time forward.
        
        Args:
          images:                (batch, C, H, W)
          captions:              (batch, tgt_seq_len)  — input IDs (already shifted right)
          memory_attention_mask: (batch, src_seq_len)  — optional padding mask for image tokens
        
        Returns:
          logits: (batch, tgt_seq_len, vocab_size)
        """
        # 1) encode images → (batch, src_seq_len, embed_dim)
        memory = self.encoder(images)
        
        # 2) decode with BART’s decoder
        logits = self.decoder(
            tgt_ids               = captions,
            memory                = memory,
            memory_attention_mask = memory_attention_mask,
            use_cache             = False,
            decoder_attention_mask=decoder_attention_mask
        )
        return logits

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 35,
        num_beams: int = 4
    ) -> torch.LongTensor:
        """
        Inference‐time generation (beam search) using BART’s .generate(...)
        
        Args:
          images:     (batch, C, H, W)
          max_length: maximum number of output tokens
          num_beams:  number of beams for beam search
        
        Returns:
          generated_ids: (batch, generated_seq_len)
        """
        # 1) get CNN features and project them
        memory = self.encoder(images)                                  # (B, S, D)
        proj   = self.decoder.memory_proj(memory)                      # (B, S, D_model)
        enc_out = BaseModelOutput(last_hidden_state=proj)              # wrap for HF API
        
        # 2) call BART’s generation
        return self.decoder.bart.generate(
            input_ids       = None,
            encoder_outputs = enc_out,
            max_length      = max_length,
            num_beams       = num_beams,
            eos_token_id    = self.decoder.bart.config.eos_token_id,
            pad_token_id    = self.decoder.bart.config.pad_token_id,
        )
