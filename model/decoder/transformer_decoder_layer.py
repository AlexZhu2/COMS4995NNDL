import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

class TransformerDecoder(nn.Module):
    def __init__(
            self,
            pretrained_model_name: str = "facebook/bart-base",
            memory_dim: int = 512,
            freeze_encoder: bool = True
    ):
        super().__init__()

        # Bart Encoder
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model_name)
        
        # Image Feature Dim -> Language Embedding Dim
        self.memory_proj = nn.Linear(memory_dim, self.bart.config.d_model)
    
    def forward(
        self,
        tgt_ids: torch.LongTensor,
        memory: torch.FloatTensor,
        memory_attention_mask: torch.BoolTensor = None,
        use_cache: bool = False
    ):
        """
        tgt_ids: (batch, tgt_seq_len)
        memory:  (batch, src_seq_len, memory_dim)
        """
        # a) project memory into BART’s hidden size
        memory = self.memory_proj(memory)  
        
        # b) call BART’s decoder using input_ids → it will
        #    (i) embed tokens via the shared vocab embedding,
        #    (ii) add learned positional embeddings internally,
        #    (iii) do self- & cross-attention.
        dec_out = self.bart.model.decoder(
            input_ids=tgt_ids,
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_attention_mask,
            use_cache=use_cache
        )
        
        # c) grab the last hidden states and project to vocab
        hidden_states = dec_out.last_hidden_state                   # (batch, tgt_len, d_model)
        logits = self.bart.lm_head(hidden_states)                   # (batch, tgt_len, vocab_size)
        
        return (logits, dec_out.past_key_values) if use_cache else logits