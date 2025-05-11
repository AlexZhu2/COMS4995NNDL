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

        # Bart model
        self.bart = BartForConditionalGeneration.from_pretrained(pretrained_model_name)
        
        # Image feature projection
        self.memory_proj = nn.Linear(memory_dim, self.bart.config.d_model)
    
    def _generate_causal_mask(self, seq_len: int, device: torch.device):
        """
        Create causal mask: (tgt_seq_len, tgt_seq_len)
        """
        mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
        return mask

    def forward(
        self,
        tgt_ids: torch.LongTensor,
        memory: torch.FloatTensor,
        decoder_attention_mask=None,
        memory_attention_mask: torch.BoolTensor = None,
        use_cache: bool = False
    ):
        """
        tgt_ids: (batch, tgt_seq_len)
        memory:  (batch, src_seq_len, memory_dim)
        """
        # a) project memory
        memory = self.memory_proj(memory)

        # ✅ If decoder_attention_mask not passed → create from tgt_ids
        if decoder_attention_mask is None:
            decoder_attention_mask = (tgt_ids != self.bart.config.pad_token_id).int()

        # c) call BART decoder
        dec_out = self.bart.model.decoder(
            input_ids=tgt_ids,
            attention_mask=decoder_attention_mask,   # ← now guaranteed to exist
            encoder_hidden_states=memory,
            encoder_attention_mask=memory_attention_mask,
            use_cache=use_cache,
            output_attentions=True,
            return_dict=True 
        )

        # get cross attention for attention mapping
        cross_attns = dec_out.cross_attentions

        # d) project to vocab
        hidden_states = dec_out.last_hidden_state
        logits = self.bart.lm_head(hidden_states)

        return (logits, dec_out.past_key_values, cross_attns) if use_cache else (logits, cross_attns)
