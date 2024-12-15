import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, List

from attn import MultiHeadAttention, MLP, TokenVocab, create_pos_encoding, LNorm

@dataclass
class TransformerConfig:
    num_heads: int = 2
    widening_factor: int = 4
    num_layers: int = 3
    key_size: int = 5
    embedding_size: int = 64
    output_size: int = 1
    in_context_length: int = 17
    in_context_length_test: int = 17
    test_points: int = 1
    dropout_rate: float = 0.0
    only_attention: bool = True
    use_layer_norm: bool = True
    use_pe: bool = True
    pe_size: int = 6
    concat_pe: bool = False
    output_mapping: bool = False
    input_mapping: bool = False
    use_bias_p: bool = True
    zero_embeddings: bool = False
    deq: bool = True
    init_scale: float = 0.02
    use_softmax: bool = False
    use_non_lin_mix: bool = False
    first_layer_sm: bool = False
    y_update: bool = False
    input_mlp: bool = False
    input_mlp_out_dim: int = 0
    gd_mlp_config: bool = False
    sum_norm: bool = False
    dampening: float = 1.0
    clip: float = 0.0
    ana_copy: bool = False
    flip: bool = False
    vocab_size: int = 0
    vocab_token_dim: int = 0
    vocab_init: float = 0.01
    return_logits: bool = False
    include_query: bool = False
    name: Optional[str] = None

class TransformerLayer(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Single Transformer Layer comprising Multi-Head Attention and MLP with optional normalization and dropout.
        """
        super(TransformerLayer, self).__init__()
        self.config = config
        
        self.attention = MultiHeadAttention(
            num_heads=config.num_heads,
            key_size=config.key_size,
            model_size=config.embedding_size,
            use_softmax=config.use_softmax,
            use_non_lin_mix=config.use_non_lin_mix,
            use_bias=config.use_bias_p,
            sum_normalization=config.sum_norm
        )
        
        if not config.only_attention:
            self.mlp = MLP(
                input_dim=config.embedding_size,
                output_dim=config.embedding_size,
                widening_factor=config.widening_factor,
                use_bias=config.use_bias_p
            )
        
        if config.use_layer_norm:
            self.lnorm1 = LNorm(config.embedding_size)
            self.lnorm2 = LNorm(config.embedding_size)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.dampening = config.dampening
        self.clip = config.clip
    
    def forward(self, h: torch.Tensor, is_training: bool, include_query: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Transformer layer.
        
        Args:
            h (Tensor): Input tensor. Shape (batch_size, seq_length, embedding_size)
            is_training (bool): Indicates if the model is in training mode.
            include_query (bool): Whether to include the query vector in attention.
        
        Returns:
            Tuple[Tensor, Tensor]: Output tensor and attention weights.
        """
        residual = h
        if self.config.use_layer_norm:
            h = self.lnorm1(h)
        
        if not include_query:
            key = h[:, :-1, :]  # Exclude the last token for key and value
            value = h[:, :-1, :]
        else:
            key = h
            value = h
        
        attn_out, att_map = self.attention(h, key, value)
        attn_out = self.dropout(attn_out)
        h = residual + self.dampening * attn_out
        
        if self.clip > 0:
            h = torch.clamp(h, -self.clip, self.clip)
        
        if not self.config.only_attention:
            residual = h
            if self.config.use_layer_norm:
                h = self.lnorm2(h)
            mlp_out = self.mlp(h)
            mlp_out = self.dropout(mlp_out)
            h = residual + self.dampening * mlp_out
            
            if self.clip > 0:
                h = torch.clamp(h, -self.clip, self.clip)
        
        return h, att_map

class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        """
        Flexible Transformer implementation.
        """
        super(Transformer, self).__init__()
        self.config = config
        
        # Positional Encoding
        if self.config.use_pe and self.config.pe_size > 0:
            pos_encoding = create_pos_encoding(self.config.in_context_length, self.config.pe_size, self.config.flip)
            pos_encoding_test = create_pos_encoding(self.config.in_context_length_test, self.config.pe_size, self.config.flip)
            self.register_buffer('pos_encoding', pos_encoding)  # Shape: (seq_length, pe_size)
            self.register_buffer('pos_encoding_test', pos_encoding_test)
        else:
            self.pos_encoding = None
            self.pos_encoding_test = None
        
        # Embedding Layer
        if self.config.input_mapping:
            self.embeddings = nn.Linear(self.config.embedding_size, self.config.embedding_size, bias=self.config.use_bias_p)
        else:
            self.embeddings = nn.Identity()
        
        # Input MLP
        if self.config.input_mlp:
            self.input_mlp_layer = MLP(
                input_dim=self.config.embedding_size,
                output_dim=self.config.input_mlp_out_dim,
                widening_factor=self.config.widening_factor,
                use_bias=self.config.use_bias_p
            )
        
        # Transformer Layers
        self.layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Output Layer
        if self.config.output_mapping:
            self.output_layer = nn.Linear(self.config.embedding_size, self.config.output_size)
        else:
            self.output_layer = nn.Identity()
        
        # Token Vocabulary
        if self.config.return_logits and self.config.vocab_size > 0:
            self.vocab = TokenVocab(
                vocab_size=self.config.vocab_size,
                embed_dim=self.config.vocab_token_dim,
                init_scale=self.config.vocab_init
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """
        Initialize weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -self.config.vocab_init, self.config.vocab_init)
    
    def forward(self, x: torch.Tensor, is_training: bool = True, predict_test: bool = False) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass for the Transformer model.
        
        Args:
            x (Tensor): Input tensor. Shape (batch_size, seq_length, embedding_size) or (batch_size, seq_length)
            is_training (bool): Indicates if the model is in training mode.
            predict_test (bool): Indicates if test predictions are being made.
        
        Returns:
            Tuple[Tensor, List[Tensor], List[Tensor]]:
                - Output tensor. Shape depends on `output_mapping` and `return_logits`.
                - List of intermediate hidden states.
                - List of attention maps.
        """
        # Token Embedding
        if self.config.vocab_size > 0 and self.config.vocab_token_dim > 0:
            x = self.vocab(x)  # Shape: (batch, seq, embed_dim)
        
        # Dropout Rate Adjustment
        dropout_rate = self.config.dropout_rate if is_training else 0.0
        
        # Input Mapping
        embeddings = self.embeddings(x)
        
        # Input MLP
        if self.config.input_mlp:
            embeddings = embeddings + self.input_mlp_layer(embeddings)
        
        # Positional Encoding
        if self.config.use_pe:
            if self.config.concat_pe:
                if predict_test:
                    pos_encoding = self.pos_encoding_test.unsqueeze(0).repeat(embeddings.size(0), 1, 1)
                    pos_encoding = pos_encoding * 0 if self.config.zero_embeddings else pos_encoding
                    h = torch.cat([embeddings, pos_encoding], dim=2)
                else:
                    pos_encoding = self.pos_encoding.unsqueeze(0).repeat(embeddings.size(0), 1, 1)
                    pos_encoding = pos_encoding * 0 if self.config.zero_embeddings else pos_encoding
                    h = torch.cat([embeddings, pos_encoding], dim=2)
            else:
                if predict_test:
                    h = self.pos_encoding_test + embeddings
                else:
                    h = self.pos_encoding + embeddings
        else:
            h = embeddings
        
        stack_h = []
        stack_att = []
        
        for layer in self.layers:
            h, att_map = layer(h, is_training, self.config.include_query)
            stack_h.append(h)
            stack_att.append(att_map)
        
        out = self.output_layer(h)
        
        if self.config.return_logits:
            out = self.vocab(out, logits=True)
        
        return out, stack_h, stack_att