import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, key_size, model_size, use_softmax=True, use_non_lin_mix=False, use_bias=True, sum_normalization=False):
        """
        Multi-Head Attention module.
        
        Args:
            num_heads (int): Number of attention heads.
            key_size (int): Dimension of keys/queries.
            model_size (int): Dimension of the model (input and output).
            use_softmax (bool): Whether to apply softmax to attention scores.
            use_non_lin_mix (bool): Whether to apply non-linear mixing after attention.
            use_bias (bool): Whether to use bias in linear projections.
            sum_normalization (bool): Whether to normalize attention scores by their sum.
        """
        super(MultiHeadAttention, self).__init__()
        assert model_size % num_heads == 0, "model_size must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.key_size = key_size
        self.model_size = model_size
        self.use_softmax = use_softmax
        self.use_non_lin_mix = use_non_lin_mix
        self.sum_normalization = sum_normalization
        
        self.query = nn.Linear(model_size, num_heads * key_size, bias=use_bias)
        self.key = nn.Linear(model_size, num_heads * key_size, bias=use_bias)
        self.value = nn.Linear(model_size, num_heads * key_size, bias=use_bias)
        self.out = nn.Linear(num_heads * key_size, model_size, bias=use_bias)
        
        self.scale = math.sqrt(key_size)
    
    def forward(self, query, key, value):
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            query (Tensor): Shape (batch_size, seq_length, model_size)
            key (Tensor): Shape (batch_size, seq_length_k, model_size)
            value (Tensor): Shape (batch_size, seq_length_v, model_size)
        
        Returns:
            Tensor: Output after attention. Shape (batch_size, seq_length, model_size)
            Tensor: Attention weights. Shape (batch_size, num_heads, seq_length, seq_length_k)
        """
        batch_size, seq_length, _ = query.size()
        
        # Linear projections
        Q = self.query(query).view(batch_size, seq_length, self.num_heads, self.key_size).transpose(1, 2)  # (batch, heads, seq, key)
        K = self.key(key).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)           # (batch, heads, seq_k, key)
        V = self.value(value).view(batch_size, -1, self.num_heads, self.key_size).transpose(1, 2)       # (batch, heads, seq_v, key)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq, seq_k)
        
        if self.use_softmax:
            attn = F.softmax(scores, dim=-1)  # (batch, heads, seq, seq_k)
        else:
            attn = scores  # Apply other normalization if needed
        
        if self.sum_normalization:
            attn = attn / attn.sum(dim=-1, keepdim=True)
        
        context = torch.matmul(attn, V)  # (batch, heads, seq, key)
        
        if self.use_non_lin_mix:
            context = F.relu(context)  # Example non-linearity
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)  # (batch, seq, heads*key)
        out = self.out(context)  # (batch, seq, model_size)
        
        return out, attn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, widening_factor=4, use_bias=True, activation=F.relu):
        """
        Simple Multi-Layer Perceptron.
        
        Args:
            input_dim (int): Input feature dimension.
            output_dim (int): Output feature dimension.
            widening_factor (int): Factor to expand hidden layers.
            use_bias (bool): Whether to use bias in linear layers.
            activation (callable): Activation function.
        """
        super(MLP, self).__init__()
        hidden_dim = input_dim * widening_factor
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        self.activation = activation
    
    def forward(self, x):
        """
        Forward pass for MLP.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Output tensor after MLP.
        """
        return self.fc2(self.activation(self.fc1(x)))

class TokenVocab(nn.Module):
    def __init__(self, vocab_size, embed_dim, init_scale=0.01):
        """
        Token Vocabulary module using Embedding.
        
        Args:
            vocab_size (int): Size of the vocabulary.
            embed_dim (int): Dimension of each embedding vector.
            init_scale (float): Scale for initializing embeddings.
        """
        super(TokenVocab, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -init_scale, init_scale)
    
    def forward(self, x, logits=False):
        """
        Forward pass for TokenVocab.
        
        Args:
            x (Tensor): Input token indices. Shape (batch, seq_length)
            logits (bool): If True, return raw embeddings. Otherwise, return normalized embeddings.
        
        Returns:
            Tensor: Embedding vectors.
        """
        if logits:
            return self.embedding(x)  # Raw logits
        else:
            return self.embedding(x)  # Can apply normalization if needed

class LNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        """
        Layer Normalization.
        
        Args:
            normalized_shape (int or list): Input shape from an expected input.
            eps (float): A value added to the denominator for numerical stability.
            elementwise_affine (bool): If True, this module has learnable affine parameters.
        """
        super(LNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)
    
    def forward(self, x):
        """
        Forward pass for LayerNorm.
        
        Args:
            x (Tensor): Input tensor.
        
        Returns:
            Tensor: Normalized tensor.
        """
        return self.layer_norm(x)

def create_pos_encoding(seq_length, pe_size, flip=False):
    """
    Create sinusoidal positional encoding.
    
    Args:
        seq_length (int): Length of the sequence.
        pe_size (int): Dimension of positional encoding.
        flip (bool): Whether to flip the positional encoding.
    
    Returns:
        Tensor: Positional encoding matrix of shape (seq_length, pe_size)
    """
    pe = torch.zeros(seq_length, pe_size)
    position = torch.arange(0, seq_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, pe_size, 2).float() * (-math.log(10000.0) / pe_size))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if flip:
        pe = torch.flip(pe, [0])
    return pe  # Shape: [seq_length, pe_size]