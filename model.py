import torch
import torch.nn as nn


################################################################################


class BERT(nn.Module):
    def __init__(self, c):
        super(BERT, self).__init__()
        self.embedding_layer = EmbeddingLayer(c)
        self.encoder_layers = \
            nn.ModuleList([EncoderLayer(c) for _ in range(c.n_layers)])
        self.c = c # config

    def attn_pad_mask(self, x):
        """Used to hide [PAD] tokens during attention"""
        pad_token_id = self.c.pad_token_id
        batch_size = self.c.batch_size
        seq_len = self.c.seq_len
        # pad_mask: (batch_size, seq_len, seq_len)
        # Each sequence's mask is a matrix repeating the "[PAD]" mask for that
        # sequence at the vector level (repeating rows)
        pad_mask = x.data.eq(pad_token_id).unsqueeze(1). \
            expand(batch_size, seq_len, seq_len) 
        return pad_mask

    def forward(self, x, y_i):
        # attn_pad_mask: (batch_size, seq_len, seq_len)
        attn_pad_mask = self.attn_pad_mask(x)
        # x: (batch_size, seq_len, d_model)
        # Sequences are now embeddings representing tokens and their positiions
        x = self.embedding_layer(x)
        # for layer in self.layers:
        #     x, attn = layer(x, attn_pad_mask)  
        return -1


class EmbeddingLayer(nn.Module):
    def __init__(self, c):
        super(EmbeddingLayer, self).__init__()
        self.pos_emb = nn.Embedding(c.seq_len, c.d_model) 
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.norm = nn.LayerNorm(c.d_model)
        self.c = c

    def forward(self, x):
        seq_len = self.c.seq_len
        # x_pos_embs: (batch_size, seq_len)
        # Repeating rows of [0-seq_len-1]
        x_pos = torch.arange(seq_len, dtype=torch.long). \
            unsqueeze(0).expand_as(x)
        # x_pos_embs: (batch_size, seq_len, d_model)
        x_pos_embs = self.pos_emb(x_pos)
        # x_tok_embs: (batch_size, seq_len, d_model)
        x_tok_embs = self.tok_emb(x)
        # x: (batch_size, seq_len, d_model)
        # Concat embeddings representing token positions and tokens themselves
        x = x_pos_embs + x_tok_embs
        return self.norm(x)


class EncoderLayer(nn.Module):
    def __init__(self, c):
        super(EncoderLayer, self).__init__()
        self.attn_layer = AttentionLayer(c)
        # self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, x, attn_pad_mask):
        x, attn = self.attn_layer(x, x, x, attn_pad_mask)
        return -1


class AttentionLayer():
    """Only has one attention head for now"""
    def __init__(self, c):
        super(AttentionLayer, self).__init__()
        self.W_Q = nn.Linear(c.d_model, c.d_k)
        self.W_K = nn.Linear(c.d_model, c.d_k)
        self.W_V = nn.Linear(c.d_model, c.d_v)

    def forward(self):
        return -1