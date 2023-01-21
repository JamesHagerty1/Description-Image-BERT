import torch
import torch.nn as nn


################################################################################


class BERT(nn.Module):
    def __init__(self, c):
        super(BERT, self).__init__()
        self.embedding_layer = EmbeddingLayer()
        self.c = c # config

    def attn_pad_mask(self, x):
        """Used to hide [PAD] tokens during attention"""
        pad_token_id = self.c.pad_token_id
        batch_size = self.c.batch_size
        seq_len = x[0].shape[0]
        # pad_mask: (batch_size, seq_len, seq_len)
        # Each sequence's mask is a matrix repeating the "[PAD]" mask for that
        # sequence at the vector level (repeating rows)
        pad_mask = x.data.eq(pad_token_id).unsqueeze(1). \
            expand(batch_size, seq_len, seq_len) 
        return pad_mask

    def forward(self, x, y_i):
        # x: (batch_size, seq_len)
        # y_i: (batch_size, desc_len)
        # attn_pad_mask: (batch_size, seq_len, seq_len)
        attn_pad_mask = self.pad_mask(x)
        x_embs = self.embedding_layer(x)

        return -1