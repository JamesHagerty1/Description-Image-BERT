import torch
import torch.nn as nn


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def get_attn_pad_mask(self, input_ids):
        pad_token_id = -1               
        pad_attn_mask = input_ids.data.eq(pad_token_id).unsqueeze(1) 
        return pad_attn_mask_

    def forward(self, input_ids, masked_pos):
        output = self.embedding(input_ids)     
        enc_self_attn_mask = self.get_attn_pad_mask(input_ids)
        for layer in self.layers:
            output, attn = layer(output, enc_self_attn_mask)                  
        if masked_pos == None:
            return None, attn
        masked_pos = masked_pos[:,:,None]
        masked_pos = masked_pos.expand(-1, -1, output.size(-1))
        h_masked = torch.gather(output, 1, masked_pos)
        h_masked = self.linear(h_masked)
        h_masked = self.gelu(h_masked)
        h_masked = self.norm(h_masked)                                    
        logits_lm = self.decoder(h_masked) 
        logits_lm += self.decoder_bias 
        return logits_lm, attn


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)  
        self.pos_embed = nn.Embedding(maxlen, d_model)   
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long) 
        pos = pos.unsqueeze(0).expand_as(x)
        pos_embeddings = self.pos_embed(pos) 
        tok_embeddings = self.tok_embed(x)
        output = tok_embeddings + pos_embeddings
        output = self.norm(output)                                            
        return output
