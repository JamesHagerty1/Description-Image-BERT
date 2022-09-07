import math
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint



maxlen = 10 # for a sentence
batch_size = 6 # sentences per batch  
epochs = 1     
learning_rate = 0.001 
max_pred = 2  #                                 NOW tokens to mask per sentence (no padding here EVER!)
n_layers = 1 # number of Encoders stacked
n_heads = 12 # for multi head attention
d_model = 64 # embedding size
d_ff = 4 * d_model  # feedforward dim
d_k = d_v = 16  # d_k is for K and Q, d_v is for V


# For now, work with sentences that are as short as 10 words long (maxlen)
text = (
    'Hello, how are you? I am Romeo.\n'
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'
    'Nice to meet you too. How are you today?\n'
    'Great. My baseball team won the competition.\n'
    'Oh Congratulations, Juliet, that is great\n'
    'Thank you, Romeo, I had a lot of fun'
    )

sentences = re.sub("[.,!?\\-]", '', text.lower()).split('\n')  # filter '.', ',', '?', '!'
word_list = list(set(" ".join(sentences).split()))
word_dict = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}

for i, w in enumerate(word_list):
    word_dict[w] = i + 4
number_dict = {} # inverse of word_dict
for k in word_dict: 
    v = word_dict[k]
    number_dict[v] = k
vocab_size = len(word_dict)

token_list = list()
for sentence in sentences:
    arr = [word_dict[s] for s in sentence.split()]
    token_list.append(arr)



def make_batch_entry(orig_sent):
    '''
    return [mask sentence as array of word ids, ,]
    '''
    # For every 4 words, mask one 
    # Since max_pred==5, assume sentences will be no more than 20 words
    orig_sent = orig_sent.split(' ')
    mask_cnt = max_pred                                                         # NEW CHANGE!
    mask_pos = []
    while len(mask_pos) < mask_cnt:
        pos = random.randint(1,len(orig_sent)-1) # Not gonna mask 0th word to avoid pad confusion
        if not pos in mask_pos: mask_pos.append(pos)
    mask_pos.sort()
    mask_tokens = [orig_sent[i] for i in mask_pos]
    mask_token_ids = [word_dict[tok] for tok in mask_tokens]
    mask_sent_token_ids = [word_dict[tok] for tok in orig_sent]
    for pos in mask_pos:
        mask_sent_token_ids[pos] = word_dict['[MASK]']
    while(len(mask_sent_token_ids) < maxlen):
        mask_sent_token_ids.append(0)
    # Batch entry
    # print(orig_sent)
    # print(mask_sent_token_ids)
    # print(mask_tokens)
    # print(mask_token_ids)
    # print(mask_pos)
    return [mask_sent_token_ids, mask_token_ids, mask_pos]


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    # ? Does gelu being a member change how orig model works
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def forward(self, x):
        # x (batch size x sentence maxlen x d_model)
        # which is the result of multi head attention component

        # Just a linear transform which will change last dim
        res = self.fc1(x)
        # (batch size x sentence maxlen x d_ff)

        # will want to understand use of this math later
        res = self.gelu(res)

        # Just a linear transform which will change last dim
        res = self.fc2(res)
        # back to x orig dim, (batch size x sentence maxlen x d_model)

        return res



class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        # Q (batch size x n_heads x sentence maxlen x d_k)
        # K (batch size x n_heads x sentence maxlen x d_k)
        # V (batch size x n_heads x sentence maxlen x d_v)
        # attn_mask (batch size x n_heads x sentence maxlen x sentence maxlen)

        K_T = K.transpose(-1,-2)
        # (batch size x n_heads x d_k x sentence maxlen)

        scores = torch.matmul(Q, K_T) 
        scores /= np.sqrt(d_k) 
        # (batch size x n_heads x sentence maxlen x sentence maxlen) 
        # now is same dims as attn_mask so it can be filled
        scores.masked_fill_(attn_mask, -1e9) # padding idxs per sent masked with -1e9

        attn = nn.Softmax(dim=-1)(scores)
        # attn is scores softmaxxed, (batch size x n_heads x sentence maxlen x sentence maxlen)
        # part of tensor that was changed to -1e9 is 0.0 after softmax, because -1e9 is such a tiny num 

        # dims of attn and V already noted
        context = torch.matmul(attn, V)
        # (batch size x n_heads x sentence maxlen x d_v)

        return context, attn 



class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # These are used to create queries, keys, vectors out of word embeddings
        # nn.Linear arg0 is size of each input sample, arg1 is size of output
        # sample, so for this model that means that inputs ought to have
        # d_model as their last dimension, but after the linear transform of
        # nn.Linear the last dimension will become d_x*n_heads
        # e.g. we'll input matrix of dims (batch size x sentence maxlen x d_model)
        # and output matrix of dims (batch size x sentence maxlen x )
        self.W_Q = nn.Linear(d_model, d_k)
        self.W_K = nn.Linear(d_model, d_k)
        self.W_V = nn.Linear(d_model, d_v)

    def forward(self, Q, K, V, attn_mask):
        # Q, K, V are the same -- token/pos embeddings batch (batch size x sentence maxlen x d_model)
        # to be fed to W_Q, W_K, W_V
        # attn_mask (batch size x sentence maxlen x sentence maxlen)

        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q)
        k_s = self.W_K(K)
        v_s = self.W_V(V)
        # now (batch size x sentence maxlen x d_k|d_v)

        context, attn = ScaledDotProductAttention()(q_s, k_s, v_s, attn_mask)


        # to change!
        # context is (batch size x n_heads x sentence maxlen x d_v)
        # attn is (batch size x n_heads x sentence maxlen x sentence maxlen)

        context = context.transpose(1, 2)
        context = context.contiguous() # returns "same" context tensor but now contiguous in memory
        # context is (batch size x sentence maxlen x n_heads x d_v)

        # compress from 4 dims back to 3 dims (I think we only had 4 for the sake of multi head attn)
        context = context.view(batch_size, -1, n_heads * d_v) 
        # context is (batch size x sentence maxlen x n_heads*d_v)

        # remember, nn.Linear is just a transform; here arg0 tells us that the
        # last dim (tensor with floats dim) is n_heads*d_v, and that we want the
        # last dim to be transformed to be of dim d_model now
        output = nn.Linear(n_heads*d_v, d_model)(context)
        # output is (batch size x sentence maxlen x d_model)

        # Again, LayerNorm will just tug floats closer to 0, and arg0==d_model tells
        # LayerNorm the last dim to expect from input
        # use of residual part of the math overall to learn
        output = nn.LayerNorm(d_model)(output + residual)
        # still (batch size x sentence maxlen x d_model)

        return output, attn 



class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs (batch size x sentence maxlen x d_model)  res of emb layer
        # enc_self_attn_mask (batch size x sentence maxlen x sentence maxlen)

        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V 
        # enc_output (batch size x sentence maxlen x d_model)
        # attn (batch size x n_heads x sentence maxlen x sentence maxlen)
        
        enc_outputs = self.pos_ffn(enc_outputs) 
        # (batch size x sentence maxlen x d_model)

        return enc_outputs, attn



class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        # nn.Embedding creates a lookup table of vectors, where arg0 is how
        # many unique vectors there should be and arg1 is the dimension of the
        # vectors
        # (2.1)
        # There should be vocab_size unique token embeddings
        self.tok_embed = nn.Embedding(vocab_size, d_model)  
        # (1.1)
        # There should be maxlen unique positional embeddings
        self.pos_embed = nn.Embedding(maxlen, d_model)   
        # use of one arg in nn.LayerNorm means "normalize over the last dim
        # which is expected to be of arg==d_model size",
        # appears to tug the floats closer to 0 (when you pass a matrix to it)
        # which is probably what normalization is
        # (3.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x dim:    (batch size x sentence maxlen)
        # essentially our sentences but with masks and padding and where we have
        # integer ids for tokens instead of tokens

        seq_len = x.size(1)
        # (1) create positional embeddings
        # they need to be same dims as / line up with x which is our batch of
        # masked sentences where tokens are their ids (integers)
        # Create     pos = tensor([0,1,...seq_len-2,seq_len-1]) to line up to
        # an arbitrary token id masked sentence
        pos = torch.arange(seq_len, dtype=torch.long) 
        # copying pos   pos' = [pos,...pos] where len(pos') == batch_size
        # so we have positional embeddings for all sentences of batch x
        pos = pos.unsqueeze(0).expand_as(x)
        # encode indices [0,maxlen-1] 
        pos_embeddings = self.pos_embed(pos) 

        # (2) token embeddings
        tok_embeddings = self.tok_embed(x)
        
        # x and pos were (batch size x sentence maxlen), now their corresponding
        # embeddings matrix is (batch size x sentence maxlen x d_model)

        # (3) sum pos and token embeddings then normalize them
        embedding = tok_embeddings + pos_embeddings
        output = self.norm(embedding)    

        # output is (batch size x sentence maxlen x d_model), just like 
        # embedding (all it did was normalize embedding's floats)

        return output



class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

        # decoder is shared with embedding layer
        self.decoder = nn.Linear(d_model, vocab_size, bias=False)
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        

    # ? Does gelu being a member change how orig model works
    def gelu(self, x):
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

    def get_attn_pad_mask(self, input_ids):
        # mask used so that attention DOES NOT consider 0 paddings of token id sents
        # pad_attn_mask is our batch but now all integers are bools, where 0
        # (padding) are True and all other tokens are False
        pad_attn_mask = input_ids.data.eq(0).unsqueeze(1) 

        # pad_attn_mask is (batch size x sentence maxlen), same as our batch

        # took old pad_attn_mask but now, per batch entry, there is an array of maxlen
        # repeated arrays of our Falses and Trues (where those arrays are also
        # of len maxlen)
        pad_attn_mask_ = pad_attn_mask.expand(batch_size, maxlen, maxlen)  

        # pad_attn_mask_ is (batch size x sentence maxlen x sentence maxlen)

        return pad_attn_mask_

    def forward(self, input_ids, masked_pos):

        # input_ids is (batch size x sentence maxlen)
        # masked_pos is (batch size x mask maxcnt/max_pred)

        # create embeddings for batch entries that encode tokens + positions
        output = self.embedding(input_ids)     
        
        # output is (batch size x sentence maxlen x d_model)

        enc_self_attn_mask = self.get_attn_pad_mask(input_ids)

        # enc_self_attn_mask is (batch size x sentence maxlen x sentence maxlen)

        for layer in self.layers:
            output, _ = layer(output, enc_self_attn_mask)                   # ? does EncoderLayer() return redundant second item
            # output remains (batch size x sentence maxlen x d_model)

        # Deal with masked words now
        masked_pos = masked_pos[:,:,None]
        # (batch size x max_pred/mask maxcnt x 1)
        # still same contents of input masked_pos, but new ordering

        masked_pos = masked_pos.expand(-1, -1, output.size(-1))
        # (batch size x max_pred/mask maxcnt words  x d_model)
        # the d_model vector is just an integer repeating d_model times across
        # the array, same integers (token ids) from our input masked_pos

        # Gather the words at our masked_pos idxs from output, 
        # torch.gather args
        # arg0 "source tensor"
        # arg1 "axis along which to index"
        # arg2 "indices of elements to gather"
        h_masked = torch.gather(output, 1, masked_pos)

        # h_masked same dims as masked_pos (batch size x max_pred/mask maxcnt x d_model)

        # You are already familiar with the fns used below from previous components
        h_masked = self.linear(h_masked)
        # still (batch size x max_pred/mask maxcnt x d_model)
        h_masked = self.gelu(h_masked)
        h_masked = self.norm(h_masked)
        # still (batch size x max_pred/mask maxcnt x d_model)

        # self.decoder is a Linear(d_model, vocab_size)
        logits_lm = self.decoder(h_masked) 
        # (batch size x max_pred x vocab_size)

        # Will want to learn usefulness of this final bias
        # decoder bias dim (vocab_size)
        # currently assuming it just adds to the vocab_size last dim arrays of logits_lm
        # because they are of the same dimensionality, vocab_size and vocab_size
        logits_lm += self.decoder_bias 
        
        return logits_lm



model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

batch = []
for i in range(batch_size):
    batch.append( make_batch_entry(sentences[i%len(sentences)]) )
input_ids, masked_tokens, masked_pos = map(torch.LongTensor, zip(*batch))

for epoch in range(epochs): 
    # Training boilerplate
    optimizer.zero_grad()

    logits_lm = model(input_ids, masked_pos)
    # (batch size x max_pred x vocab_size)

    logits_lm = logits_lm.transpose(1, 2)
    # (batch size x vocab_size x max_pred)
    
    # Hold up logits_lm (predictions for masked tokens) to the official masked_tokens
    # masked_tokens (batch_size x max_pred)
    loss_lm = criterion(logits_lm, masked_tokens) 

    # Does not change loss_lm value, seems to be training boilerplate
    loss_lm = (loss_lm.float()).mean()
    
    # Previously next sentence pred was part of the loss too, not any more
    loss = loss_lm
    
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    # More training boilerplate
    loss.backward()
    optimizer.step()
print('')



# # Predictions on training set
# with torch.no_grad():
#     logits_lm = model(input_ids, masked_pos)
#     batch_preds = logits_lm.data.max(2)[1]

#     # print(input_ids)
#     # print(masked_pos)
#     # # compare following two against each other
#     print(masked_tokens)
#     print(batch_preds)
#     print('')

#     input_ids = input_ids.tolist()
#     masked_pos = masked_pos.tolist()
#     masked_tokens = masked_tokens.tolist()
#     batch_preds = batch_preds.tolist()

#     for i in range(len(input_ids)):
#         token_id_sent = input_ids[i]
#         sent_masked_pos = masked_pos[i]
#         sent_masked_tok = masked_tokens[i]
#         sent_pred_tok = batch_preds[i]

#         masked_sent = ' '.join(number_dict[id] for id in token_id_sent)
#         for j in range(max_pred):
#             token_id_sent[ sent_masked_pos[j] ] = sent_masked_tok[j]
#         orig_sent = ' '.join(number_dict[id] for id in token_id_sent)
#         for j in range(max_pred):
#             token_id_sent[ sent_masked_pos[j] ] = sent_pred_tok[j]
#         pred_sent = ' '.join(number_dict[id] for id in token_id_sent)

#         print(masked_sent)
#         print(orig_sent)
#         print(pred_sent)
#         print('')