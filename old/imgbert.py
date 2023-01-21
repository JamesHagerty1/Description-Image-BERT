import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pprint import pprint
import pickle
from PIL import Image, ImageDraw



dbfile = open('img_data_pkl', 'rb')
db = pickle.load(dbfile)
data = db['data']
tokens_to_ids = db['tokens_to_ids']
ids_to_tokens = db['ids_to_tokens']
batch = db['batch']
print(len(batch), len(batch[0]), len(batch[0][0]), len(batch[0][1]), len(batch[0][2]))
input_ids, masked_tokens, masked_pos = map(torch.LongTensor, zip(*batch))
dbfile.close()
print(len(input_ids), len(masked_tokens), len(masked_pos))
print(input_ids.shape)


epochs = 100000000000
# learning_rate = 0.001 # using optimizer default lr
max_pred = 1  #         

n_layers = 1 # number of Encoders stacked
d_model = 64 # embedding size
d_ff =  d_model  # feedforward dim
d_k = d_v = 16  # d_k is for K and Q, d_v is for V

vocab_size = len(ids_to_tokens)
samples = len(data) * 2
batch_size = samples #// 8
maxlen = input_ids.shape[1]



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
        # Q (batch size x sentence maxlen x d_k)
        # K (batch size x sentence maxlen x d_k)
        # V (batch size x sentence maxlen x d_v)
        # attn_mask (batch size x sentence maxlen x sentence maxlen)
        # but it's generalizable that with multi head, this fn can stay same
        # since before the final two dimensions of matrix, torch batches 

        K_T = K.transpose(-1,-2)
        # (batch size x d_k x sentence maxlen)

        scores = torch.matmul(Q, K_T) 
        scores /= np.sqrt(d_k) 
        # (batch size x sentence maxlen x sentence maxlen) 
        # now is same dims as attn_mask so it can be filled
        scores.masked_fill_(attn_mask, -1e9) # padding idxs per sent masked with -1e9

        attn = nn.Softmax(dim=-1)(scores)
        # attn is scores softmaxxed, (batch size x sentence maxlen x sentence maxlen)
        # part of tensor that was changed to -1e9 is 0.0 after softmax, because -1e9 is such a tiny num 

        # dims of attn and V already noted
        context = torch.matmul(attn, V)
        # (batch size x sentence maxlen x d_v)

        return context, attn 



class SingleHeadAttention(nn.Module):
    def __init__(self):
        super(SingleHeadAttention, self).__init__()
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
        # context is (batch size x sentence maxlen x d_v)
        # attn is (batch size x sentence maxlen x sentence maxlen)

        # ? I wonder why this layer isn't a member / what difference it makes differentiation-wise
        output = nn.Linear(d_v, d_model)(context)
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
        self.enc_self_attn = SingleHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # enc_inputs (batch size x sentence maxlen x d_model)  res of emb layer
        # enc_self_attn_mask (batch size x sentence maxlen x sentence maxlen)

        enc_outputs, attn = self.enc_self_attn(
            enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask) # enc_inputs to same Q,K,V 
        # enc_output (batch size x sentence maxlen x d_model)
        # attn (batch size x sentence maxlen x sentence maxlen)
        
        enc_outputs = self.pos_ffn(enc_outputs) 
        # (batch size x sentence maxlen x d_model)

        return enc_outputs, attn # return attention just to look at it!



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
        output = tok_embeddings + pos_embeddings
        output = self.norm(output)                                            

        # output is (batch size x sentence maxlen x d_model), just like 
        # embedding (all it did was normalize floats)

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

        pad_token_id = -1                 # currently using a dict that has no pad token / pad token id

        pad_attn_mask = input_ids.data.eq(pad_token_id).unsqueeze(1) 
        print("YO")
        print(pad_attn_mask.shape, input_ids.shape)
        # pad_attn_mask is (batch size x sentence maxlen), same as our batch

        # took old pad_attn_mask but now, per batch entry, there is an array of maxlen
        # repeated arrays of our Falses and Trues (where those arrays are also
        # of len maxlen)
        pad_attn_mask_ = pad_attn_mask.expand(batch_size, maxlen, maxlen)  
        print(pad_attn_mask_.shape)

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
            output, attn = layer(output, enc_self_attn_mask)                   # ? does EncoderLayer() return redundant second item
            # output remains (batch size x sentence maxlen x d_model)

        # for cases where I only want to view attn, and no inference
        if masked_pos == None:
            return None, attn

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

        
        return logits_lm, attn



def train():
    model = BERT()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adagrad(model.parameters())
    
    min_avg_loss = 5

    for epoch in range(epochs): 
        avg_loss = 0

        for iter in range( samples // batch_size ):

            input_ids_ = input_ids[iter*batch_size:iter*batch_size+batch_size]
            masked_pos_ = masked_pos[iter*batch_size:iter*batch_size+batch_size]
            masked_tokens_ = masked_tokens[iter*batch_size:iter*batch_size+batch_size]


            # Training boilerplate
            optimizer.zero_grad()

            logits_lm, _ = model(input_ids_, masked_pos_)
            # (batch size x max_pred x vocab_size)


            logits_lm = logits_lm.transpose(1, 2)
            # (batch size x vocab_size x max_pred)

            # Hold up logits_lm (predictions for masked tokens) to the official masked_tokens
            # masked_tokens (batch_size x max_pred)
            loss_lm = criterion(logits_lm, masked_tokens_) 

            # Does not change loss_lm value, seems to be training boilerplate
            loss_lm = (loss_lm.float()).mean()

            with torch.no_grad():
                avg_loss += loss_lm.item()

            # More training boilerplate
            loss_lm.backward()
            optimizer.step()

            return # TEMP

        avg_loss /= (samples // batch_size)
        
        print('Epoch:', '%12d' % (epoch + 1), 'cost =', '{:.6f}'.format(avg_loss))

        if avg_loss < 0.2 and avg_loss < min_avg_loss:
            torch.save(model, 'ImgBert')
        if avg_loss < min_avg_loss:
            min_avg_loss = avg_loss
        if avg_loss < 0.02:
            break

        # sanity check
        if epoch % 100 == 0:
            f = open('imgbert.txt', 'w')
            s = f'Epoch {epoch+1}, min_avg_loss {min_avg_loss}, avg_loss {avg_loss}'
            f.write(s)
            f.close()
        


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def draw_attn(attn, img_name, pixels, img_w, img_h, img_word_len, txt_len):
    print(img_name)
    
    scale = 10
    canvas_pad = 80

    canvas_w = txt_len*img_w*scale + canvas_pad*(txt_len+1)
    canvas_h = img_h*scale + canvas_pad*2

    img = Image.new('RGB', (canvas_w, canvas_h), (230, 230, 230))
    draw = ImageDraw.Draw(img)

    # reconstruct imgs
    for i in range(txt_len):
        x = canvas_pad + canvas_pad*i + img_w*scale*i
        y = canvas_pad
        for j in range(len(pixels)):
            rgb = (255, 255, 255) if pixels[j] == '1' else (0, 0, 0)
            row = j // img_h
            col = j % img_w

            draw.rectangle([x+col*scale, y+row*scale,  x+col*scale+scale-1, y+row*scale+scale-1],
                fill=rgb)

    #
    r, g, b = 255, 162, 0

    for i in range(txt_len):
        x = canvas_pad + canvas_pad*i + img_w*scale*i
        y = canvas_pad
        attn_row = attn[i]

        for j in range(len(pixels) // img_word_len):
            attn_score = attn_row[j+txt_len].item() * 1000
            attn_score = min(255, int(attn_score))
            
            if attn_score == 0:
                continue

            tint_factor = 1 - attn_score / 255
            tint_factor *= 0.6
            r_ = int(r + (255 - r) * tint_factor)
            g_ = int(g + (255 - g) * tint_factor)
            b_ = int(b + (255 - b) * tint_factor)

            col = j % (img_w // img_word_len)
            row = j // (img_w // img_word_len)

            x0 = x+col*img_word_len*scale 
            y0 = y + row*scale
            x1 = x+col*img_word_len*scale + img_word_len*scale - 1 
            y1 = y+scale - 1 + row*scale
            
            draw.rectangle([x0, y0, x1, y1], outline=(r_, g_, b_), width=3)

    img.save('attn_cards/'+img_name+'.png')


def view_attn():
    with torch.no_grad():
        global batch_size
        view_batch = [[tokens_to_ids[tok] for tok in sent] for sent in data]
        batch_size = len(view_batch)
        view_input_ids = torch.tensor(view_batch)
        
        txt_len = 3
        img_word_len = 8
        img_w = 64
        img_h = 64

        model = torch.load('ImgBert')
        _, attn = model(view_input_ids, None)

        for i in range(batch_size):
            img_name = ' '.join(data[i][:txt_len])
            pixels = ''.join(data[i][txt_len:])
            draw_attn(attn[i], img_name, pixels, img_w, img_h, img_word_len, txt_len)
        

if __name__ == '__main__':
    train()

