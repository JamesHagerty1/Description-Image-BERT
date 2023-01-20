from PIL import Image
import os
from collections import defaultdict
from pprint import pprint
import re
import pickle

img_d = 64

def make_small_cards():
    for filename in os.listdir( 'cards' ):
        if not filename.endswith( '.png' ):
            continue
        img = Image.open( f'cards/{filename}' )
        img_ = img.resize( (img_d, img_d) )
        img_ = img_.convert( 'L' )
        img_ = img_.point( lambda x: 0 if x<185 else 255, '1' )
        img_.save( f'small_cards/{filename}' )

def make_data():
    word_len = 8 # invariant: img pixel cols % word_len == 0

    data = []
    tokens_to_ids = {'[MASK]':0}
    ids_to_tokens = {}
    batch = []

    for filename in os.listdir( 'small_cards' ):
        if not filename.endswith( '.png' ):
            continue

        img = Image.open( f'small_cards/{filename}' )
        pixels = img.load()
        rows, cols = img.size

        s = []
        for r in range(rows):
            for c in range(cols):
                pix = '1' if (pixels[c, r] == 255) else '0'
                s.append(pix)
        s = ''.join(s)

        # # sanity check
        # for i in range(64):
        #     print( s[64*i:64*i+64] )
        # print('')

        card_sent = (filename.replace('.png','')).split('_')
        for i in range(len(s) // word_len):
            pix_word = s[i*word_len:i*word_len+word_len]
            card_sent.append(pix_word)

        data.append(card_sent)

    for sent in data:
        for word in sent:
            if not word in tokens_to_ids:
                tokens_to_ids[word] = len(tokens_to_ids)

    for k in tokens_to_ids:
        ids_to_tokens[ tokens_to_ids[k] ] = k

    for sent in data:
        input_ids = [tokens_to_ids[word] for word in sent]
        
        entry1 = input_ids[:]
        masktok1 = input_ids[0]
        entry1[0] = 0
        batch.append( [entry1, [masktok1], [0]] )

        entry2 = input_ids[:]
        masktok2 = input_ids[2]
        entry2[2] = 0
        batch.append( [entry2, [masktok2], [2]] )

    db = {
        'data': data,
        'tokens_to_ids': tokens_to_ids,
        'ids_to_tokens': ids_to_tokens,
        'batch': batch,
    }
    f = open('img_data_pkl', 'ab')
    pickle.dump(db, f)
    f.close()


if __name__ == '__main__':
    
    pass