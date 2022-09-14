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

def card_words():
    word_counts = defaultdict(int)
    word_len = 8 # invariant: img pixel cols % word_len == 0

    for filename in os.listdir( 'small_cards' ):
        if not filename.endswith( '.png' ):
            continue
        img = Image.open( f'small_cards/{filename}' )
        pixels = img.load()
        rows, cols = img.size

        for i in range(rows):
            for j in range(cols // word_len):
                offset = j * word_len
                word = [pixels[i, offset+k] for k in range(word_len)]
                word = ['0' if bright < 185 else '1' for bright in word]
                word = ''.join(word)
                word_counts[word] += 1
        
    pprint(word_counts)
    print(len(word_counts))

def card_sentences():
    word_len = 8
    f = open('cards.txt', 'w')
    for filename in os.listdir( 'small_cards' ):
        if not filename.endswith( '.png' ):
            continue
        card_name = filename.replace('.png','')
        card_name = ' '.join( card_name.split('_') ) + ' '
        # print(card_name)

        f.write(card_name)

        img = Image.open( f'small_cards/{filename}' )
        pixels = img.load()
        rows, cols = img.size

        for i in range(rows):
            for j in range(cols // word_len):
                offset = j * word_len
                word = [pixels[i, offset+k] for k in range(word_len)]
                word = ['0' if bright < 185 else '1' for bright in word]
                word = ''.join(word)
                # print(word)

                f.write(word)
                f.write(' ')

        f.write('\n')

def card_training_data():
    tokens_to_ids = {'[MASK]': 0}
    ids_to_tokens = {}
    batch = []

    f = open('cards.txt', 'r')
    data = f.read()
    f.close()
    data = data.strip()
    data = data.split('\n')
    data = [sent.split(' ') for sent in data]

    for sent in data:
        for word in sent:
            if not word in tokens_to_ids:
                tokens_to_ids[word] = len(tokens_to_ids)

    for k in tokens_to_ids:
        v = tokens_to_ids[k]
        ids_to_tokens[v] = k

    for sent in data:
        mask_me_1 = sent[0]
        mask_me_2 = sent[2]
        mask_me_1_id = tokens_to_ids[mask_me_1]
        mask_me_2_id = tokens_to_ids[mask_me_2]
        
        sent[0] = '[MASK]'
        token_ids_1 = [tokens_to_ids[token] for token in sent]
        batch.append([token_ids_1, [mask_me_1_id], [0]])
        sent[0] = mask_me_1

        sent[2] = '[MASK]'
        token_ids_2 = [tokens_to_ids[token] for token in sent]
        batch.append([token_ids_2, [mask_me_2_id], [2]])
        sent[2] = mask_me_2

    db = {}
    db['data'] = data
    db['tokens_to_ids'] = tokens_to_ids
    db['ids_to_tokens'] = ids_to_tokens
    db['batch'] = batch
    dbfile = open('img_data_pkl', 'ab')
    pickle.dump(db, dbfile)
    dbfile.close()

def reconstruct_img(img_list):
    card_name = ' '.join(img_list[:3])
    print(card_name)

    card_words = img_list[3:]
    card_pixels = []
    for word in card_words:
        for pixel in word:
            card_pixels.append(pixel)
    # print(card_pixels)
    # print(len(card_pixels))

    canvas = Image.open( f'test_cards/canvas.png' )
    w, h = canvas.size

    i = 0
    for x in range(w):
        for y in range(h):
            color = card_pixels[i]
            i += 1
            color = 0 if color == '0' else 255
            canvas.putpixel((x,y),(color))
    
    canvas.save(f'test_cards/{card_name}.png')
            
def preview_train():
    dbfile = open('img_data_pkl', 'rb')
    db = pickle.load(dbfile)

    data = db['data']
    # tokens_to_ids = db['tokens_to_ids']
    # ids_to_tokens = db['ids_to_tokens']
    # batch = db['batch']

    reconstruct_img(data[12])

    dbfile.close()

if __name__ == '__main__':
    # make_small_cards()
    # card_words()
    # card_sentences()
    # card_training_data()
    preview_train()