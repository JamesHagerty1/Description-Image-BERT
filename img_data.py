from PIL import Image
import os
from collections import defaultdict
from pprint import pprint

def make_small_cards():
    for filename in os.listdir( 'cards' ):
        if not filename.endswith( '.png' ):
            continue
        img = Image.open( f'cards/{filename}' )
        img_ = img.resize( (64, 64) )
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
        

if __name__ == '__main__':
    # make_small_cards()
    card_words()