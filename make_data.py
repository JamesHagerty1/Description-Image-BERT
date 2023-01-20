"""
-Create/determine all of the training data and model parameters. 
-The size of the data's vocabulary is one of the model's parameters.
-Data is images turned into sentences (see README for visual illustration).
"""


import json
import os
from PIL import Image


DATA_DIR = "data/"
IMG_PIXELS = 4096
BIN_WORD_LEN = 8 

# Model parameters independent of data
N_LAYERS = 1 
D_FF = D_MODEL = 64 
D_K = D_V = 16 


def image_to_sentence(filename):
    sentence = []
    cardname = filename.replace(".png", "")
    sentence.extend( cardname.split("_") )
    image = Image.open(f"{DATA_DIR}/{filename}")
    pixels = image.load()
    rows, cols = image.size
    binary_pixels = ["1" if (pixels[c, r] == 255) else "0" for r in range(rows) 
        for c in range(cols)]
    assert(len(binary_pixels) == IMG_PIXELS), "Bad image dims"
    assert (len(binary_pixels) % BIN_WORD_LEN == 0), "Bad image dims"
    binary_image_words = [''.join(word) for word in 
        [binary_pixels[i:i+BIN_WORD_LEN] for i in 
        range(0, len(binary_pixels), BIN_WORD_LEN)]]
    sentence.extend(binary_image_words)
    return sentence


def make_image_sentences():
    global DATA_DIR
    sentences = []
    for filename in os.listdir(DATA_DIR):
        sentence = image_to_sentence(filename)
        sentences.append(sentence)
    return sentences


def main():
    sentences = make_image_sentences()
    
    from visual_data import card_sentence_image
    card_sentence_image(sentences[0])
        

if __name__ == "__main__":
    main()