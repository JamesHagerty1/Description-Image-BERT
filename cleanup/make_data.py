"""
-Create/determine all of the training data and model parameters. 
-The size of the data's vocabulary is one of the model's parameters.
-Data is images turned into sentences (see README for visual illustration).
"""


import json
import os


DATA_DIR = "data"

# Model parameters independent of data
N_LAYERS = 1 
D_FF = D_MODEL = 64 
D_K = D_V = 16 


def image_to_sentence(filename):
    print(filename)

def make_image_sentences():
    global DATA_DIR
    sentences = []
    for filename in os.listdir(DATA_DIR):
        sentence = image_to_sentence(filename)
        sentences.append(sentence)
    return sentences


def main():
    sentences = make_image_sentences()
        

if __name__ == "__main__":
    main()