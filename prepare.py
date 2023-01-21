"""
-Create/determine all of the training data and model parameters. 
-The size of the data's vocabulary is one of the model's parameters.
-Data is images turned into sentences (see README for visual illustration).
"""


import json
import os
from PIL import Image


################################################################################


DATA_DIR = "data/"
IMG_PIXELS = 4096
BIN_WORD_LEN = 8 
BATCH_SIZE = 18

# Model parameters independent of data
N_LAYERS = 1 
D_FF = D_MODEL = 64 
D_K = D_V = 16 


######### JSON #################################################################


# More concise syntax to access JSON fields
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def make_config_json(vocab_size):
    json_data = {
        "vocab_size" : vocab_size,
        "batch_size" : BATCH_SIZE,
        "n_layers" : N_LAYERS,
        "d_ff" : D_FF,
        "d_model" : D_MODEL,
        "d_k" : D_K,
        "d_v" : D_V,}
    with open("config.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)


def make_train_json(sentences, tokens_ids_d):
    def masked_entry(sentence, mask_i):
        masked_token_id = tokens_ids_d[sentence[mask_i]]
        sentence[mask_i] = "[MASK]"
        return {"masked_sentence" : str(sentence),
            "masked_tokens" : str([tokens_ids_d[token] for token in sentence]),
            "masked_tokens_ids" : str([masked_token_id]), 
            "masked_tokens_i" : str([mask_i])}
    json_data = []
    for sentence in sentences:
        entry = {}
        entry["image_sentence"] = str(sentence)
        masked_sentences = []
        masked_entry_1 = masked_entry(sentence[:], 0)
        masked_entry_2 = masked_entry(sentence[:], 2)
        masked_sentences.append(masked_entry_1)
        masked_sentences.append(masked_entry_2)
        entry["masked_sentences"] = masked_sentences
        json_data.append(entry)
    with open("train.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)


######### PARSING ##############################################################


def make_vocabulary(sentences):
    tokens_ids_d = {"[MASK]" : 0}
    for sentence in sentences:
        for word in sentence:
            if word not in tokens_ids_d:
                tokens_ids_d[word] = len(tokens_ids_d)
    ids_tokens_d = {tokens_ids_d[token] : token for token in tokens_ids_d}
    return tokens_ids_d, ids_tokens_d


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


################################################################################


def main():
    sentences = make_image_sentences()
    tokens_ids_d, ids_tokens_d = make_vocabulary(sentences)
    vocab_size = len(tokens_ids_d)
    make_train_json(sentences, tokens_ids_d)
    make_config_json(vocab_size)

if __name__ == "__main__":
    main()
