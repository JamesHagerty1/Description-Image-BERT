"""
-Helpers to create any image dataset for this model
-60x60 images where the only pixel colors are white, mid-gray, black
"""


import numpy as np
import json
from matplotlib import pyplot as plt
from PIL import Image as im


################################################################################


IMG_DIM = 60
IMG_WORD_DIM = 3

# (Grayscale) 0-85 -> black, 85-170 -> mid-gray, 170-255-> white
BLACK_MAX = 85
WHITE_MIN = 170
BLACK = 0
GRAY = 128
WHITE = 255


################################################################################


def crop():
    pass


def standardize_image(input_path, output_path):
    global BLACK_MAX, WHITE_MIN, BLACK, GRAY, WHITE
    def brightness_bucket(brightness):
        if (brightness <= BLACK_MAX): return BLACK
        elif (BLACK_MAX < brightness < WHITE_MIN): return GRAY
        else: return WHITE
    image = im.open(input_path)
    image = image.resize((IMG_DIM, IMG_DIM))
    image = image.convert("L") # grayscale
    image = image.point(lambda brightness: brightness_bucket(brightness))
    image.save(output_path)


################################################################################


def trinary_image_tokens(input_path):
    global BLACK, GRAY, WHITE
    image = im.open(input_path)
    pixels = np.array(image)
    for brightness, trinary_pixel in [(BLACK, 0), (GRAY, 1), (WHITE, 2)]:
        pixels = np.where(pixels == brightness, trinary_pixel, pixels)
    sentence = []
    for r in range(0, IMG_DIM, IMG_WORD_DIM):
        for c in range(0, IMG_DIM, IMG_WORD_DIM):
            token = ''.join(
                pixels[r:r+IMG_WORD_DIM,c:c+IMG_WORD_DIM].flatten().astype(str))
            sentence.append(token)
    return sentence


################################################################################


def vocabulary_json(add_tokens):
    token_to_id = {"[PAD]" : 0, "[MASK]" : 1, "[DESC]" : 2, "[IMG]" : 3}
    for token in add_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {token_to_id[k] : k for k in token_to_id}
    json_data = {"token_to_id" : token_to_id, "id_to_token" : id_to_token}


################################################################################


def tokens_image(tokens):
    print(tokens)



################################################################################


def main():
    standardize_image("./images/dog.png", "./images/trinary_dog.png")
    tokens = trinary_image_tokens("./images/trinary_dog.png")
    tokens_image(tokens)

if __name__ == "__main__":
    main()
