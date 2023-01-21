"""
-Helpers to create any image dataset for this model
-60x60 images where the only pixel colors are white, mid-gray, black
"""


import numpy as np
import os
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

DESC_MAX_LEN = 16


################################################################################


def crop(input_path, output_path):
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
    assert (image.size == (IMG_DIM, IMG_DIM)), "Incorrect image dims"
    pixels = np.array(image)
    assert (set(pixels.flatten()) == {BLACK, GRAY, WHITE}), "Non-trinary image"
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
    with open("vocabulary.json", "w") as json_file:
        json.dump(json_data, json_file, indent=2)


################################################################################


def valid_image_tokens(image_tokens):
    assert (len(image_tokens) == (IMG_DIM ** 2 // IMG_WORD_DIM ** 2)), \
        "Too few image tokens"
    for token in image_tokens:
        assert (len(token) == (IMG_WORD_DIM ** 2)), "Wrong image token length"
        s = set(list(token)).union({"0", "1", "2"})
        assert (len(s) == 3), "Not a trinary image token"
    return True 


def input_tokens(description_tokens, image_tokens):
    assert (len(description_tokens) <= DESC_MAX_LEN), "Description too long"
    assert (valid_image_tokens(image_tokens)), "Invalid image tokens"
    tokens = ["[DESC]"]
    description_tokens = description_tokens[:]
    while (len(description_tokens) < DESC_MAX_LEN):
        description_tokens.append("[PAD]")
    tokens.extend(description_tokens)
    tokens.append("[IMG]")
    tokens.extend(image_tokens)
    return tokens


def json_dataset_append(dataset_path, description_tokens, image_tokens, 
    masked_indices):
    """Used to iteratively create datasets tailored to this model"""
    assert (len(description_tokens) <= DESC_MAX_LEN), "Description too long"
    assert (len(masked_indices) <= len(description_tokens)), "Too many masks"
    assert (valid_image_tokens(image_tokens)), "Invalid image tokens"
    assert (dataset_path.endswith(".json")), "Non-json dataset file"
    if (not os.path.isfile(dataset_path)):
        with open(dataset_path, "w") as json_file:
            json.dump([], json_file, indent=2)

################################################################################


def tokens_matrix(image_tokens):
    matrix = np.zeros((IMG_DIM, IMG_DIM))
    for i, token in enumerate(image_tokens):
        token_matrix = np.array(list(token)).astype(int).reshape((3, 3))
        r = IMG_WORD_DIM * (i // (IMG_DIM // IMG_WORD_DIM)) # top
        c = (i * IMG_WORD_DIM) % IMG_DIM # left
        matrix[r:r+IMG_WORD_DIM,c:c+IMG_WORD_DIM] = token_matrix
    return matrix


def tokens_image(image_tokens, output_path):
    matrix = tokens_matrix(image_tokens)
    plt.figure(figsize=(5,5))
    plt.imsave(output_path, matrix)


################################################################################


def main():
    standardize_image("./images/dog.png", "./images/trinary_dog.png")
    tokens = trinary_image_tokens("./images/trinary_dog.png")
    tokens_image(tokens, "./images/re_trinary_dog.png")

if __name__ == "__main__":
    main()
