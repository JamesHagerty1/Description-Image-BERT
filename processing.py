"""
-Helpers to create any image dataset for this model
-Helpers to tokenize data
-60x60 images where the only pixel colors are white, mid-gray, black
-Helpers to visualize how attention is applied
"""


import numpy as np
import os
import json
import torch
from matplotlib import pyplot as plt
from PIL import Image as im
from data_loading import init_dataloader


################################################################################


IMG_DIM = 60
IMG_WORD_DIM = 3

# (Grayscale) 0-85 -> black, 85-170 -> mid-gray, 170-255-> white
BLACK_MAX = 85
WHITE_MIN = 170
BLACK = 0
GRAY = 128
WHITE = 255

SPECIAL_TOKEN_TO_ID = {"[DESC]" : 0, "[MASK]" : 1, "[PAD]" : 2, "[IMG]" : 3}
VOCAB_JSON_PATH = "./data/vocabulary.json"

DESC_MAX_LEN = 16
DESC_MAX_MASKS = 16 # <= DESC_MAX_LEN
# "[DESC]" and "[IMG]" + description tokens + image tokens
SEQ_LEN = 2 + DESC_MAX_LEN + (IMG_DIM ** 2 // IMG_WORD_DIM ** 2)

VISUALS_DIR = "./visuals/"


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
    """TBD -- revise for vocabulary appending, like json_dataset_append()"""
    global SPECIAL_TOKEN_TO_ID, VOCAB_JSON_PATH
    token_to_id = SPECIAL_TOKEN_TO_ID
    for token in add_tokens:
        if token not in token_to_id:
            token_to_id[token] = len(token_to_id)
    id_to_token = {token_to_id[k] : k for k in token_to_id}
    json_data = {"token_to_id" : token_to_id, "id_to_token" : id_to_token}
    with open(VOCAB_JSON_PATH, "w") as json_file:
        json.dump(json_data, json_file, indent=2)


################################################################################


def valid_masked_indices(masked_indices, description_tokens):
    global DESC_MAX_LEN
    assert (len(masked_indices) <= len(description_tokens)), "Too many masks"
    assert (len(set(masked_indices)) == len(masked_indices)), "Repeating masks"
    s = set(range(DESC_MAX_LEN)).union(masked_indices)
    assert (len(s) == DESC_MAX_LEN), "Out of bounds mask indices"
    return True


def valid_image_tokens(image_tokens):
    global IMG_DIM, IMG_WORD_DIM
    assert (len(image_tokens) == (IMG_DIM ** 2 // IMG_WORD_DIM ** 2)), \
        "Too few image tokens"
    for token in image_tokens:
        assert (len(token) == (IMG_WORD_DIM ** 2)), "Wrong image token length"
        s = set(list(token)).union({"0", "1", "2"})
        assert (len(s) == 3), "Not a trinary image token"
    return True 


def input_tokens(description_tokens, image_tokens):
    global DESC_MAX_LEN
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
    global DESC_MAX_LEN, DESC_MAX_MASKS, VOCAB_JSON_PATH
    assert (dataset_path.endswith(".json")), "Non-json dataset file"
    assert (len(description_tokens) <= DESC_MAX_LEN), "Description too long"
    assert (valid_image_tokens(image_tokens)), "Invalid image tokens"
    assert(valid_masked_indices(masked_indices, description_tokens)), \
        "Invalid masked indices"
    if (not os.path.isfile(dataset_path)):
        with open(dataset_path, "w") as json_file:
            json.dump([], json_file, indent=2)
    # Append to dataset
    with open(dataset_path) as json_file:
        json_data = json.load(json_file)
    # Format for model input (but as tokens rather than their integer ids)
    tokens = input_tokens(description_tokens, image_tokens)
    # Show json viewer where masks fall on tokens for readability
    masked_tokens = tokens[:]
    for i in masked_indices:
        masked_tokens[i+1] = "[MASK]" # i=0 "[DESC]"
    # Model inputs
    with open(VOCAB_JSON_PATH) as json_file:
        vocab = json.load(json_file)
    token_to_id = vocab["token_to_id"]
    tokens_ids = [token_to_id[token] for token in tokens]
    masked_tokens_ids = [token_to_id[token] for token in masked_tokens]
    masked_ids = [tokens_ids[i+1] for i in masked_indices] # i=0 "[DESC]"
    while (len(masked_ids) < DESC_MAX_MASKS):
        masked_ids.append(0) # "[DESC]" id 0 (filler/ mask)
    masked_indices = masked_indices[:]
    masked_indices = [i + 1 for i in masked_indices] # i=0 "[DESC]"
    while (len(masked_indices) < DESC_MAX_MASKS):
        masked_indices.append(0) # "[DESC]" always at index 0 (filler mask)
    json_data.append({"tokens" : str(tokens), 
        "masked_tokens" : str(masked_tokens), "tokens_ids" : str(tokens_ids),
        "masked_tokens_ids" : str(masked_tokens_ids), 
        "masked_indices" : str(masked_indices), "masked_ids" : str(masked_ids)})
    with open(dataset_path, "w") as json_file:
        json.dump(json_data, json_file, indent=2)


######## Token sequence sanity checks ##########################################

def tokens_matrix(image_tokens):
    global IMG_DIM
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


# TEMP ref
def vis():
    m = np.empty([60, 60, 3], dtype=np.uint8)
    m[0][0] = np.array([128, 128, 128])
    m[0][1] = np.array([255, 0, 0])
    m[0][2] = np.array([0, 255, 0])
    m[0][3] = np.array([0, 0, 255])
    m[0][4] = np.array([255, 255, 255])

    fig, axs = plt.subplots(2, 2)
    plt.setp(axs, xticks=[], yticks=[])

    axs[0][0].set_title("<title>")
    axs[0][0].imshow(m)
    axs[0][1].imshow(m)
    axs[1][0].imshow(m)
    axs[1][1].imshow(m)

    plt.savefig("visuals/test.png")


################################################################################


def main():
    # standardize_image("./data/images/dog.png", "./data/images/trinary_dog.png")
    # tokens = trinary_image_tokens("./data/images/trinary_dog.png")
    # tokens_image(tokens, "./data/images/re_trinary_dog.png")

    # with open("./data/cards_dataset.json") as json_file:
    #     json_data = json.load(json_file)
    # tokens = eval(json_data[0]["tokens"])[2+DESC_MAX_LEN:]
    # tokens_image(tokens, "./data/images/test.png")

    model = torch.load("./models/ImgBert-loss:0.021")
    dataloader = init_dataloader("./data/cards_dataset.json", 1)
    for i, batch in enumerate(dataloader):
        x, y_i, y, desc = batch
        print(desc[0])
        with torch.no_grad():
            y_hat, attn = model(x, y_i)
        print(y_hat.shape)
        break


if __name__ == "__main__":
    main()
