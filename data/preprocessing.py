"""
-Helpers to create any image dataset for this model
-60x60 images where the only pixel colors are white, mid-gray, black
"""


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
    def trinary_brightness(brightness):
        if (brightness <= BLACK_MAX): return BLACK
        elif (BLACK_MAX < brightness < WHITE_MIN): return GRAY
        else: return WHITE
    image = im.open(input_path)
    image = image.resize((IMG_DIM, IMG_DIM))
    image = image.convert("L") # grayscale
    image = image.point(lambda brightness: trinary_brightness(brightness))
    image.save(output_path)


################################################################################


def trinary_image_sentence(input_path):
    global BLACK, GRAY, WHITE
    print(input_path)
    image = im.open(input_path)
    rows, cols = image.size
    grayscale_pixels = image.load()
    grayscale_to_trinary = {BLACK : "0", GRAY : "1", WHITE : "2"}
    trinary_pixels = [grayscale_to_trinary[grayscale_pixels[c, r]] 
        for r in range(rows) for c in range(cols)]
    print(len(trinary_pixels))
    # TBD, sentence itself


################################################################################


def vocabulary_json(words):
    token_to_id = {"[PAD]" : 0, "[MASK]" : 1, "[DESC]" : 2, "[IMG]" : 3}

    id_to_token = {token_to_id[k] : k for k in token_to_id}
    json_data = {"token_to_id" : token_to_id, "id_to_token" : id_to_token}


################################################################################
def main():
    standardize_image("./images/dog.png", "./images/test.png")

if __name__ == "__main__":
    main()
