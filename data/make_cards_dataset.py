"""
-Cards images and minimal vocabulary needed for cards images
"""


import os
from data import standardize_image, trinary_image_tokens, vocabulary_json


################################################################################


IMAGES_DIR = "./cards/"
DATASET_DIR = "./cards_dataset/"


################################################################################


def main():
    # Images
    for filename in os.listdir(IMAGES_DIR):
        input_path = f"{IMAGES_DIR}{filename}"
        output_path = f"{DATASET_DIR}{filename}"
        standardize_image(input_path, output_path)
    # Vocabulary
    tokens = set()
    for filename in os.listdir(DATASET_DIR):
        cardname = filename.replace(".png", "")
        tokens |= set(cardname.split("_"))
        sentence = trinary_image_tokens(f"{DATASET_DIR}/{filename}")
        tokens |= set(sentence)
    vocabulary_json(tokens)

if __name__ == "__main__":
    main()
