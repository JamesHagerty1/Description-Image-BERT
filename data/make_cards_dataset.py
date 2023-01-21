"""
-Cards images and minimal vocabulary needed for cards images
"""


import os
from preprocessing import standardize_image, trinary_image_sentence, vocabulary_json


################################################################################


IMAGES_DIR = "./cards/"
DATASET_DIR = "./cards_dataset/"


################################################################################


def main():
    # # Images
    # for filename in os.listdir(IMAGES_DIR):
    #     input_path = f"{IMAGES_DIR}{filename}"
    #     output_path = f"{DATASET_DIR}{filename}"
    #     standardize_image(input_path, output_path)
    # Vocabulary
    words = set()
    for filename in os.listdir(DATASET_DIR):
        cardname = filename.replace(".png", "")
        words |= set(cardname.split("_"))
        sentence = trinary_image_sentence(f"{DATASET_DIR}/{filename}")
        break
    print(words)

if __name__ == "__main__":
    main()
