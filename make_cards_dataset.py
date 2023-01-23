"""
-Cards images and minimal vocabulary needed for cards images
"""


import os
from processing import standardize_image, trinary_image_tokens, \
    vocabulary_json, json_dataset_append


################################################################################


IMAGES_DIR = "./data/cards/"
DATASET_DIR = "./data/cards_dataset/"
JSON_PATH = "./data/cards_dataset.json"


################################################################################


def main():
    # Images
    for filename in os.listdir(IMAGES_DIR):
        input_path = f"{IMAGES_DIR}{filename}"
        output_path = f"{DATASET_DIR}{filename}"
        standardize_image(input_path, output_path)

    # Vocabulary (currently limited to token-words of cards dataset)
    tokens = set()
    for filename in os.listdir(DATASET_DIR):
        cardname = filename.replace(".png", "")
        tokens |= set(cardname.split("_"))
        image_tokens = trinary_image_tokens(f"{DATASET_DIR}/{filename}")
        tokens |= set(image_tokens)
    vocabulary_json(tokens)
    
    # Data for model
    for filename in sorted(os.listdir(DATASET_DIR), \
        key=lambda x:(int(x.split("_")[0]), x.split("_")[-1])):
        description_tokens = filename.replace(".png", "").split("_")
        image_tokens = trinary_image_tokens(f"{DATASET_DIR}/{filename}")
        # "8 of hearts" -> "[MASK] of hearts", "8 of [MASK]"
        json_dataset_append(JSON_PATH, description_tokens, image_tokens, [0])
        json_dataset_append(JSON_PATH, description_tokens, image_tokens, [2])

if __name__ == "__main__":
    main()
