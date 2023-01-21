import json
from matplotlib import pyplot as plt


################################################################################


def card_sentence_image(sentence):
    sentence = sentence[3:]
    image_bits = [int(bit) for word in sentence for bit in word]
    matrix = [image_bits[i*64:i*64+64] for i in range(64)]

    # TEMP eventually will draw attention as shade of orange over black or white
    for i in range(30):
        matrix[3][i] = 2

    plt.figure(figsize=(5,5))
    plt.imsave("vis.png", matrix)


################################################################################


def main():
    with open("train.json") as json_file:
        json_data = json_file.read()
    training_data = json.loads(json_data)
    sentence = eval(training_data[1]["image_sentence"])
    card_sentence_image(sentence)

if __name__ == "__main__":
    main()
