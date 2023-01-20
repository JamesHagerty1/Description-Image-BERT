from matplotlib import pyplot as plt
import random


def card_sentence_image(sentence):
    sentence = sentence[3:]
    image_bits = [int(bit) for word in sentence for bit in word]
    matrix = [image_bits[i*64:i*64+64] for i in range(64)]
    plt.figure(figsize=(5,5))
    plt.imsave("vis.png", matrix)
