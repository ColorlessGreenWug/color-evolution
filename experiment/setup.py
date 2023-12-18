import pandas as pd
import numpy as np
import random 
import base64

C = ["p", "b", "t", "d", "k", "g", "m", "n"]
V = ["i", "u", "e", "o", "a"]

def encode(data):
    bdata = data.encode()
    encoded_bdata = base64.b64encode(bdata)
    encoded_data = encoded_bdata.decode()
    return encoded_data

used_words = set()
def random_word():
    if random.random() < 0.5:
        consonants = random.sample(C, 2)
        vowels = random.choice(V) + random.choice(V)
        word = ''.join([consonants[0], vowels[0], consonants[1], vowels[1]])
    else:
        consonants = random.sample(C, 3)
        vowels = random.choice(V) + random.choice(V) + random.choice(V)
        word = ''.join([consonants[0], vowels[0], consonants[1], vowels[1], consonants[2], vowels[2]])
    if word in used_words:
        return random_word()
    used_words.add(word)
    return word

def generate_initial_words(path, method):
    df = pd.DataFrame(None, columns = list(range(40)))
    if method == "random":
        df.loc['initial'] = [encode(random_word()) for _ in range(40)]
    df.to_csv(path + "data.csv")

generate_initial_words("", "random")