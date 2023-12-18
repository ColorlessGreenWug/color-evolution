import pandas as pd
import random
from PIL import ImageColor
import numpy as np
import cv2
import base64
import time

def encode(data):
    bdata = data.encode()
    encoded_bdata = base64.b64encode(bdata)
    encoded_data = encoded_bdata.decode()
    return encoded_data

def decode(encoded_data):
    encoded_bdata = encoded_data.encode()
    bdata = base64.b64decode(encoded_bdata)
    data = bdata.decode()
    return data

# parameters
TOTAL_SIZE = 40
SAMPLE_SIZE = 20
LEARN_TIME = 10 * 60
CHAIN_ID = 1
TEXT_HEIGHT = 128
IMAGE_HEIGHT = 384
WIDTH = 512

COLOR_HEXCODES = {
    0: "#f4cccc",
    1: "#fce5cd",
    2: "#fff2cc",
    3: "#d9ead3",
    4: "#d0e0e3",
    5: "#cfe2f3",
    6: "#d9d2e9",
    7: "#ead1dc",
    8: "#ea9999",
    9: "#f9cb9c",
    10: "#ffe599",
    11: "#b6d7a8",
    12: "#a2c4c9",
    13: "#9fc5e8",
    14: "#b4a7d6",
    15: "#d5a6bd",
    16: "#e06666",
    17: "#f6b26b",
    18: "#ffd966",
    19: "#93c47d",
    20: "#76a5af",
    21: "#6fa8dc",
    22: "#8e7cc3",
    23: "#c27ba0",
    24: "#cc0000",
    25: "#e69138",
    26: "#f1c232",
    27: "#6aa84f",
    28: "#45818e",
    29: "#3d85c6",
    30: "#674ea7",
    31: "#a64d79",
    32: "#990000",
    33: "#b45f06",
    34: "#bf9000",
    35: "#38761d",
    36: "#134f5c",
    37: "#0b5394",
    38: "#351c75",
    39: "#741b47"
}

EXPERIMENT_DESCRIPTION = "Hello"


print(EXPERIMENT_DESCRIPTION)
name = input("Please enter your name.\n")


COLOR_IDS = list(range(40))

def put_text(image, text):
    x0 = 40
    y0 = 50
    dy = 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        image = cv2.putText(image, line, (x0, y ), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 1)
    return image

def generate_text(text):
    image = np.full((TEXT_HEIGHT + IMAGE_HEIGHT, WIDTH, 3), 255, np.uint8)
    return put_text(image, text)

def generate_image(color_id, text):
    image = np.full((TEXT_HEIGHT + IMAGE_HEIGHT, WIDTH, 3), 255, np.uint8)
    r, g, b = ImageColor.getrgb(COLOR_HEXCODES[color_id])
    image[TEXT_HEIGHT:, :, 0] = b
    image[TEXT_HEIGHT:, :, 1] = g
    image[TEXT_HEIGHT:, :, 2] = r
    return put_text(image, text)

# read data
data = pd.read_csv("data.csv", index_col=0)

# generate sample
learn_space = []
images = []
while len(learn_space) < SAMPLE_SIZE:
    sample = random.choice(COLOR_IDS)
    if sample not in learn_space:
        learn_space.append(sample)
        text = f"color {len(learn_space)}/{SAMPLE_SIZE}: {decode(data.iloc[-1, sample])}"
        images.append(generate_image(sample, text))

# show samples to the participant
start_time = time.time()
end_time = start_time + LEARN_TIME
pointer = 0

while True:
    cur_time = time.time()
    if cur_time >= end_time:
        cv2.imshow("Learning Phase", generate_text("Time's up!"))
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
        break

    cv2.imshow("Learning Phase", images[pointer])
    key = cv2.waitKey(int((end_time - cur_time) * 1000))
    if key == 27:
        break
    elif key == 97:
        pointer = max(0, pointer - 1)
    elif key == 100:
        pointer = min(SAMPLE_SIZE - 1, pointer + 1)
    cv2.destroyAllWindows()
    
def valid(word):
    if not word:
        return False
    
    C = {"p", "b", "t", "d", "k", "g", "m", "n"}
    V = {"i", "u", "e", "o", "a"}
    for i, c in enumerate(word):
        if i % 2 == 0 and not c in C:
            return False
        if i % 2 == 1 and not c in V:
            return False
    return True

# generate results
results = []
for color_id in COLOR_IDS:
    prompt = "Your turn!\ncurrent word: "
    word = ""
    while True:
        cv2.imshow("Producing Phase", generate_image(color_id, prompt + word))
        key = cv2.waitKey(0)
        if key == 13:
            if len(word) != 4 and len(word) != 6:
                print(f"{word} invalid! try reenter")
                word = ""
            else:
                break
        elif key == 8 or key == 127:
            if word:
                word = word[:-1]
        else:
            word += chr(key)
            if not valid(word):
                print(f"{word} invalid! try reenter")
                word = word[:-1]
        cv2.destroyAllWindows()
                
    print("success!", word)
    results.append(encode(word))
    

data.loc[name] = results
data.to_csv("data.csv")