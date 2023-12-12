import pandas as pd
import base64

def decode(encoded_data):
    encoded_bdata = encoded_data.encode()
    bdata = base64.b64decode(encoded_bdata)
    data = bdata.decode()
    return data

data = pd.read_csv("data.csv", index_col=0)

for col in data.columns[1:]:
    data[col] = data[col].apply(decode)

data.to_csv("decoded_data.csv")