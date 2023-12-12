import pandas as pd
from collections import Counter
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = "data/"

def frequency_data(list_of_strings, length=1):
    '''
    frequency of substrings of length len in list_of_strings
    '''
    freq = Counter()
    for string in list_of_strings:
        freq.update([string[i:i+length] for i in range(len(string)-1)])
    return freq

def calc_entropy(freq):
    '''
    entropy of a frequency distribution
    '''
    total = sum(freq.values())
    probs = [freq[key]/total for key in freq.keys()]
    return -sum([p*np.log2(p) for p in probs])

rows = []
for l in [1, 2, 4]:
    for file in glob(data_dir + "*.csv"):
        data = pd.read_csv(file, index_col=0)
        for i, row in data.iterrows():
            freq = frequency_data(row, length=l)
            entropy = calc_entropy(freq)
            r = [file[-10:-4], i, l, entropy]
            rows.append(r)
freq_df = pd.DataFrame(rows, columns=["file", "iter", "length", "entropy"])
print

sns.set_style("whitegrid")
# sns.set_palette("Set2")
# sns.set_context("paper")
# sns.set(font_scale=1.5)
for l in [1, 2, 4]:
    plt.figure()
    sns.lineplot(x="iter", y="entropy", hue="file", data=freq_df[freq_df["length"]==l], markers=True)
    plt.savefig("freq_analysis_len{}.png".format(l), dpi=300)


