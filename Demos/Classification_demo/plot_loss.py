import numpy as np
import pandas as pd

import os
import errno

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns

####################### RETRIEVE DATA FROM MEMORY ########################
def retrieve_data(method, type):
    method = method.lower()
    file_name = method + "_" + type + "_loss.csv"
    df = pd.read_csv(file_name, header=None)
    num = [i for i in range(1, len(df)+1)]
    df.columns = COLUMNS
    df["Epochs"] = num
    return df

####################### PLOT DATA ########################
def plot_data(data, type):
    colors = sns.color_palette(n_colors=5)
    c = data.columns
    fig_name = "resnest_" + type + "_loss.pdf"
    clear_output(True)
    sns.lineplot(x=c[2], y=c[1], hue="Optimizers", palette=colors, data=data)
    #plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=5, borderaxespad=0, frameon=False, fontsize='xx-small')
    plt.grid(ls='--')
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.clf()

##########################################################
def evaluate_data(type):
    print("Evaluating the {} data".format(type))
    i = 0
    df = None
    for meth in METHODS:
        df_meth = retrieve_data(meth, type)
        if i == 0:
            df = df_meth
        elif i > 0:
            df = df.append(df_meth, ignore_index=True)
        i = i + 1
    plot_data(df, type)

##########################################################
COLUMNS = ["Optimizers", "Loss"]
METHODS = ["Adam", "TAdam", "RoAdam", "AMSGrad", "TAMSGrad"]
TYPES = ["train", "test"]

##########################################################
def main():
    for type in TYPES:
        type = type.lower()
        evaluate_data(type)

####################### LAUNCHER ########################
if __name__ == "__main__":
    main()
