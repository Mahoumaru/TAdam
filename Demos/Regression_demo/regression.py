# coding:utf-8

import os
import sys
import errno
import itertools

import matplotlib as mpl
mpl.use("Agg")
import torch.multiprocessing as mp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
import torchoptim.optimizers.TAdam as TAdam #import TAdam
import torchoptim.optimizers.RoAdam as RoAdam #import RoAdam
import torch.utils.data
from torch import nn
from torch.nn import functional as F
from torch.distributions.studentT import StudentT
from torch.distributions.bernoulli import Bernoulli

######################################################
# hyperparameters
# number of multiprocess
# N_PROCESS = mp.cpu_count() - 1
N_PROCESS = 4

# whether use cuda or not: in such simple regression tasks, cpu is basically faster than gpu
USE_CUDA = False

# if replot previous results, please change to False
IS_LEARN = True

# learning conditions
N_TRIAL = 50
N_SAMPLE = 1000
N_BATCH = 10

# network conditions
# please define default dof by adjusting default argv in __init__ of RobustAdam
N_NEURON = 50
N_LAYER = 5

# noise conditions
NOISE_DF = 1.0
NOISE_SCALE = 1e-2

# comparison
# to plot accurate curves of probability vs. loss, please change steps in PROBS
PROBS = list(np.linspace(0, 100, num=10+1))
METHODS = ["Adam", "TAdam", "RoAdam"]#, "RMSprop"]

######################################################
class RegressionModel(nn.Module):
    def __init__(self, i_size, o_size, method, n_neuron, n_layer, ldir=None):
        # p = number of features
        super(RegressionModel, self).__init__()
        # network
        layers = []
        layers.append(nn.Linear(i_size, n_neuron))
        for i in range(n_layer-1):
            layers.append(nn.Linear(n_neuron, n_neuron))
        layers.append(nn.Linear(n_neuron, o_size))
        self.network = nn.ModuleList(layers)
        # optimizer
        if "Adam" == method:
            self.optimizer = optim.Adam(self.parameters())
        elif "TAdam" == method:
            self.optimizer = TAdam(self.parameters(), k_dof=1.)
        elif "RoAdam" == method:
            self.optimizer = RoAdam(self.parameters())
        elif "RMSprop" == method:
            self.optimizer = optim.RMSprop(self.parameters())
        # loss function
        self.fn_loss = nn.MSELoss()
        # load parameters
        try:
            self.load_state_dict(torch.load(ldir + "model.pt"))
            self.optimizer.load_state_dict(torch.load(ldir + "optimizer.pt"))
            print("load parameters are in "+ldir)
        except:
            print("parameters are initialized")
        # send to gpu
        self.use_cuda = USE_CUDA and torch.cuda.is_available()
        if self.use_cuda:
            self.cuda()
        else:
            self.cpu()

###################
    def forward(self, x):
        h = x
        for i, f in enumerate(self.network):
            h = F.relu(f(h)) if i < len(self.network) - 1 else f(h)
        return h

###################
    def criterion(self, yp, yt):
        return self.fn_loss(yp, yt)

###################
    def div_criterion(self, yp, yt):
        return self.div_fn_loss(yp, yt)

###################
    def release(self, sdir):
        # save models
        torch.save(self.state_dict(), sdir + "model.pt")
        torch.save(self.optimizer.state_dict(), sdir + "optimizer.pt")

######################################################
def make_Dirs(d):
    for i, p in enumerate(d.split("/")):
        p = "/".join(d.split("/")[:i]) + "/" + p
        if not os.path.isdir(p):
            try:
                os.mkdir(p)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise
                pass

######################################################
def target(x, prob=0, istest=False):
    # target function: sin(2pix)
    rtv = torch.sin(2.0 * np.pi * x)
    # add student-t noise
    noise = torch.zeros_like(rtv) if istest else StudentT(df=NOISE_DF, scale=NOISE_SCALE).sample(x.size()) * Bernoulli(prob/100.0).sample(x.size())
    if rtv.is_cuda:
        noise = noise.cuda()
    return rtv + noise

######################################################
def process(prob, method, n_trial, n_sample, n_batch, n_neuron, n_layer, is_learn=True):
    sdir = "./result/" + str(prob) + "/" + method + "/" + str(n_trial) + "/"
    make_Dirs(sdir)
    if is_learn:
        model = RegressionModel(1, 1, method, n_neuron, n_layer)
        print("Start learning: {}".format(sdir))
        train(model, sdir, prob, n_sample, n_batch, method)
        test(model, sdir)
        model.release(sdir)
    return sdir

######################################################
def train(model, sdir, prob, n_sample, n_batch, method):
    n_trial = int(sdir.split("/")[-2])
    torch.manual_seed(n_trial)
    np.random.seed(n_trial)

    model.train()
    list_loss = []
    list_sample = range(1, n_batch * n_sample + 1, n_batch)
    for n_ in list_sample:
        model.optimizer.zero_grad()
        x_ = torch.zeros(n_batch, 1).uniform_()
        if model.use_cuda:
            x_ = x_.cuda()
        yp = model(x_)
        yt = target(x_, prob=prob)
        loss = model.criterion(yp, yt)
        loss.backward()
        if "RoAdam" in method:
            model.optimizer.step(loss.detach())
        else:
            model.optimizer.step()
        list_loss.append(loss.data.item())
    else:
        np.savetxt(sdir + "train.csv", np.array([list_sample, list_loss]).T, delimiter=",")
        plt.clf()
        plt.plot(list_sample, list_loss, label="Loss")
        lgd = plt.legend()
        plt.xlabel("Number of samples")
        plt.ylabel("Loss")
        plt.savefig(sdir + "train.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

######################################################
def test(model, sdir, steps=1000):
    model.eval()
    x_ = torch.linspace(0.0, 1.0, steps=steps).view(steps, 1)
    if model.use_cuda:
        x_ = x_.cuda()
    yp = model(x_)
    yt = target(x_, istest=True)
    loss = model.criterion(yp, yt)
    x_ = x_.data.cpu().numpy().flatten()
    yp = yp.data.cpu().numpy().flatten()
    yt = yt.data.cpu().numpy().flatten()
    np.savetxt(sdir + "test.csv", np.array([x_, yp, yt]).T, delimiter=",")
    plt.clf()
    plt.plot(x_, yp, label="Prediction")
    plt.plot(x_, yt, label="Ground truth")
    lgd = plt.legend()
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.savefig(sdir + "test.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    print("Result of {}: {}".format(sdir, loss.data.item()))
    res = np.array([loss.data.item()]).T
    np.savetxt(sdir + "loss.csv", res, delimiter=",")

######################################################
def eval(sdirs):
    # configurations of plot
    sns.set(context = "paper", style = "white", palette = "Set2", font = "Arial", font_scale = 2, rc = {"lines.linewidth": 1.0, "pdf.fonttype": 42})
    sns.set_palette("Set2", 8, 1)
    colors = sns.color_palette(n_colors=10)
    markers = ["o", "s", "d", "*", "+", "x", "v", "^", "<", ">"]
    fig = plt.figure(figsize=(8, 6))

    # prepare dataframes
    columns_train = ["Probability", "Method", "n_repeat", "Number of samples", "Loss"]
    columns_test = ["Probability", "Method", "n_repeat", "Input", "Prediction", "Ground truth"]
    columns_loss = ["Probability", "Method", "n_repeat", "Loss"]
    common_dir = os.path.commonprefix(sdirs)
    probs = sorted(set([sdir[len(common_dir):].split("/")[0] for sdir in sdirs]))
    methods = sorted(set([sdir[len(common_dir):].split("/")[1] for sdir in sdirs]))
    df_train = pd.DataFrame(columns=columns_train)
    df_test = pd.DataFrame(columns=columns_test)
    df_loss = pd.DataFrame(columns=columns_loss)

    # load files
    print("Load files")
    for dir in sdirs:
        # load train.csv
        filename = dir + "train.csv"
        if os.path.isfile(filename):
            df = pd.read_csv(filename, header=None)
            df.columns = columns_train[-2:]
            info = dir[len(common_dir):].split("/")[:-1]
            df = pd.concat([df, pd.DataFrame([info]*len(df.index), columns=columns_train[0:-2])], axis=1)
            df_train = df_train.append(df, ignore_index=True, sort=True)
        # load test.csv
        filename = dir + "test.csv"
        if os.path.isfile(filename):
            df = pd.read_csv(filename, header=None)
            df.columns = columns_test[-3:]
            info = dir[len(common_dir):].split("/")[:-1]
            df = pd.concat([df, pd.DataFrame([info]*len(df.index), columns=columns_test[0:-3])], axis=1)
            df_test = df_test.append(df, ignore_index=True, sort=True)
        # load loss.csv
        filename = dir + "loss.csv"
        if os.path.isfile(filename):
            df = pd.read_csv(filename, header=None)
            df.columns = columns_loss[-1:]
            info = dir[len(common_dir):].split("/")[:-1]
            df = pd.concat([df, pd.DataFrame([info]*len(df.index), columns=columns_loss[0:-1])], axis=1)
            df_loss = df_loss.append(df, ignore_index=True, sort=True)
    # plot results
    print("Plot results")
    # plot loss (loss w.r.t prob)
    print("Summary")
    df = df_loss.astype({columns_loss[0]: float})
    df.to_csv(common_dir + "loss.csv", index=False)
    plt.clf()
    sns.lineplot(x=columns_loss[0], y=columns_loss[-1], hue="Method", data=df)
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.5, 1.15), loc="upper center", frameon=True, ncol=4)
    plt.tight_layout()
    plt.xlim([df[columns_loss[0]].min(), df[columns_loss[0]].max()])
    plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(0,0))
    plt.savefig(common_dir + "loss.pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
    # plot respective results
    for prob in probs:
        # plot train (learning curves)
        print("Learning curve of {}".format(prob))
        df = df_train[df_train["Probability"] == prob]
        df.to_csv(common_dir + "train_" + prob + ".csv", index=False)
        plt.clf()
        sns.lineplot(x=columns_train[-2], y=columns_train[-1], hue="Method", data=df, n_boot=100)
        handles, labels = plt.gca().get_legend_handles_labels()
        lgd = plt.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.5, 1.15), loc="upper center", frameon=True, ncol=4)
        plt.xlim([df[columns_train[-2]].min(), df[columns_train[-2]].max()])
        plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        plt.savefig(common_dir + "train_" + prob + ".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')
        # plot test (approximated curves)
        print("Approximated curve of {}".format(prob))
        df = df_test[df_test["Probability"] == prob]
        df.to_csv(common_dir + "test_" + prob + ".csv", index=False)
        plt.clf()
        sns.lineplot(x=columns_test[-3], y=columns_test[-2], hue="Method", data=df, n_boot=100)
        df = df[(df["n_repeat"] == "1") & (df["Method"] == methods[0])]
        plt.plot(df[columns_test[-3]], df[columns_test[-1]], label=columns_test[-1], ls="dashed", c="k")
        handles, labels = plt.gca().get_legend_handles_labels()
        lgd = plt.legend(handles=handles[1:], labels=labels[1:], bbox_to_anchor=(0.5, 1.15), loc="upper center", frameon=True, ncol=4)
        plt.xlim([df[columns_test[-3]].min(), df[columns_test[-3]].max()])
        plt.gca().ticklabel_format(style="sci", axis="y", scilimits=(0,0))
        plt.savefig(common_dir + "test_" + prob + ".pdf", bbox_extra_artists=(lgd,), bbox_inches='tight')

######################################################
def main():
    make_Dirs("./result/")

    pool = mp.Pool(processes=N_PROCESS)
    sdirs = pool.starmap(process, [(prob, method, n_trial, N_SAMPLE, N_BATCH, N_NEURON, N_LAYER, IS_LEARN) for prob, method, n_trial in itertools.product(PROBS, METHODS, range(1, N_TRIAL+1))])
    pool.close()

    print("Evaluate data")
    eval(sdirs)

######################################################
if __name__ == "__main__":
    main()
