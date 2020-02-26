"""
Runs one instance of the environment and optimizes using the Soft Actor
Critic algorithm. Can use a GPU for the agent (applies to both sample and
train). No parallelism employed, everything happens in one python process; can
be easier to debug.
Requires OpenAI gym (and maybe mujoco).
"""

from rlpyt.samplers.serial.sampler import SerialSampler
from rlpyt.envs.gym import make as gym_make
from rlpyt.algos.pg.ppo import PPO
from modrlpytalgos.pg.ppo import PPO as PPOngc ## No gradient clipping ppo version
from rlpyt.agents.pg.mujoco import MujocoFfAgent
from rlpyt.runners.minibatch_rl import MinibatchRl#Eval
from rlpyt.utils.logging.context import logger_context

import torchoptim.optimizers.TAdam as TAdam
import torch
import argparse
import itertools

import os
import errno
import os.path as osp

from IPython.display import clear_output
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import numpy as np

from datetime import datetime

import pybullet_envs
from gym import wrappers

import math

######################################### UTILITIES ##########################################
#####################
def argument():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env_id', help='environment ID', default='HopperBulletEnv-v0')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    parser.add_argument('--cuda_idx', help='gpu to use ', type=int, default=0)
    parser.add_argument("-sn", "--ntrial_start", type=int, default=0, help='Select the starting simulation number.')
    parser.add_argument('-bs', "--batch_size", help='batch size ', type=int, default=64)
    parser.add_argument('-elc', "--c2", help='entropy loss coef. ', type=float, default=0.01)
    parser.add_argument('-clip', "--clip", help='clipping parameter. ', type=float, default=0.2)
    parser.add_argument('-T', "--horizon", help='Horizon T ', type=int, default=2048)
    parser.add_argument('-swap', "--swap", help='swapp lr ', type=int, default=0)
    parser.add_argument('--gc', action="store_true", help='run with gradient norm clipping (default: False)')
    args = parser.parse_args()
    return args

#####################
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

#####################
def plot_data(directory, data, name):
    sns.set(context = "paper", style = "white", palette = "Set2", font = "Arial", font_scale = 2, rc = {"lines.linewidth": 1.0, "pdf.fonttype": 42})
    sns.set_palette("Set2", 8, 1)
    colors = sns.color_palette(n_colors=10)
    c = data.columns
    fig_name = "ppo_" + name + ".pdf"
    clear_output(True)
    sns.lineplot(x=c[0], y=c[1], hue="Methods", data=data)
    plt.grid(ls='--')
    plt.legend(loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=4, borderaxespad=0, frameon=False, fontsize='xx-small')
    plt.tight_layout()
    plt.savefig(directory+fig_name)
    plt.clf()

#####################
def retrieve_data(directory, method, col='ReturnAverage'):
    data = pd.read_csv(directory)
    nl = [k+1 for k in range(len(data))]
    #ml = [method for _ in range(len(data))]
    if method == "adam":
        ml = [method + " {:.0e}".format(LEARNING_RATE) for _ in range(len(data))]
    elif method == "tadam":
        ml = [method + " {:.0e}".format(TLEARNING_RATE) for _ in range(len(data))]
    ra = list(data[col])
    d = {"thousand steps": nl, col: ra, "Methods": ml}
    return pd.DataFrame(data=d)

#####################
def evaluate_data(directories, file_name, col='ReturnAverage'):
    print("Evaluating the data")
    i = 0
    df = None
    for sdir in tqdm(directories):
        for meth in METHODS:
            if ("/" + meth) in sdir:
                df_meth = retrieve_data(sdir + "/" + file_name, meth, col=col)
                if i == 0:
                    df = df_meth
                elif i > 0:
                    df = df.append(df_meth, ignore_index=True)
                i = i + 1
                break
    location = MAIN_DIRECTORY + "/" + "rlpyt_plots" + "/"
    make_Dirs(location)
    plot_data(location, df, col)

####################### RECORDING VIDEO FUNCTION ########################
def record_video(dirs, video_path):
    print("Record interaction videos")
    SUPER_LIST = []
    for meth in METHODS:
        mlist = []
        for sdir in dirs:
            if "/"+meth in sdir:
                mlist.append(sdir)
        SUPER_LIST.append(mlist)
    assert len(SUPER_LIST) == len(METHODS), "We have more sub lists than methods! Check again."

    for lst, meth in zip(SUPER_LIST, METHODS):
        for dir in lst:
            idx = dir.find(meth) + len(meth) + 1
            n = dir[idx]
            print("########### {}, {} ###########".format(meth, n))
            print("dir {}; Method {}".format(dir, meth))
            record_test(dir, video_path + meth + '_videos/', n)

####################### VIDEO RECORDING TEST FUNCTION ########################
def record_test(directory, video_path, n):
    make_Dirs(video_path + n + '/')
    env = gym_make(ENV_ID)
    env = wrappers.Monitor(env, video_path + n + '/', video_callable=lambda episode_id: True, force=True)
    env.seed(int(n)*7)
    np.random.seed(int(n)*7)
    torch.manual_seed(int(n)*7)

    agent = MujocoFfAgent()
    agent.initialize(env.spaces)

    netword_state_dict = None
    try:
        network_state_dict = torch.load(directory + 'agent_model.pth')
    except (FileNotFoundError):
        print("No data found for the PPO agent (No existing model).")
        network_state_dict = None
        return

    if network_state_dict != None:
        agent.load_state_dict(network_state_dict)
    else:
        return

    agent.to_device(0)

    frame_idx   = 0
    print("Start Test Episode for {}".format(n))
    done = False
    ### Interaction
    step = 0
    state = env.reset()
    prev_action = env.action_space.sample()
    prev_reward = 0.
    while not done:# or step < MAX_STEPS:
        env.render()
        state = torch.FloatTensor(state)
        prev_action = torch.FloatTensor(prev_action)
        prev_reward = torch.FloatTensor([prev_reward])
        #agent.eval_mode(step) # determinitic distribution. The std is ignored.
        action = agent.step(state, prev_action, prev_reward).action
        action = action.detach().cpu().numpy()
        next_state, reward, done, _ = env.step(action)

        state = next_state
        prev_action = action
        prev_reward = reward
        frame_idx += 1
        step += 1

        if done:
            break
    env.close()

######################################### TRAIN ##########################################
def build_and_train(env_id="Pendulum-v0", run_ID=0, cuda_idx=None, method="adam", trial=0):
    sampler = SerialSampler(
        EnvCls=gym_make,
        env_kwargs=dict(id=env_id),
        batch_T=HORIZON,  # Time-step per sampler iteration, T.
        batch_B=1,  # One environment (i.e. sampler Batch dimension).
        max_decorrelation_steps=400 #(int): if taking random number of steps before start of training, to decorrelate batch states
    )
    if method == "adam":
        algo = ALGO(learning_rate=LEARNING_RATE, value_loss_coeff=C1,
                   entropy_loss_coeff=C2, gae_lambda=GAE_PARAM,
                   minibatches=NUM_MINIBATCHES, epochs=NUM_EPOCHS,
                   ratio_clip=CLIPPING, linear_lr_schedule=False)
    elif method == "tadam":
        algo = ALGO(learning_rate=TLEARNING_RATE, value_loss_coeff=C1,
                   entropy_loss_coeff=C2, OptimCls=TAdam, gae_lambda=GAE_PARAM,
                   minibatches=NUM_MINIBATCHES, epochs=NUM_EPOCHS,
                   ratio_clip=CLIPPING, linear_lr_schedule=False)
    elif method == "amsgrad":
        algo = ALGO(learning_rate=LEARNING_RATE, value_loss_coeff=C1,
                   entropy_loss_coeff=C2, optim_kwargs={'amsgrad': True},
                   gae_lambda=GAE_PARAM,  minibatches=NUM_MINIBATCHES,
                   epochs=NUM_EPOCHS, ratio_clip=CLIPPING,
                   linear_lr_schedule=False)
    elif method == "tamsgrad":
        algo = ALGO(learning_rate=TLEARNING_RATE, value_loss_coeff=C1,
                   entropy_loss_coeff=C2, OptimCls=TAdam, optim_kwargs={'amsgrad': True},
                   gae_lambda=GAE_PARAM, minibatches=NUM_MINIBATCHES,
                   epochs=NUM_EPOCHS, ratio_clip=CLIPPING,
                   linear_lr_schedule=False)
    agent = MujocoFfAgent()
    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=MAX_FRAMES,
        log_interval_steps=1e3,
        affinity=dict(cuda_idx=cuda_idx),
        seed=trial
    )
    config = dict(env_id=env_id)
    name = "ppo_" + env_id
    log_dir = method + "_" + str(trial)
    DATA_DIRECTORY = "/home/isc-lab/Documents/rlpyt/data/local/" + datetime.now().strftime("%Y%m%d")
    with logger_context(log_dir, run_ID, name, config):
        runner.train()
    log_dir = osp.join(log_dir, f"run_{run_ID}")
    exp_dir = osp.join(DATA_DIRECTORY, log_dir)
    ### Save Model
    torch.save(agent.state_dict(), exp_dir + 'agent_model.pth')
    return exp_dir

######################################### HYPERPARAMETERS ##########################################
ARGS = argument()

ENV_ID = ARGS.env_id

METHODS = ["adam", "tadam"]#, "amsgrad", "tamsgrad"]#
N_TRIALS = 2
NTRIAL_START = ARGS.ntrial_start
MAX_FRAMES  = int(1e6)

N_PROCESSES = 4

if ARGS.swap == 0:
    TLEARNING_RATE = 1e-3
    LEARNING_RATE = 3e-4
else:
    TLEARNING_RATE = 3e-4
    LEARNING_RATE = 1e-3

if ARGS.gc:
    ALGO = PPO
else:
    ALGO = PPOngc

CLIPPING = ARGS.clip #0.2
HORIZON = ARGS.horizon #default = 2048
NUM_EPOCHS = 10
NUM_MINIBATCHES = HORIZON // ARGS.batch_size
GAE_PARAM = 0.95
C1 = 1. # Irrelevant
C2 = ARGS.c2 # 0 = No entropy bonus, default = 0.01

MAIN_DIRECTORY = "./ppo_alpha_results/"

######################################### MAIN ##########################################
if __name__ == "__main__":
    make_Dirs(MAIN_DIRECTORY)
    mp = torch.multiprocessing.get_context('spawn')
    pool = mp.Pool(processes = N_PROCESSES)
    sdirs = pool.starmap(build_and_train, [(ENV_ID, ARGS.run_ID, ARGS.cuda_idx, method, n_simulation + NTRIAL_START) for method, n_simulation in itertools.product(METHODS, range(1, N_TRIALS+1))])
    print("Close Pools")
    pool.close()
    print("")
    print("Save Directories.")
    np.savetxt(MAIN_DIRECTORY + "/dirs.csv", np.array(sdirs).T, fmt="%s")

    """
    dirs = pd.read_csv(MAIN_DIRECTORY + "/" + "rlpyt_plots/" + "dirs.csv", header=None)
    sdirs = list(dirs[0])
    """

    record_video(sdirs, MAIN_DIRECTORY)

    print("Evaluate Data:")
    file_name = "progress.csv"
    evaluate_data(sdirs, file_name)
    ### Plot the loss functions averages
    evaluate_data(sdirs, file_name, col='lossAverage')
    ### Plot the gradients norms averages
    evaluate_data(sdirs, file_name, col='gradNormAverage')
    ### Plot the perplexity averages
    evaluate_data(sdirs, file_name, col='perplexityAverage')
    ### Plot the entropy averages
    evaluate_data(sdirs, file_name, col='entropyAverage')
