# coding:utf-8
import os
import numpy as np

start_signal = "Starting trainings for resnet"
os.system("echo " + start_signal)

os.system("echo Adam")
os.system("python main.py --model=resnet --optim=adam --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot.py -p Adam")

os.system("echo TAdam")
os.system("python main.py --model=resnet --optim=tadam --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot.py -p Adam TAdam")

os.system("echo AMSGrad")
os.system("python main.py --model=resnet --optim=amsgrad --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot.py -p Adam TAdam AMSGrad")

os.system("echo TAMSGrad")
os.system("python main.py --model=resnet --optim=tamsgrad --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot.py -p Adam TAdam AMSGrad TAMSGrad")

os.system("echo RoAdam")
os.system(" python main.py --model=resnet --optim=roadam --lr=0.001 --beta1=0.99 --beta2=0.999 --beta3=0.999")
os.system("python plot.py -p Adam TAdam RoAdam AMSGrad TAMSGrad")
