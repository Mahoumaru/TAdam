# coding:utf-8
import os
import numpy as np

start_signal = "Starting trainings for resnet"
os.system("echo " + start_signal)

os.system("echo Adam")
os.system("python main.py --model=resnet --optim=adam --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot_originalversion.py -p Adam")

os.system("echo TAdam")
os.system("python main.py --model=resnet --optim=tadam --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot_originalversion.py -p Adam TAdam")

os.system("echo AMSGrad")
os.system("python main.py --model=resnet --optim=amsgrad --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot_originalversion.py -p Adam TAdam AMSGrad")

os.system("echo TAMSGrad")
os.system("python main.py --model=resnet --optim=tamsgrad --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot_originalversion.py -p Adam TAdam AMSGrad TAMSGrad")

os.system("echo RoAdam")
os.system(" python main.py --model=resnet --optim=roadam --lr=0.001 --beta1=0.99 --beta2=0.999")
os.system("python plot_originalversion.py -p Adam TAdam AdaBound RoAdam AMSGrad TAMSGrad")

"""os.system("echo AdaBound")
os.system("python main.py --model=resnet --optim=adabound --lr=0.001 --final_lr=0.1")
os.system("python plot_originalversion.py -p Adam TAdam AMSGrad TAMSGrad AdaBound")

os.system("echo TAdaBound")
os.system("python3 main.py --model=resnet --optim=tadabound --lr=0.001 --final_lr=0.1")
os.system("python3 plot.py -p Adam TAdam AdaBound TAdaBound")

os.system("echo AMSBound")
os.system("python3 main.py --model=resnet --optim=amsbound --lr=0.001 --final_lr=0.1")
os.system("python3 plot.py -p Adam TAdam AdaBound TAdaBound AMSGrad TAMSGrad AMSBound")

os.system("echo TAMSBound")
os.system("python3 main.py --model=resnet --optim=tamsbound --lr=0.001 --final_lr=0.1")
os.system("python3 plot.py -p Adam TAdam AdaBound TAdaBound AMSGrad TAMSGrad AMSBound TAMSBound")"""

"""os.system("echo SGD")
os.system("python3 main.py")
os.system("python3 plot.py -p Adam TAdam AdaBound TAdaBound AMSGrad TAMSGrad AMSBound TAMSBound SGD")"""
