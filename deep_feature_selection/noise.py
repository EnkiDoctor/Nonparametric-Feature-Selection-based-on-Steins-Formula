import sys
sys.path.append("../../src")

import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from torch.nn.parameter import Parameter
from utils import data_load_l, measure, mse
from models import Net_linear, Net_nonlinear
from dfs import DFS_epoch, training_l

import numpy as np
import time
import numpy as np
import argparse
import gc
#n_samples = 2000
def create_sigma_matrix(size, rho):
    # 创建一个空矩阵
    sigma = np.zeros((size, size))
    # 填充矩阵
    for i in range(size):
        for j in range(size):
            sigma[i, j] = rho ** abs(i - j)
    return sigma


def generate_samples_rho(d, n_samples, rho = 0):
    sigma = create_sigma_matrix(d, rho)
    samples = np.random.multivariate_normal(mean=np.zeros(d), cov= sigma, size=n_samples)
    return samples, sigma

def generate_samples(d, n_samples, iid=True):
    if iid:
        sigma = np.eye(d)
    else:
        sigma = create_sigma_matrix(d, 0.5)
    samples = np.random.multivariate_normal(mean=np.zeros(d), cov=sigma, size=n_samples)
    return samples, sigma

def relu(x):
    return np.maximum(0, x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def quadratic_k(X, A1, k ):
    output1 = (A1.T @ X.T).T
    output1 = np.multiply(output1, output1)
    linear_par = np.random.rand(k, 1)
    linear_output = (output1 @ linear_par) 
    return linear_output

def quadratic_k_noise(X, A1, k, c ):
    output1 = (A1.T @ X.T).T
    output1 = np.multiply(output1, output1)
    linear_par = np.random.rand(k, 1)
    linear_output = (output1 @ linear_par) 
    return linear_output + c*np.random.randn(n_samples, 1)

def calculate_accuracy(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    # 计算交集的大小
    intersection = set1.intersection(set2)
    accuracy = len(intersection) / len(set1)
    return accuracy

def linear_func(X,A1,k):
    output1 = (A1.T @ X.T).T
    par = np.random.rand(k, 1)
    output = output1 @ par
    return output

def cos_func(X,A1,k): 
    output1 = (A1.T @ X.T).T
    part1 = 3*np.multiply(output1, output1)
    part2 =  4*np.cos(output1)
    par = np.random.rand(k, 1)
    output = (part1 + part2) @ par
    return output 
def cos_func_noise(X,A1,k,c):
    output1 = (A1.T @ X.T).T
    part1 = 3*np.multiply(output1, output1)
    part2 =  4*np.cos(output1)
    par = np.random.rand(k, 1)
    output = (part1 + part2) @ par
    return output + c*np.random.randn(n_samples, 1)

def cos_func2(X,A1,k):
    output1 = (A1.T @ X.T).T
    part1 = np.multiply(output1, output1)
    part1 = np.multiply(part1, part1)
    part2 = np.cos(output1)
    par = np.random.rand(k, 1)
    output = (part1 + part2) @ par
    return output   

def cos_func2_noise(X,A1,k,c):
    output1 = (A1.T @ X.T).T
    part1 = np.multiply(output1, output1)
    part1 = np.multiply(part1, part1)
    part2 = np.cos(output1)
    par = np.random.rand(k, 1)
    output = (part1 + part2) @ par
    return output + c*np.random.randn(n_samples, 1)

def cos_func3(X,A1,k):
    output1 = (A1.T @ X.T).T
    part1 = np.multiply(output1, output1)
    part1 = np.multiply(part1, part1)

    part2 = 10 * np.cos(output1)
    part3 = 2 * np.multiply(output1, output1)
    par = np.random.rand(k, 1)
    output = (part1 - part2 + part3) @ par 
    return output
def cos_func3_noise(X,A1,k,c):
    output1 = (A1.T @ X.T).T
    part1 = np.multiply(output1, output1)
    part1 = np.multiply(part1, part1)

    part2 = 10 * np.cos(output1)
    part3 = 2 * np.multiply(output1, output1)
    par = np.random.rand(k, 1)
    output = (part1 - part2 + part3) @ par 
    return output + c*np.random.randn(n_samples, 1)

def manual_foward(X):
    A1 = np.load('neural_par/AA1.npy')
    A2 = np.load('neural_par/AA2.npy')
    A3 = np.load('neural_par/AA3.npy')
    A4 = np.load('neural_par/AA4.npy')
    X = np.maximum(0, np.dot(X, A1.T))
    X = np.maximum(0, np.dot(X, A2.T))
    X = np.maximum(0, np.dot(X, A3.T))
    X = np.dot(X, A4.T)
    return X

def exp_func(X,A1,k):
    output1 = (A1.T @ X.T).T
    part1 = np.exp(output1)
    par = np.random.rand(k, 1)
    output = part1 @ par
    return output

def exp_cos(X,A1,k):
    output1 = (A1.T @ X.T).T
    part1 = np.exp(output1)
    part2 = np.cos(output1)
    par = np.random.rand(k, 1)
    output = (part1 + part2) @ par
    return output
def exp_cos_noise(X,A1,k,c):
    output1 = (A1.T @ X.T).T
    part1 = np.exp(output1)
    part2 = np.cos(output1)
    par = np.random.rand(k, 1)
    output = (part1 + part2) @ par
    return output + c*np.random.randn(n_samples, 1)

def exp_poly(X, A1, k):
    output1 = (A1.T @ X.T).T
    part1 = np.exp(output1)
    part2 = np.multiply(output1, output1)
    part3 = np.multiply(part2, part2)
    par = np.random.rand(k, 1)
    output = (part1 - part2 + part3) @ par
    return output
def exp_poly_noise(X, A1, k, c):
    output1 = (A1.T @ X.T).T
    part1 = np.exp(output1)
    part2 = np.multiply(output1, output1)
    part3 = np.multiply(part2, part2)
    par = np.random.rand(k, 1)
    output = (part1 - part2 + part3) @ par
    return output + c*np.random.randn(n_samples, 1)


import os
from sklearn.preprocessing import StandardScaler

def append_results_to_txt(folder_path, file_name, results):
    # 确保文件夹存在
    os.makedirs(folder_path, exist_ok=True)
    
    # 构建完整的文件路径
    file_path = os.path.join(folder_path, file_name)
    
    # 追加结果到文件
    with open(file_path, 'a') as file:
        file.write(results + '\n')
    
    print(f"Results appended to {file_path}")




def test_s(n = 15,d =100, s = 5,  n_samples = 2000,func_type = 'quadratic',  rho = 0,  noise = 0):

    d = d
    k = s
    s = s
    p = d 
    n_samples = n_samples

    acc = 0
    iter = 0
    fpr = 0 
    
    for j in range(n):
        non_zero_indices = np.random.choice(d, s, replace=False)
        A1 = np.zeros((d, k))
        A1[non_zero_indices] = np.random.rand(s, k) 
        X, sigma = generate_samples_rho(d, n_samples, rho)

        if func_type == "quadratic":
            y = quadratic_k_noise(X,A1,k, noise)
        elif func_type == "cos":
            y = cos_func_noise(X, A1, k, noise)
        elif func_type == "cos2":
            y = cos_func2_noise(X, A1, k, noise)
        elif func_type == "cos3":
            y = cos_func3_noise(X, A1, k, noise)
        elif func_type == "exp_cos":
            y = exp_cos_noise(X, A1, k, noise)  
        elif func_type == "exp_poly":
            y = exp_poly_noise(X, A1, k, noise)
        else:
            raise ValueError("Invalid function type")

        X = torch.tensor(X, dtype=torch.float32)
        Y = torch.tensor(y, dtype=torch.float32)
        
        c = 1
        epochs = 10 
        n_hidden1 = 100
        n_hidden2 = 50
        learning_rate = 0.001
        Ts = 1000
        step = 1


        #model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1)
        #best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1)
        model = Net_linear(n_feature=p, n_hidden1=n_hidden1, n_output=1)
        best_model = Net_linear(n_feature=p, n_hidden1=n_hidden1,  n_output=1)

        optimizer = torch.optim.SGD(list(model.parameters()), lr=learning_rate, weight_decay=0.0025*c)
        optimizer0 = torch.optim.SGD(model.hidden0.parameters(), lr=learning_rate, weight_decay=0.0005*c)

        lf = torch.nn.MSELoss()


        hist = []
        SUPP = []
        supp_x = list(range(p)) # initial support
        SUPP.append(supp_x)
        best_supp = set(supp_x)
        ### DFS algorithm

        for i in range(epochs):
            model, supp_x, _ = DFS_epoch(model, s, supp_x, X, Y, lf, optimizer0, optimizer, Ts, step)
            supp_x.sort()
            # Save current loss function value and support
            hist.append(lf(model(X), Y).data.numpy().tolist())
            SUPP.append(supp_x)
            # Prevent divergence of optimization over support, save the current best model
            if hist[-1] == min(hist):
                best_model.load_state_dict(model.state_dict())
                best_supp = supp_x
            # Early stop criteria
            if len(SUPP[-1]) == len(SUPP[-2]) and len(set(SUPP[-1]).difference(SUPP[-2])) == 0:
                break

            supp_true = set(non_zero_indices)
            
        temp_acc = calculate_accuracy(supp_true, best_supp)
        #print("Accuracy:", temp_acc)
        acc += temp_acc
        iter += 1
        fpr += (1 - temp_acc)*s/d 

        # delete cache and model
        # Delete model and optimizer
        del model
        del best_model
        del optimizer
        del optimizer0

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()


    print("Final TPR:", acc / iter)
    print("Final FPR:", fpr / iter)

    result_string = f"d = {d} s = {s} rho = {rho} n = {n} func = {func_type}  TPR: {acc / iter}, FPR: {fpr / iter}, noise = {noise}, n_samples = {n_samples}"
    append_results_to_txt('dfs_result', 'noise_results_200.txt', result_string)
    print(result_string)
    return acc / iter, fpr / iter


import json
result_dic_tpr = {'quadratic': [], 'cos': [], 'cos2': [], 'cos3': [], 'exp_poly':[]}
result_dic_fpr = {'quadratic': [], 'cos': [], 'cos2': [], 'cos3': [], 'exp_poly':[]}

print('start')

n_samples = 2000 
for Atype in ['exp_poly','quadratic','cos3', 'cos2']: 
    for noise in [0,1,2,3,4,5,6,7,8,9,10]:
        a,b = test_s(n = 30, d = 200, s = 5, n_samples = 2000,func_type = Atype,  rho = 0,  noise = noise)
        result_dic_tpr[Atype].append(a)
        result_dic_fpr[Atype].append(b)

with open('../DATA/dfs/noise_50s_tpr_net_200_5.json', 'w') as f:
    json.dump(result_dic_tpr, f)
with open('../DATA/dfs/noise_50s_fpr_net_200_5.json', 'w') as f:
    json.dump(result_dic_fpr, f)
