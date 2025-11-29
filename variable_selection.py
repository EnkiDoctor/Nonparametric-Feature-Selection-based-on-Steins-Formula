import numpy as np
from scipy.linalg import eigh
from typing import Dict, List, Tuple
import os
import sys
from typing import Dict, List, Tuple
import warnings
import time
from scipy.stats import multivariate_t

def create_sigma_matrix(size, rho):
    sigma = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            sigma[i, j] = rho ** (abs(i - j))
    return sigma

def generate_samples(d, n_samples, iid=True):
    if iid:
        sigma = np.eye(d)
    else:
        sigma = create_sigma_matrix(d, 0.5)
    samples = np.random.multivariate_normal(mean=np.zeros(d), cov= sigma, size=n_samples)
    return samples, sigma

def generate_samples_rho(d, n_samples, rho = 0):
    sigma = create_sigma_matrix(d, rho)
    samples = np.random.multivariate_normal(mean=np.zeros(d), cov= sigma, size=n_samples)
    return samples, sigma

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x))


def largest_abs_eigenvectors(matrix, k):
    eigenvalues, eigenvectors = eigh(matrix)
    indices = np.argsort(np.abs(eigenvalues))[-k:]
    largest_eigenvectors = eigenvectors[:, indices]
    return largest_eigenvectors

def S(sigma, x):
    sigma_inv = np.linalg.inv(sigma)
    result = sigma_inv @ x.T
    return result.T

def delta_S(sigma):
    return np.linalg.inv(sigma)

def yxxt_avg(X, y):
    return ((X.T * y.T) @ X)/X.shape[0]

def EyTx(samples, y, sigma):
    return yxxt_avg(S(sigma, samples), y) - (y.sum()/samples.shape[0]) * delta_S(sigma)

def threshold_matrix(matrix, threshold):
    return np.where(matrix > threshold, matrix, 0)


def top_k_indices(arr, k = 5):
    indices = np.argpartition(arr, -k)[-k:]
    top_k_indices_sorted = indices[np.argsort(arr[indices])[::-1]]
    return top_k_indices_sorted


def test_origin(X,y, sigma, k):
    r = EyTx(X, y, sigma)
    top5 = largest_abs_eigenvectors(r, k = k)
    return top_k_indices(np.linalg.norm(top5, axis=1), k)

def test_num(X, y, sigma, k, num):
    r = EyTx(X, y, sigma)
    top5 = largest_abs_eigenvectors(r, k = k)
    return top_k_indices(np.linalg.norm(top5, axis=1), num)

def calculate_accuracy(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    intersection = set1.intersection(set2)
    accuracy = len(intersection) / len(set1)
    return accuracy




# Data Generation Functions
def random_rotation_matrix(s: int) -> np.ndarray:
    """Generate a random s x s rotation matrix using QR decomposition."""
    A = np.random.normal(0, 1, (s, s))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def create_sigma_matrix(size: int, rho: float) -> np.ndarray:
    """Create a Toeplitz covariance matrix with correlation rho."""
    sigma = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            sigma[i, j] = rho ** abs(i - j)
    return sigma


def generate_samples_rho(d: int, n_samples: int, rho: float = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Generate multivariate Gaussian samples with given correlation structure."""
    sigma = create_sigma_matrix(d, rho)
    X = np.random.multivariate_normal(mean=np.zeros(d), cov=sigma, size=n_samples)
    return X, sigma


def quadratic_k(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    output1 = (A1.T @ X.T).T
    output1 = np.square(output1)
    linear_par = np.random.rand(k, 1) if func_params is None else func_params
    return (output1 @ linear_par).flatten()


def cos_func(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    output1 = (A1.T @ X.T).T
    part1 = 3 * np.square(output1)
    part2 = 4 * np.cos(output1)
    par = np.random.rand(k, 1) if func_params is None else func_params
    return ((part1 + part2) @ par).flatten()

def cos_func2(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    output1 = (A1.T @ X.T).T
    part1 = np.square(np.square(output1))
    part2 = np.cos(output1)
    par = np.random.rand(k, 1) if func_params is None else func_params
    return ((part1 + part2) @ par).flatten()

def cos_func3(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    output1 = (A1.T @ X.T).T
    part1 = np.square(np.square(output1))
    part2 = 10 * np.cos(output1)
    part3 = 2 * np.square(output1)
    par = np.random.rand(k, 1) if func_params is None else func_params
    return ((part1 - part2 + part3) @ par).flatten()

def exp_poly(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    output1 = (A1.T @ X.T).T
    part1 = np.exp(output1)
    part2 = np.square(output1)
    part3 = np.square(part2)
    par = np.random.rand(k, 1) if func_params is None else func_params
    return ((part1 - part2 + part3) @ par).flatten()

def additive(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    if k != 5:
        raise ValueError("additive function requires k=5")
    d = X.shape[1]
    if A1.shape != (d, 5):
        raise ValueError(f"A1 must have shape ({d}, 5), got {A1.shape}")

    A_quad = A1[:, 0:1]
    A_cos = A1[:, 1:2]
    A_cos2 = A1[:, 2:3]
    A_cos3 = A1[:, 3:4]
    A_exp = A1[:, 4:5]

    params = np.random.rand(5, 1) if func_params is None else func_params
    p_quad, p_cos, p_cos2, p_cos3, p_exp = np.split(params, 5)

    out1 = quadratic_k(X, A_quad, 1, p_quad)
    out2 = cos_func2(X, A_cos, 1, p_cos)
    out3 = cos_func3(X, A_cos2, 1, p_cos2)
    out4 = cos_func3(X, A_cos3, 1, p_cos3)
    out5 = exp_poly(X, A_exp, 1, p_exp)

    return out1 + out2 + out3 + out4 + out5

def multiplication(X: np.ndarray, A1: np.ndarray, k: int, func_params: np.ndarray = None) -> np.ndarray:
    if k != 5:
        raise ValueError("multiplication function requires k=5")
    d = X.shape[1]
    if A1.shape != (d, 5):
        raise ValueError(f"A1 must have shape ({d}, 5), got {A1.shape}")

    A_quad1 = A1[:, 0:1]
    A_quad2 = A1[:, 1:2]
    A_cos2 = A1[:, 2:3]
    A_cos3 = A1[:, 3:4]
    A_exp = A1[:, 4:5]

    params = np.random.rand(5, 1) if func_params is None else func_params
    p_quad1, p_quad2, p_cos2, p_cos3, p_exp = np.split(params, 5)

    o1 = quadratic_k(X, A_quad1, 1, p_quad1)
    o2 = quadratic_k(X, A_quad2, 1, p_quad2)
    o3 = cos_func2(X, A_cos2, 1, p_cos2)
    o4 = cos_func3(X, A_cos3, 1, p_cos3)
    o5 = exp_poly(X, A_exp, 1, p_exp)

    return o1 * o3 + o5 + o2 * o4


def func_choose(X: np.ndarray, A1: np.ndarray, k: int, func_type: str, func_params: np.ndarray = None) -> np.ndarray:
    """Select and apply the specified nonlinear function."""
    mapping = {
        'quadratic': quadratic_k,
        'cos': cos_func,
        'cos2': cos_func2,
        'cos3': cos_func3,
        'exp_poly': exp_poly,
        'additive': additive,
        'multiplication': multiplication
    }
    if func_type not in mapping:
        raise ValueError(f"Unsupported function type: {func_type}")
    return mapping[func_type](X, A1, k, func_params)


# Performance Metrics
def calculate_selection_metrics(true_indices: np.ndarray, predicted_indices: np.ndarray, d: int) -> dict:
    """Compute feature selection metrics (TPR, FPR, precision, recall, F1)."""
    true_set = set(true_indices)
    pred_set = set(predicted_indices)

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = d - len(true_set | pred_set)

    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(true_set) if true_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {'tpr': tpr, 'fpr': fpr, 'precision': precision, 'recall': recall,
           'f1_score': f1, 'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn}



def create_sigma_matrix_t(size, rho=0.5):
    """Create correlation matrix for t-distribution"""
    sigma = np.ones((size, size)) * rho
    np.fill_diagonal(sigma, 1)
    return sigma

def generate_t_samples_local(d, n_samples, nu, iid=True):
    """Generate samples from multivariate t-distribution"""
    if iid:
        sigma = np.eye(d)
    else:
        sigma = create_sigma_matrix_t(d)
    
    samples = multivariate_t.rvs(loc=np.zeros(d), shape=sigma, df=nu, size=n_samples) 
    return samples, sigma

def generate_t_samples_local(d, n_samples, nu, iid=True):
    """Generate samples from multivariate t-distribution"""
    if iid:
        sigma = np.eye(d)
    else:
        sigma = create_sigma_matrix_t(d)
    
    samples = multivariate_t.rvs(loc=np.zeros(d), shape=sigma, df=nu, size=n_samples) 
    return samples, sigma

def multivariate_t_hessian(x, nu, mu=None, Sigma=None):
    """Compute Hessian matrix of multivariate t-distribution"""
    d = x.shape[0]
    x = x.reshape(-1, 1)
    
    if mu is None:
        mu = np.zeros(d).reshape(-1, 1)
    else:
        mu = np.array(mu).reshape(-1, 1)
    
    if Sigma is None:
        Sigma = np.eye(d)
    else:
        Sigma = np.array(Sigma)
    
    x_minus_mu = x - mu
    Sigma_inv = np.linalg.inv(Sigma)
    quad_form = np.dot(x_minus_mu.T, np.dot(Sigma_inv, x_minus_mu)).item()
    denom = (nu + quad_form) ** 2
    
    Sigma_inv_x_minus_mu = np.dot(Sigma_inv, x_minus_mu)
    term1 = 2 * np.dot(Sigma_inv_x_minus_mu, Sigma_inv_x_minus_mu.T)
    term2 = (nu + quad_form) * Sigma_inv
    
    hessian = (nu + d) * (term1 - term2) / denom
    return hessian

def compute_mean_yT_local(y, X, nu, mu=None, Sigma=None):
    """Compute mean_yT matrix for t-distribution (vectorized version)"""
    n_samples, d = X.shape
    if y.shape[0] != n_samples:
        raise ValueError("y and X must have the same number of samples (rows).")

    if mu is None:
        mu = np.zeros(d)
    if Sigma is None:
        Sigma = np.eye(d)
    else:
        Sigma = np.array(Sigma)

    Sigma_inv = np.linalg.pinv(Sigma)
    X_centered = X - mu  # Shape: (n_samples, d)
    quad_form = np.sum((X_centered @ Sigma_inv) * X_centered, axis=1)  # Shape: (n_samples,)
    denom = nu + quad_form  # Shape: (n_samples,)

    c1 = (nu + d) * (2 + nu + d) / denom**2  # Shape: (n_samples,)
    c2 = (nu + d) / denom  # Shape: (n_samples,)
    V = Sigma_inv @ X_centered.T  # Shape: (d, n_samples)

    sum_term1 = np.einsum('ai,i,bi->ab', V, y * c1, V)  # Shape: (d, d)
    sum_term2 = np.sum(y * c2) * Sigma_inv  # Shape: (d, d)

    sum_yT = sum_term1 - sum_term2  # Shape: (d, d)
    mean_yT = sum_yT / n_samples  # Shape: (d, d)
    
    return mean_yT

def top_k_eigenvectors_local(matrix, k=5):
    eigenvalues, eigenvectors = eigh(matrix)
    top_k_indices = np.argsort(np.abs(eigenvalues))[-k:]
    top_k_eigenvectors = eigenvectors[:, top_k_indices]
    return top_k_eigenvectors

def top_k_indices_local(arr, k=5):
    abs_arr = np.abs(arr)
    indices = np.argpartition(abs_arr, -k)[-k:]
    top_k_indices_sorted = indices[np.argsort(abs_arr[indices])[::-1]]
    return top_k_indices_sorted


    