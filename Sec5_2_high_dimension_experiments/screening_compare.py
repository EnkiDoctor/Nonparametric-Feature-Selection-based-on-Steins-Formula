import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.linalg import eigh
from scipy.stats import multivariate_t
import time
import warnings
import sys
import os
import gc
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

dfs_path = os.path.join(parent_dir, "deep_feature_selection")
prediction_path = os.path.join(parent_dir, "prediction")
screening_path = os.path.join(parent_dir, "screening")
t_path = os.path.join(parent_dir, "server_t")

for p in [dfs_path, prediction_path, screening_path, t_path]:
    if p not in sys.path:
        sys.path.append(p)

try:
    from models import Net_linear, Net_nonlinear
    from dfs import DFS_epoch, training_l
    from utils import data_load_l, measure, mse
    DFS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: DFS modules not available: {e}")
    DFS_AVAILABLE = False

# Import variable selection modules
try:
    from variable_selection import *
    STEIN_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Stein modules not available: {e}")
    STEIN_AVAILABLE = False

# Local t-distribution functions (to avoid conflicts)
T_DIST_AVAILABLE = True

# LassoNet import
try:
    from lassonet import LassoNetRegressor
    LASSONET_AVAILABLE = True
except ImportError:
    print("Warning: LassoNet not available. Install with: pip install lassonet")
    LASSONET_AVAILABLE = False

# Lasso import
try:
    from sklearn.linear_model import Lasso
    LASSO_AVAILABLE = True
except ImportError:
    print("Warning: Lasso not available. Install scikit-learn")
    LASSO_AVAILABLE = False

warnings.filterwarnings('ignore')

def get_device(prefer_gpu: bool = True, device_id: int = 0, verbose: bool = True) -> torch.device:
    """Get available computation device"""
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        if verbose:
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            print(f"Using GPU: {gpu_name} (Memory: {gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        if verbose:
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU")
            else:
                print("Manually selected CPU")
    return device

def clear_gpu_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class LassoNetSelector:
    """LassoNet feature selector"""
    
    def __init__(self, random_seed: int = 42, hidden_dims: tuple = (100, 50), standardize: bool = False,
                 lambda_start: float = 100, path_multiplier: float = 1.5, stable: bool = False):
        self.random_seed = random_seed
        self.hidden_dims = hidden_dims
        self.standardize = standardize
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.stable = stable
        
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       num_features_to_select: int = 5) -> Tuple[np.ndarray, float]:
        """Perform feature selection using LassoNet"""
        if not LASSONET_AVAILABLE:
            raise ImportError("LassoNet not available")
        
        start_time = time.time()
        
        try:
            if self.standardize:
                scaler = StandardScaler()
                X_train_proc = scaler.fit_transform(X_train)
                y_train_proc = y_train.copy()
            else:
                X_train_proc = X_train.copy()
                y_train_proc = y_train.copy()
            
            model = LassoNetRegressor(hidden_dims=self.hidden_dims, random_state=self.random_seed, 
                                     lambda_start=self.lambda_start, path_multiplier=self.path_multiplier)
            
            if self.stable:
                oracle, order, wrong, paths, prob = model.stability_selection(X_train_proc, y_train_proc)
                selected_indices = np.sort(order[:num_features_to_select])
            else:
                X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                    X_train_proc, y_train_proc, test_size=0.2, random_state=self.random_seed
                )
                
                path = model.path(X_train_split, y_train_split, return_state_dicts=True)
                
                best_val_score = -float('inf')
                best_model_idx = -1
                
                for idx, save in enumerate(path):
                    try:
                        model.load(save.state_dict)
                        val_score = model.score(X_val_split, y_val_split)
                        if val_score > best_val_score:
                            best_val_score = val_score
                            best_model_idx = idx
                    except:
                        continue
                
                if best_model_idx >= 0:
                    model.load(path[best_model_idx].state_dict)
                else:
                    middle_idx = len(path) // 2
                    model.load(path[middle_idx].state_dict)
                
                feature_importances = model.feature_importances_
                if hasattr(feature_importances, 'detach'):
                    feature_importances = feature_importances.detach().cpu().numpy()
                elif hasattr(feature_importances, 'numpy'):
                    feature_importances = feature_importances.numpy()
                
                selected_indices = np.argsort(feature_importances)[-num_features_to_select:]
                selected_indices = np.sort(selected_indices)
            
            training_time = time.time() - start_time
            return selected_indices, training_time
            
        except Exception as e:
            training_time = time.time() - start_time
            np.random.seed(self.random_seed)
            selected_indices = np.sort(np.random.choice(X_train.shape[1], num_features_to_select, replace=False))
            return selected_indices, training_time

class LassoSelector:
    """Lasso feature selector"""

    def __init__(self, random_seed: int = 42, alpha: float = 0.1, standardize: bool = True):
        self.random_seed = random_seed
        self.alpha = alpha
        self.standardize = standardize
        
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       num_features_to_select: int = 5) -> Tuple[np.ndarray, float]:
        """Perform feature selection using Lasso"""
        if not LASSO_AVAILABLE:
            raise ImportError("Lasso not available")
        
        start_time = time.time()
        
        if self.standardize:
            scaler = StandardScaler()
            X_train_proc = scaler.fit_transform(X_train)
        else:
            X_train_proc = X_train.copy()
        
        y_train_proc = y_train.copy()
        
        lasso = Lasso(alpha=self.alpha, random_state=self.random_seed)
        lasso.fit(X_train_proc, y_train_proc)
        
        importance = np.abs(lasso.coef_)
        selected_indices = np.argsort(importance)[-num_features_to_select:]
        selected_indices = np.sort(selected_indices)
        
        training_time = time.time() - start_time
        return selected_indices, training_time

class DFSSelector:
    """DFS feature selector"""
    
    def __init__(self, random_seed: int = 42, device: torch.device = None, 
                 n_hidden1: int = 100, n_hidden2: int = 50, learning_rate: float = 0.001,
                 weight_decay_c: float = 1.0, step: int = 4):
        self.random_seed = random_seed
        self.device = device if device is not None else get_device(verbose=False)
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.learning_rate = learning_rate
        self.weight_decay_c = weight_decay_c
        self.step = step
        
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       s: int = 5, Ts: int = 25, epochs: int = 10) -> Tuple[np.ndarray, float]:
        """Perform feature selection using DFS"""
        if not DFS_AVAILABLE:
            raise ImportError("DFS not available")
            
        start_time = time.time()
        
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
            torch.cuda.manual_seed_all(self.random_seed)
        
        clear_gpu_memory()
        
        X = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        Y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device)
        
        n_samples, p = X_train.shape
        c = self.weight_decay_c
        n_hidden1 = self.n_hidden1
        n_hidden2 = self.n_hidden2
        learning_rate = self.learning_rate
        step = self.step
        
        model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1).to(self.device)
        best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1).to(self.device)
        
        optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=0.0025*c)
        optimizer0 = torch.optim.Adam(model.hidden0.parameters(), lr=learning_rate, weight_decay=0.0005*c)
        
        lf = torch.nn.MSELoss()
        
        hist = []
        SUPP = []
        supp_x = list(range(p))
        SUPP.append(supp_x.copy())
        best_supp = supp_x.copy()
        
        for epoch in range(epochs):
            model, supp_x, LOSS = DFS_epoch(model, s, supp_x, X, Y, lf, optimizer0, optimizer, Ts, step)
            supp_x.sort()
            
            current_loss = lf(model(X), Y).cpu().data.numpy().tolist()
            hist.append(current_loss)
            SUPP.append(supp_x.copy())
            
            if hist[-1] == min(hist):
                best_model.load_state_dict(model.state_dict())
                best_supp = supp_x.copy()
            
            if len(SUPP[-1]) == len(SUPP[-2]) and (np.array(SUPP[-1]) == np.array(SUPP[-2])).all():
                break
        
        training_time = time.time() - start_time
        
        if len(best_supp) > s or len(best_supp) == 0:
            print(f"Warning: DFS abnormal output (selected {len(best_supp)} features), using random fallback")
            np.random.seed(self.random_seed)
            best_supp = np.sort(np.random.choice(p, s, replace=False))
       
        del model, best_model, optimizer, optimizer0, X, Y
        clear_gpu_memory()
        gc.collect()
        
        return np.array(list(best_supp)), training_time

class SteinSelector:
    """Stein feature selector"""
    
    def __init__(self, random_seed: int = 42, distribution_type: str = 'gaussian'):
        self.random_seed = random_seed
        self.distribution_type = distribution_type  # 'gaussian' or 't'
        
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       sigma: np.ndarray, num_features_to_select: int = 5, 
                       nu: float = 5) -> Tuple[np.ndarray, float]:
        if not STEIN_AVAILABLE and self.distribution_type == 'gaussian':
            raise ImportError("Stein method not available")
        if not T_DIST_AVAILABLE and self.distribution_type == 't':
            raise ImportError("t-distribution method not available")
        
        start_time = time.time()
        
        if self.distribution_type == 'gaussian':
            selected_indices = test_origin(X_train, y_train, sigma, num_features_to_select)
        elif self.distribution_type == 't':
            matrix = compute_mean_yT_local(y_train, X_train, nu, mu=None, Sigma=sigma)
            topk = top_k_eigenvectors_local(matrix, k=num_features_to_select)
            selected_indices = top_k_indices_local(np.linalg.norm(topk, axis=1), num_features_to_select)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        training_time = time.time() - start_time
        return selected_indices, training_time

class SteinScreeningSelector:
    """Stein with Screening feature selector"""
    
    def __init__(self, random_seed: int = 42, m: int = 10, delta: float = 0.9, distribution_type: str = 'gaussian'):
        self.random_seed = random_seed
        self.m = m
        self.delta = delta
        self.distribution_type = distribution_type
        
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       sigma: np.ndarray, num_features_to_select: int = 5, 
                       nu: float = 5) -> Tuple[np.ndarray, float]:
        if not STEIN_AVAILABLE and self.distribution_type == 'gaussian':
            raise ImportError("Stein method not available")
        if not T_DIST_AVAILABLE and self.distribution_type == 't':
            raise ImportError("t-distribution method not available")
        
        start_time = time.time()
        
        current_X = X_train.copy()
        current_sigma = sigma.copy()
        current_indices = np.arange(X_train.shape[1])
        current_d = X_train.shape[1]
        
        for i in range(self.m):
            if self.distribution_type == 'gaussian':
                matrix = EyTx(current_X, y_train, current_sigma)
            elif self.distribution_type == 't':
                matrix = compute_mean_yT_local(y_train, current_X, nu, mu=None, Sigma=current_sigma)
            else:
                raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
            
            diag_abs = np.abs(np.diag(matrix))
            next_d = int(np.ceil(current_d * self.delta))
            top_indices = np.argsort(diag_abs)[-next_d:]
            
            current_X = current_X[:, top_indices]
            current_sigma = current_sigma[top_indices][:, top_indices]
            current_indices = current_indices[top_indices]
            current_d = next_d
        
        if self.distribution_type == 'gaussian':
            selected_indices_in_current = test_origin(current_X, y_train, current_sigma, num_features_to_select)
        elif self.distribution_type == 't':
            matrix = compute_mean_yT_local(y_train, current_X, nu, mu=None, Sigma=current_sigma)
            topk = top_k_eigenvectors_local(matrix, k=num_features_to_select)
            selected_indices_in_current = top_k_indices_local(np.linalg.norm(topk, axis=1), num_features_to_select)
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        original_selected_indices = current_indices[selected_indices_in_current]
        
        training_time = time.time() - start_time
        return original_selected_indices, training_time

# ============================================================================
# Performance metrics
# ============================================================================

def calculate_selection_metrics(true_indices: np.ndarray, predicted_indices: np.ndarray, d: int) -> dict:
    """Calculate feature selection performance metrics"""
    true_set = set(true_indices)
    pred_set = set(predicted_indices)
    
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    tn = d - len(true_set.union(pred_set))
    
    precision = tp / len(pred_set) if len(pred_set) > 0 else 0
    recall = tp / len(true_set) if len(true_set) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    return {
        'tpr': tpr,
        'fpr': fpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

class GaussianFeatureSelectionComparator:
    """Comparator for five feature selection methods under Gaussian/t-distribution"""
    
    def __init__(self, d: int = 2000, k: int = 5, s: int = 5, 
                 n_samples: int = 1000, rho: float = 0, func_type: str = 'quadratic',
                 random_seed: int = 42, use_gpu: bool = True, device_id: int = 0,
                 distribution_type: str = 'gaussian', nu: float = 5,
                 lassonet_hidden_dims: tuple = (100, 50), lassonet_standardize: bool = True,
                 lassonet_lambda_start: float = 1, lassonet_path_multiplier: float = 1.5,
                 lassonet_stable: bool = False,
                 lasso_alpha: float = 0.1, lasso_standardize: bool = True,
                 dfs_n_hidden1: int = 100, dfs_n_hidden2: int = 50, dfs_learning_rate: float = 0.001,
                 dfs_weight_decay_c: float = 1.0, dfs_step: int = 4,
                 screening_m: int = 10, screening_delta: float = 0.9):
        self.d = d
        self.k = k
        self.s = s
        self.n_samples = n_samples
        self.rho = rho
        self.func_type = func_type
        self.random_seed = random_seed
        self.device = get_device(prefer_gpu=use_gpu, device_id=device_id, verbose=False)
        self.distribution_type = distribution_type
        self.nu = nu
        
        self.lassonet_selector = LassoNetSelector(
            random_seed=random_seed,
            hidden_dims=lassonet_hidden_dims,
            standardize=lassonet_standardize,
            lambda_start=lassonet_lambda_start,
            path_multiplier=lassonet_path_multiplier,
            stable=lassonet_stable
        ) if LASSONET_AVAILABLE else None
        
        self.lasso_selector = LassoSelector(
            random_seed=random_seed,
            alpha=lasso_alpha,
            standardize=lasso_standardize
        ) if LASSO_AVAILABLE else None
        
        self.dfs_selector = DFSSelector(
            random_seed=random_seed,
            device=self.device,
            n_hidden1=dfs_n_hidden1,
            n_hidden2=dfs_n_hidden2,
            learning_rate=dfs_learning_rate,
            weight_decay_c=dfs_weight_decay_c,
            step=dfs_step
        ) if DFS_AVAILABLE else None
        
        self.stein_selector = SteinSelector(
            random_seed=random_seed,
            distribution_type=distribution_type
        ) if (STEIN_AVAILABLE or T_DIST_AVAILABLE) else None
        
        self.stein_screening_selector = SteinScreeningSelector(
            random_seed=random_seed,
            m=screening_m,
            delta=screening_delta,
            distribution_type=distribution_type
        ) if (STEIN_AVAILABLE or T_DIST_AVAILABLE) else None
        
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate data (supports Gaussian and t-distribution)"""
        if self.distribution_type == 'gaussian':
            X_train, sigma = generate_samples_rho(self.d, self.n_samples, self.rho)
        elif self.distribution_type == 't':
            X_train, sigma = generate_t_samples_local(self.d, self.n_samples, self.nu, iid=(self.rho == 0))
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")
        
        A1 = np.zeros((self.d, self.k))
        true_indices = np.random.choice(self.d, self.s, replace=False)
        A1[true_indices] = np.random.rand(self.s, self.k)
        func_params = np.random.rand(self.k, 1)
        y_train = func_choose(X_train, A1, self.k, self.func_type, func_params).flatten()
        
        return X_train, y_train, true_indices, sigma
    
    def run_single_comparison(self, verbose: bool = False) -> dict:
        """Run one comparison of all five methods"""
        X_train, y_train, true_indices, sigma = self.generate_data()
        
        results = {
            'true_indices': true_indices,
            'stein_results': None,
            'stein_screening_results': None,
            'lassonet_results': None,
            'lasso_results': None,
            'dfs_results': None
        }
        
        if verbose:
            print(f"True feature indices: {sorted(true_indices)}")
        
        # Method 1: Stein
        if self.stein_selector is not None:
            if verbose: print("Running Stein method...")
            try:
                stein_indices, stein_time = self.stein_selector.select_features(
                    X_train, y_train, sigma, self.s, nu=self.nu
                )
                stein_metrics = calculate_selection_metrics(true_indices, stein_indices, self.d)
                stein_metrics['selected_indices'] = stein_indices
                stein_metrics['selection_time'] = stein_time
                results['stein_results'] = stein_metrics
                if verbose:
                    print(f"  Stein selected: {sorted(stein_indices)}")
                    print(f"  TPR: {stein_metrics['tpr']:.4f}, FPR: {stein_metrics['fpr']:.4f}")
                    print(f"  Time: {stein_time:.4f}s")
            except Exception as e:
                if verbose: print(f"  Stein failed: {e}")
        
        # Method 2: Stein with Screening
        if self.stein_screening_selector is not None:
            if verbose: print("Running Stein+Screening method...")
            try:
                stein_screening_indices, stein_screening_time = self.stein_screening_selector.select_features(
                    X_train, y_train, sigma, self.s, nu=self.nu
                )
                stein_screening_metrics = calculate_selection_metrics(true_indices, stein_screening_indices, self.d)
                stein_screening_metrics['selected_indices'] = stein_screening_indices
                stein_screening_metrics['selection_time'] = stein_screening_time
                results['stein_screening_results'] = stein_screening_metrics
                if verbose:
                    print(f"  Stein+Screening selected: {sorted(stein_screening_indices)}")
                    print(f"  TPR: {stein_screening_metrics['tpr']:.4f}, FPR: {stein_screening_metrics['fpr']:.4f}")
                    print(f"  Time: {stein_screening_time:.4f}s")
            except Exception as e:
                if verbose: print(f"  Stein+Screening failed: {e}")
        
        # Method 3: LassoNet
        if self.lassonet_selector is not None:
            if verbose:
                name = "LassoNet (stable)" if self.lassonet_selector.stable else "LassoNet"
                print(f"Running {name}...")
            try:
                lassonet_indices, lassonet_time = self.lassonet_selector.select_features(
                    X_train, y_train, self.s
                )
                lassonet_metrics = calculate_selection_metrics(true_indices, lassonet_indices, self.d)
                lassonet_metrics['selected_indices'] = lassonet_indices
                lassonet_metrics['selection_time'] = lassonet_time
                results['lassonet_results'] = lassonet_metrics
                if verbose:
                    name = "LassoNet (stable)" if self.lassonet_selector.stable else "LassoNet"
                    print(f"  {name} selected: {sorted(lassonet_indices)}")
                    print(f"  TPR: {lassonet_metrics['tpr']:.4f}, FPR: {lassonet_metrics['fpr']:.4f}")
                    print(f"  Time: {lassonet_time:.2f}s")
            except Exception as e:
                if verbose: print(f"  LassoNet failed: {e}")
        
        # Method 4: Lasso
        if self.lasso_selector is not None:
            if verbose: print("Running Lasso...")
            try:
                lasso_indices, lasso_time = self.lasso_selector.select_features(
                    X_train, y_train, self.s
                )
                lasso_metrics = calculate_selection_metrics(true_indices, lasso_indices, self.d)
                lasso_metrics['selected_indices'] = lasso_indices
                lasso_metrics['selection_time'] = lasso_time
                results['lasso_results'] = lasso_metrics
                if verbose:
                    print(f"  Lasso selected: {sorted(lasso_indices)}")
                    print(f"  TPR: {lasso_metrics['tpr']:.4f}, FPR: {lasso_metrics['fpr']:.4f}")
                    print(f"  Time: {lasso_time:.4f}s")
            except Exception as e:
                if verbose: print(f"  Lasso failed: {e}")
        
        # Method 5: DFS
        if self.dfs_selector is not None:
            if verbose: print("Running DFS...")
            try:
                dfs_indices, dfs_time = self.dfs_selector.select_features(
                    X_train, y_train, self.s, Ts=25, epochs=10
                )
                dfs_metrics = calculate_selection_metrics(true_indices, dfs_indices, self.d)
                dfs_metrics['selected_indices'] = dfs_indices
                dfs_metrics['selection_time'] = dfs_time
                results['dfs_results'] = dfs_metrics
                if verbose:
                    print(f"  DFS selected: {sorted(dfs_indices)}")
                    print(f"  TPR: {dfs_metrics['tpr']:.4f}, FPR: {dfs_metrics['fpr']:.4f}")
                    print(f"  Time: {dfs_time:.2f}s")
            except Exception as e:
                if verbose: print(f"  DFS failed: {e}")
        
        return results
    
    def run_multiple_comparisons(self, n_experiments: int = 50, verbose: bool = True) -> dict:
        """Run multiple experiments and return statistics"""
        dist_name = "t-distribution" if self.distribution_type == 't' else "Gaussian"
        print(f"Running {n_experiments} experiments under {dist_name}")
        print(f"Parameters: d={self.d}, k={self.k}, s={self.s}, n_samples={self.n_samples}, rho={self.rho}")
        if self.distribution_type == 't':
            print(f"t-distribution df: {self.nu}")
        print("="*80)
        
        metrics = {
            'stein': {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'time': []},
            'stein_screening': {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'time': []},
            'lassonet': {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'time': []},
            'lasso': {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'time': []},
            'dfs': {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'time': []}
        }
        
        successful_experiments = 0
        success_counts = {method: 0 for method in metrics.keys()}
        
        for i in range(n_experiments):
            if verbose:
                print(f"Experiment {i+1}/{n_experiments}")
            
            current_seed = self.random_seed + i * 1000
            self.random_seed = current_seed
            np.random.seed(current_seed)
            torch.manual_seed(current_seed)
            
            try:
                results = self.run_single_comparison(verbose=False)
                
                method_keys = ['stein', 'stein_screening', 'lassonet', 'lasso', 'dfs']
                result_keys = ['stein_results', 'stein_screening_results', 'lassonet_results', 'lasso_results', 'dfs_results']
                
                for method_key, result_key in zip(method_keys, result_keys):
                    if results[result_key] is not None:
                        res = results[result_key]
                        metrics[method_key]['tpr'].append(res['tpr'])
                        metrics[method_key]['fpr'].append(res['fpr'])
                        metrics[method_key]['precision'].append(res['precision'])
                        metrics[method_key]['recall'].append(res['recall'])
                        metrics[method_key]['f1_score'].append(res['f1_score'])
                        metrics[method_key]['time'].append(res['selection_time'])
                        success_counts[method_key] += 1
                    else:
                        for key in metrics[method_key].keys():
                            metrics[method_key][key].append(None)
                
                successful_experiments += 1
                
            except Exception as e:
                if verbose:
                    print(f"  Experiment {i+1} failed: {e}")
                continue
        
        def safe_stats(values):
            clean_values = [v for v in values if v is not None]
            if len(clean_values) == 0:
                return {'mean': None, 'std': None, 'count': 0}
            return {
                'mean': np.mean(clean_values),
                'std': np.std(clean_values),
                'count': len(clean_values)
            }
        
        summary_stats = {}
        for method in metrics.keys():
            summary_stats[method] = {}
            for metric in ['tpr', 'fpr', 'precision', 'recall', 'f1_score', 'time']:
                summary_stats[method][metric] = safe_stats(metrics[method][metric])
        
        print(f"\nPerformance comparison of five methods ({successful_experiments} experiments):")
        print("="*80)
        
        method_names = {
            'stein': 'Stein',
            'stein_screening': 'Stein+Screening',
            'lassonet': 'LassoNet (stable)' if self.lassonet_selector and self.lassonet_selector.stable else 'LassoNet',
            'lasso': 'Lasso',
            'dfs': 'DFS'
        }
        
        for method in metrics.keys():
            print(f"\n{method_names[method]} method:")
            if summary_stats[method]['tpr']['count'] > 0:
                print(f"  TPR:   {summary_stats[method]['tpr']['mean']:.4f} ± {summary_stats[method]['tpr']['std']:.4f}")
                print(f"  FPR:   {summary_stats[method]['fpr']['mean']:.4f} ± {summary_stats[method]['fpr']['std']:.4f}")
                print(f"  Avg time: {summary_stats[method]['time']['mean']:.4f}s")
            else:
                print(f"  All experiments failed")
        
        print(f"\n=== TPR ranking (higher is better) ===")
        tpr_ranking = []
        for method in metrics.keys():
            if summary_stats[method]['tpr']['mean'] is not None:
                tpr_ranking.append((method_names[method], summary_stats[method]['tpr']['mean']))
        tpr_ranking.sort(key=lambda x: x[1], reverse=True)
        for i, (method, tpr) in enumerate(tpr_ranking, 1):
            print(f"  {i}. {method}: {tpr:.10f}")
        
        print(f"\n=== FPR ranking (lower is better) ===")
        fpr_ranking = []
        for method in metrics.keys():
            if summary_stats[method]['fpr']['mean'] is not None:
                fpr_ranking.append((method_names[method], summary_stats[method]['fpr']['mean']))
        fpr_ranking.sort(key=lambda x: x[1])
        for i, (method, fpr) in enumerate(fpr_ranking, 1):
            print(f"  {i}. {method}: {fpr:.10f}")
        
        return {
            'raw_metrics': metrics,
            'summary_stats': summary_stats,
            'success_counts': success_counts,
            'total_experiments': successful_experiments
        }



def run_gaussian_comparison(d: int = 2000, k: int = 5, s: int = 5, 
                           n_samples: int = 1000, rho: float = 0, func_type: str = 'quadratic',
                           n_experiments: int = 50, random_seed: int = 42, 
                           use_gpu: bool = True, device_id: int = 0,
                           distribution_type: str = 'gaussian', nu: float = 5,
                           lassonet_hidden_dims: tuple = (100, 50), lassonet_standardize: bool = True,
                           lassonet_lambda_start: float = 1, lassonet_path_multiplier: float = 1.5,
                           lassonet_stable: bool = False,
                           lasso_alpha: float = 0.1, lasso_standardize: bool = True,
                           dfs_n_hidden1: int = 100, dfs_n_hidden2: int = 50, dfs_learning_rate: float = 0.001,
                           dfs_weight_decay_c: float = 1.0, dfs_step: int = 4,
                           screening_m: int = 10, screening_delta: float = 0.9) -> dict:
    """Run comparison of five feature selection methods under Gaussian/t-distribution"""
    
    comparator = GaussianFeatureSelectionComparator(
        d=d, k=k, s=s, n_samples=n_samples, rho=rho, func_type=func_type,
        random_seed=random_seed, use_gpu=use_gpu, device_id=device_id,
        distribution_type=distribution_type, nu=nu,
        lassonet_hidden_dims=lassonet_hidden_dims,
        lassonet_standardize=lassonet_standardize,
        lassonet_lambda_start=lassonet_lambda_start,
        lassonet_path_multiplier=lassonet_path_multiplier,
        lassonet_stable=lassonet_stable,
        lasso_alpha=lasso_alpha,
        lasso_standardize=lasso_standardize,
        dfs_n_hidden1=dfs_n_hidden1,
        dfs_n_hidden2=dfs_n_hidden2,
        dfs_learning_rate=dfs_learning_rate,
        dfs_weight_decay_c=dfs_weight_decay_c,
        dfs_step=dfs_step,
        screening_m=screening_m,
        screening_delta=screening_delta
    )
    
    return comparator.run_multiple_comparisons(n_experiments=n_experiments, verbose=True)

def run_multiple_function_comparison(func_types: List[str] = None, d: int = 2000, k: int = 5, s: int = 5,
                                    n_samples: int = 1000, rho: float = 0,
                                    n_experiments: int = 20, random_seed: int = 42,
                                    use_gpu: bool = True, device_id: int = 0,
                                    distribution_type: str = 'gaussian', nu: float = 5,
                                    lassonet_hidden_dims: tuple = (100, 50), lassonet_standardize: bool = True,
                                    lassonet_lambda_start: float = 100, lassonet_path_multiplier: float = 1.5,
                                    lassonet_stable: bool = False,
                                    lasso_alpha: float = 0.1, lasso_standardize: bool = True,
                                    dfs_n_hidden1: int = 100, dfs_n_hidden2: int = 50, dfs_learning_rate: float = 0.05,
                                    dfs_weight_decay_c: float = 1.0, dfs_step: int = 4,
                                    screening_m: int = 10, screening_delta: float = 0.9) -> dict:
    """Run comparison across multiple function types"""
    
    if func_types is None:
        func_types = ['quadratic', 'cos', 'cos2', 'cos3', 'exp_poly', 'additive', 'multiplication']
    
    all_results = {}
    
    print(f"Starting multi-function comparison...")
    print(f"Function types: {func_types}")
    print(f"Parameters: d={d}, k={k}, s={s}, n_samples={n_samples}, rho={rho}")
    print(f"Running {n_experiments} experiments per function type")
    print("="*80)
    
    for i, func_type in enumerate(func_types, 1):
        print(f"\n[{i}/{len(func_types)}] Testing function type: {func_type.upper()}")
        print("-" * 60)
        
        try:
            func_seed = random_seed + i * 10000
            
            results = run_gaussian_comparison(
                d=d, k=k, s=s, n_samples=n_samples, rho=rho, func_type=func_type,
                n_experiments=n_experiments, random_seed=func_seed,
                use_gpu=use_gpu, device_id=device_id,
                distribution_type=distribution_type, nu=nu,
                lassonet_hidden_dims=lassonet_hidden_dims,
                lassonet_standardize=lassonet_standardize,
                lassonet_lambda_start=lassonet_lambda_start,
                lassonet_path_multiplier=lassonet_path_multiplier,
                lassonet_stable=lassonet_stable,
                lasso_alpha=lasso_alpha,
                lasso_standardize=lasso_standardize,
                dfs_n_hidden1=dfs_n_hidden1,
                dfs_n_hidden2=dfs_n_hidden2,
                dfs_learning_rate=dfs_learning_rate,
                dfs_weight_decay_c=dfs_weight_decay_c,
                dfs_step=dfs_step,
                screening_m=screening_m,
                screening_delta=screening_delta
            )
            
            all_results[func_type] = results
            
            summary = results['summary_stats']
            print(f"\n{func_type.upper()} function summary:")
            methods = ['stein', 'stein_screening', 'lassonet', 'lasso', 'dfs']
            method_names = {
                'stein': 'Stein',
                'stein_screening': 'Stein+Screening',
                'lassonet': 'LassoNet',
                'lasso': 'Lasso',
                'dfs': 'DFS'
            }
            
            for method in methods:
                if summary[method]['tpr']['mean'] is not None:
                    tpr = summary[method]['tpr']['mean']
                    fpr = summary[method]['fpr']['mean']
                    print(f"  {method_names[method]}: TPR={tpr:.3f}, FPR={fpr:.3f}")
                else:
                    print(f"  {method_names[method]}: Failed")
            
        except Exception as e:
            print(f"Function type {func_type} failed: {e}")
            all_results[func_type] = None
    
    print(f"\n" + "="*80)
    print("Multi-function comparison completed!")
    
    # Performance comparison table by function type
    print(f"\n=== Performance comparison by function type ===")
    methods = ['stein', 'stein_screening', 'lassonet', 'lasso', 'dfs']
    method_names = {
        'stein': 'Stein',
        'stein_screening': 'Stein+Screening', 
        'lassonet': 'LassoNet',
        'lasso': 'Lasso',
        'dfs': 'DFS'
    }
    
    for func_type in func_types:
        if func_type in all_results and all_results[func_type] is not None:
            print(f"\n【{func_type.upper()} function】:")
            print("-" * 60)
            
            func_results = []
            for method in methods:
                summary_stats = all_results[func_type]['summary_stats'][method]
                if summary_stats['tpr']['mean'] is not None:
                    tpr = summary_stats['tpr']['mean']
                    fpr = summary_stats['fpr']['mean']
                    time_avg = summary_stats['time']['mean']
                    func_results.append((method_names[method], tpr, fpr, time_avg))
                else:
                    func_results.append((method_names[method], None, None, None))
            
            successful_results = [r for r in func_results if r[1] is not None]
            failed_results = [r for r in func_results if r[1] is None]
            
            if successful_results:
                successful_results.sort(key=lambda x: x[1], reverse=True)
                
                print("  TPR ranking (higher is better):")
                for i, (method, tpr, fpr, time_avg) in enumerate(successful_results, 1):
                    print(f"    {i}. {method}: TPR={tpr:.10f}, FPR={fpr:.10f}, Time={time_avg:.10f}s")
                
                print("\n  FPR ranking (lower is better):")
                successful_results.sort(key=lambda x: x[2])
                for i, (method, tpr, fpr, time_avg) in enumerate(successful_results, 1):
                    print(f"    {i}. {method}: FPR={fpr:.10f}, TPR={tpr:.10f}, Time={time_avg:.10f}s")
            
            if failed_results:
                print(f"\n  Failed methods: {[r[0] for r in failed_results]}")
        else:
            print(f"\n【{func_type.upper()} function】: Experiment failed")
    
    # Method success rate statistics
    print(f"\n=== Method success rate statistics ===")
    for method in methods:
        successful_count = 0
        total_count = len(func_types)
        
        for func_type in func_types:
            if (func_type in all_results and 
                all_results[func_type] is not None and
                all_results[func_type]['summary_stats'][method]['tpr']['mean'] is not None):
                successful_count += 1
        
        success_rate = successful_count / total_count * 100
        print(f"{method_names[method]}: {successful_count}/{total_count} ({success_rate:.1f}%)")
        
        if successful_count < total_count:
            failed_funcs = []
            for func_type in func_types:
                if (func_type not in all_results or 
                    all_results[func_type] is None or
                    all_results[func_type]['summary_stats'][method]['tpr']['mean'] is None):
                    failed_funcs.append(func_type)
            if failed_funcs:
                print(f"  Failed functions: {failed_funcs}")
    
    # Best method recommendation across functions
    print(f"\n=== Best method recommendation across functions ===")
    best_methods_by_func = {}
    
    for func_type in func_types:
        if func_type in all_results and all_results[func_type] is not None:
            best_tpr = -1
            best_method = None
            
            for method in methods:
                summary_stats = all_results[func_type]['summary_stats'][method]
                if summary_stats['tpr']['mean'] is not None:
                    tpr = summary_stats['tpr']['mean']
                    if tpr > best_tpr:
                        best_tpr = tpr
                        best_method = method_names[method]
            
            if best_method:
                best_methods_by_func[func_type] = (best_method, best_tpr)
    
    for func_type, (method, tpr) in best_methods_by_func.items():
        print(f"{func_type.upper()} function best method: {method} (TPR={tpr:.3f})")
    
    method_wins = {method: 0 for method in method_names.values()}
    for func_type, (method, tpr) in best_methods_by_func.items():
        method_wins[method] += 1
    
    print(f"\nOverall best method statistics:")
    for method, wins in sorted(method_wins.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method}: Best on {wins} functions")
    
    return all_results