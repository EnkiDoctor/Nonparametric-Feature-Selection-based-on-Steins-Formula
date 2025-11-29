import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple
import warnings
import time

# Add current and parent directories to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.append(p)

# Add custom module paths
dfs_path = os.path.join(parent_dir, "deep_feature_selection")
prediction_path = os.path.join(parent_dir, "prediction")
screening_path = os.path.join(parent_dir, "screening")
for p in [dfs_path, prediction_path, screening_path]:
    if p not in sys.path:
        sys.path.append(p)

warnings.filterwarnings('ignore')

# Optional dependency checks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available")
    TORCH_AVAILABLE = False

try:
    from sklearn.linear_model import Lasso
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available")
    SKLEARN_AVAILABLE = False

try:
    from lassonet import LassoNetRegressor
    LASSONET_AVAILABLE = True
except ImportError:
    print("Warning: LassoNet not available")
    LASSONET_AVAILABLE = False

try:
    from models import Net_linear, Net_nonlinear
    from dfs import DFS_epoch, training_l
    from utils import data_load_l, measure, mse
    DFS_AVAILABLE = True
except ImportError:
    print("Warning: DFS modules not available")
    DFS_AVAILABLE = False

try:
    from variable_selection import *
    STEIN_AVAILABLE = True
except ImportError:
    print("Warning: Stein modules not available")
    STEIN_AVAILABLE = False

try:
    from screening import *
    SCREENING_AVAILABLE = True
except ImportError:
    print("Warning: Screening modules not available")
    SCREENING_AVAILABLE = False


# Feature Selectors
class SteinSelector:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                        sigma: np.ndarray, num_features_to_select: int = 5) -> Tuple[np.ndarray, float]:
        if not STEIN_AVAILABLE:
            print("Stein not available, using random selection")
            np.random.seed(self.random_seed)
            idx = np.random.choice(X_train.shape[1], num_features_to_select, replace=False)
            return np.sort(idx), 0.001

        start = time.time()
        selected = test_origin(X_train, y_train, sigma, num_features_to_select)
        return selected, time.time() - start


class LassoSelector:
    def __init__(self, random_seed: int = 42, alpha: float = 0.1, standardize: bool = True):
        self.random_seed = random_seed
        self.alpha = alpha
        self.standardize = standardize

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                        num_features_to_select: int = 5) -> Tuple[np.ndarray, float]:
        if not SKLEARN_AVAILABLE:
            np.random.seed(self.random_seed)
            idx = np.random.choice(X_train.shape[1], num_features_to_select, replace=False)
            return np.sort(idx), 0.001

        start = time.time()
        X_proc = StandardScaler().fit_transform(X_train) if self.standardize else X_train
        lasso = Lasso(alpha=self.alpha, random_state=self.random_seed)
        lasso.fit(X_proc, y_train)
        importance = np.abs(lasso.coef_)
        selected = np.sort(np.argsort(importance)[-num_features_to_select:])
        return selected, time.time() - start


class LassoNetSelector:
    def __init__(self, random_seed: int = 42, hidden_dims=(50, 25), standardize: bool = False,
                 lambda_start: float = 50, path_multiplier: float = 1.25, stable: bool = False):
        self.random_seed = random_seed
        self.hidden_dims = hidden_dims
        self.standardize = standardize
        self.lambda_start = lambda_start
        self.path_multiplier = path_multiplier
        self.stable = stable

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                        num_features_to_select: int = 5) -> Tuple[np.ndarray, float]:
        if not LASSONET_AVAILABLE:
            np.random.seed(self.random_seed)
            idx = np.random.choice(X_train.shape[1], num_features_to_select, replace=False)
            return np.sort(idx), 0.001

        start = time.time()
        try:
            X_proc = StandardScaler().fit_transform(X_train) if self.standardize else X_train
            model = LassoNetRegressor(hidden_dims=self.hidden_dims, random_state=self.random_seed,
                                      lambda_start=self.lambda_start, path_multiplier=self.path_multiplier)

            if self.stable:
                _, order, _, _, _ = model.stability_selection(X_proc, y_train)
                selected = np.sort(order[:num_features_to_select])
            else:
                X_tr, X_val, y_tr, y_val = train_test_split(X_proc, y_train, test_size=0.2,
                                                            random_state=self.random_seed)
                path = model.path(X_tr, y_tr, return_state_dicts=True)
                best_idx = max(range(len(path)), key=lambda i: model.load(path[i].state_dict).score(X_val, y_val),
                               default=len(path)//2)
                model.load(path[best_idx].state_dict)
                imp = model.feature_importances_.detach().cpu().numpy()
                selected = np.sort(np.argsort(imp)[-num_features_to_select:])

            return selected, time.time() - start
        except Exception as e:
            print(f"LassoNet failed ({e}), falling back to random")
            np.random.seed(self.random_seed)
            idx = np.random.choice(X_train.shape[1], num_features_to_select, replace=False)
            return np.sort(idx), 0.001


class DFSSelector:
    def __init__(self, random_seed: int = 42, n_hidden1: int = 50, n_hidden2: int = 25,
                 learning_rate: float = 0.1, weight_decay_c: float = 1.0, step: int = 5):
        self.random_seed = random_seed
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.learning_rate = learning_rate
        self.weight_decay_c = weight_decay_c
        self.step = step

    def select_features(self, X_train: np.ndarray, y_train: np.ndarray,
                        s: int = 5, Ts: int = 25, epochs: int = 10) -> Tuple[np.ndarray, float]:
        if not (DFS_AVAILABLE and TORCH_AVAILABLE):
            np.random.seed(self.random_seed)
            idx = np.random.choice(X_train.shape[1], s, replace=False)
            return np.sort(idx), 0.001

        start = time.time()
        try:
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            X = torch.tensor(X_train, dtype=torch.float32).to(device)
            Y = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)

            n_samples, p = X_train.shape
            model = Net_nonlinear(p, self.n_hidden1, self.n_hidden2, 1).to(device)
            best_model = Net_nonlinear(p, self.n_hidden1, self.n_hidden2, 1).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate,
                                         weight_decay=0.0025 * self.weight_decay_c)
            optimizer0 = torch.optim.Adam(model.hidden0.parameters(), lr=self.learning_rate,
                                          weight_decay=0.0005 * self.weight_decay_c)
            loss_fn = torch.nn.MSELoss()

            hist, SUPP = [], []
            supp_x = list(range(p))
            SUPP.append(supp_x.copy())
            best_supp = supp_x.copy()

            for epoch in range(epochs):
                model, supp_x, _ = DFS_epoch(model, s, supp_x, X, Y, loss_fn, optimizer0, optimizer, Ts, self.step)
                supp_x.sort()
                current_loss = loss_fn(model(X), Y).item()
                hist.append(current_loss)
                SUPP.append(supp_x.copy())
                if hist[-1] == min(hist):
                    best_model.load_state_dict(model.state_dict())
                    best_supp = supp_x.copy()
                if len(SUPP[-1]) == len(SUPP[-2]) and np.array_equal(SUPP[-1], SUPP[-2]):
                    break

            result = np.array(best_supp)
            if len(result) != s:
                print(f"DFS returned {len(result)} features, using random fallback")
                result = np.sort(np.random.choice(p, s, replace=False))

            return result, time.time() - start
        except Exception as e:
            print(f"DFS failed ({e}), using random selection")
            np.random.seed(self.random_seed)
            idx = np.random.choice(X_train.shape[1], s, replace=False)
            return np.sort(idx), 0.001


# Low-Dimensional Comparison Framework
class LowDimensionComparator:
    def __init__(self, d: int = 200, k: int = 5, s: int = 5, n_samples: int = 1000,
                 rho: float = 0, func_type: str = 'quadratic', random_seed: int = 42,
                 selected_methods: list = None, **kwargs):
        self.d = d
        self.k = k
        self.s = s
        self.n_samples = n_samples
        self.rho = rho
        self.func_type = func_type
        self.random_seed = random_seed

        self.available_methods = ['stein', 'lassonet', 'lasso', 'dfs']
        self.selected_methods = self.available_methods if selected_methods is None else [
            m for m in selected_methods if m in self.available_methods]

        self.selectors = {}
        if 'stein' in self.selected_methods:
            self.selectors['stein'] = SteinSelector(random_seed)
        if 'lasso' in self.selected_methods:
            self.selectors['lasso'] = LassoSelector(random_seed,
                                                    alpha=kwargs.get('lasso_alpha', 0.1),
                                                    standardize=kwargs.get('lasso_standardize', True))
        if 'lassonet' in self.selected_methods:
            self.selectors['lassonet'] = LassoNetSelector(random_seed,
                                                          hidden_dims=kwargs.get('lassonet_hidden_dims', (50, 25)),
                                                          standardize=kwargs.get('lassonet_standardize', True),
                                                          lambda_start=kwargs.get('lassonet_lambda_start', 50),
                                                          path_multiplier=kwargs.get('lassonet_path_multiplier', 1.25),
                                                          stable=kwargs.get('lassonet_stable', False))
        if 'dfs' in self.selected_methods:
            self.selectors['dfs'] = DFSSelector(random_seed,
                                                n_hidden1=kwargs.get('dfs_n_hidden1', 50),
                                                n_hidden2=kwargs.get('dfs_n_hidden2', 25),
                                                learning_rate=kwargs.get('dfs_learning_rate', 0.1),
                                                weight_decay_c=kwargs.get('dfs_weight_decay_c', 1.0),
                                                step=kwargs.get('dfs_step', 5))

        np.random.seed(random_seed)

    def generate_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        X_train, sigma = generate_samples_rho(self.d, self.n_samples, self.rho)
        true_indices = np.random.choice(self.d, self.s, replace=False)
        A1 = np.zeros((self.d, self.k))
        A1[true_indices] = random_rotation_matrix(self.s)
        func_params = np.random.rand(self.k, 1) + 2
        y_train = func_choose(X_train, A1, self.k, self.func_type, func_params).flatten()
        return X_train, y_train, true_indices, sigma

    def run_single_comparison(self, verbose: bool = False) -> dict:
        X_train, y_train, true_indices, sigma = self.generate_data()
        results = {'true_indices': true_indices}
        for m in self.available_methods:
            results[f'{m}_results'] = None

        for method in self.selected_methods:
            try:
                if method == 'stein':
                    idx, t = self.selectors['stein'].select_features(X_train, y_train, sigma, self.s)
                elif method == 'lassonet':
                    idx, t = self.selectors['lassonet'].select_features(X_train, y_train, self.s)
                elif method == 'lasso':
                    idx, t = self.selectors['lasso'].select_features(X_train, y_train, self.s)
                elif method == 'dfs':
                    idx, t = self.selectors['dfs'].select_features(X_train, y_train, self.s, Ts=25, epochs=10)

                metrics = calculate_selection_metrics(true_indices, idx, self.d)
                metrics.update({'selected_indices': idx, 'selection_time': t})
                results[f'{method}_results'] = metrics
            except Exception as e:
                if verbose:
                    print(f"{method.upper()} failed: {e}")
        return results

    def run_multiple_comparisons(self, n_experiments: int = 50, verbose: bool = True) -> dict:
        print(f"Running {n_experiments} experiments with methods: {self.selected_methods}")
        metrics = {m: {'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [], 'time': []}
                   for m in self.selected_methods}

        for i in range(n_experiments):
            seed = self.random_seed + i * 1000
            np.random.seed(seed)
            torch.manual_seed(seed) if TORCH_AVAILABLE else None
            try:
                res = self.run_single_comparison()
                for m in self.selected_methods:
                    r = res.get(f'{m}_results')
                    if r:
                        for k in metrics[m]:
                            metrics[m][k].append(r.get(k if k != 'time' else 'selection_time'))
            except Exception as e:
                if verbose:
                    print(f"Experiment {i+1} failed: {e}")

        summary = {}
        for m in metrics:
            summary[m] = {k: {'mean': np.mean([x for x in v if x is not None]),
                              'std': np.std([x for x in v if x is not None])}
                          if any(x is not None for x in v) else {'mean': None, 'std': None}
                          for k, v in metrics[m].items()}
        return {'summary_stats': summary, 'raw_metrics': metrics}


class LowDimensionSampleSizeComparison:
    def __init__(self, data_dir: str = "low_dimension_data", distribution_type: str = 'gaussian',
                 nu: float = 5, selected_methods: list = None, selected_functions: list = None):
        self.data_dir = data_dir
        self.distribution_type = distribution_type
        self.nu = nu
        self.sample_sizes = [100, 500, 1000, 2000, 3000, 4000, 5000]
        self.available_functions = ['quadratic', 'cos', 'cos2', 'cos3', 'exp_poly', 'additive', 'multiplication']
        self.func_types = self.available_functions if selected_functions is None else [
            f for f in selected_functions if f in self.available_functions]
        self.methods = ['stein', 'lassonet', 'lasso', 'dfs'] if selected_methods is None else [
            m for m in selected_methods if m in ['stein', 'lassonet', 'lasso', 'dfs']]

        self.method_names = {'stein': 'Stein', 'lassonet': 'LassoNet', 'lasso': 'Lasso', 'dfs': 'DFS'}
        self.method_colors = {'stein': '#1f77b4', 'lassonet': '#2ca02c', 'lasso': '#d62728', 'dfs': '#9467bd'}
        os.makedirs(data_dir, exist_ok=True)
        self.results = {}

    def run_single_configuration(self, func_type: str, n_samples: int, base_params: Dict) -> Dict:
        params = base_params.copy()
        params.update({'func_type': func_type, 'n_samples': n_samples, 'selected_methods': self.methods})
        try:
            comp = LowDimensionComparator(**params)
            res = comp.run_multiple_comparisons(n_experiments=params['n_experiments'], verbose=False)
            stats = res['summary_stats']
            out = {'func_type': func_type, 'n_samples': n_samples, 'timestamp': datetime.now().isoformat()}
            for m in self.methods:
                if m in stats and stats[m]['tpr']['mean'] is not None:
                    out[m] = {k: {'mean': v['mean'], 'std': v['std']} for k, v in stats[m].items()}
                else:
                    out[m] = None
            return out
        except Exception as e:
            print(f"Error on {func_type}, n={n_samples}: {e}")
            return None

    def run_all_comparisons(self, n_experiments: int = 10, base_seed: int = 42, verbose: bool = True) -> None:
        base_params = {
            'd': 200, 'k': 5, 's': 5, 'rho': 0, 'n_experiments': n_experiments,
            'random_seed': base_seed, 'lassonet_hidden_dims': (100, 50), 'lassonet_standardize': True,
            'lasso_alpha': 0.1, 'lasso_standardize': True,
            'dfs_n_hidden1': 100, 'dfs_n_hidden2': 50, 'dfs_learning_rate': 0.1,
            'dfs_weight_decay_c': 1.0, 'dfs_step': 5
        }

        total = len(self.func_types) * len(self.sample_sizes)
        cur = 0
        for ft in self.func_types:
            self.results[ft] = {}
            for n in self.sample_sizes:
                cur += 1
                if verbose:
                    print(f"[{cur}/{total}] {ft}, n={n}")
                self.results[ft][n] = self.run_single_configuration(ft, n, base_params)

    def save_results(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = f"_{self.distribution_type}" if self.distribution_type == 't' else ""
        path = os.path.join(self.data_dir, f"sample_comparison{suffix}_{timestamp}.json")
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Results saved to {path}")
        return path

    def create_visualization(self) -> str:
        plt.style.use('default')
        cols = min(len(self.func_types), 5)
        fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 10))
        if len(self.func_types) == 1:
            axes = axes.reshape(2, 1)

        for col, ft in enumerate(self.func_types):
            ax_tpr, ax_fpr = axes[0, col], axes[1, col]
            for m in self.methods:
                tpr_vals, tpr_err, fpr_vals, fpr_err, xs = [], [], [], [], []
                for n in self.sample_sizes:
                    data = self.results.get(ft, {}).get(n, {}).get(m)
                    if data:
                        tpr_vals.append(data['tpr']['mean'])
                        tpr_err.append(data['tpr']['std'])
                        fpr_vals.append(data['fpr']['mean'])
                        fpr_err.append(data['fpr']['std'])
                        xs.append(n)

                if tpr_vals:
                    ax_tpr.errorbar(xs, tpr_vals, yerr=tpr_err, marker='o', label=self.method_names[m],
                                    color=self.method_colors[m], capsize=3)
                    ax_fpr.errorbar(xs, fpr_vals, yerr=fpr_err, marker='s', label=self.method_names[m],
                                    color=self.method_colors[m], capsize=3)

            ax_tpr.set_title(f"{ft} - TPR")
            ax_fpr.set_title(f"{ft} - FPR")
            ax_tpr.set_ylim(0, 1.1)
            ax_fpr.set_ylim(0, 0.05)
            ax_tpr.grid(True, alpha=0.3)
            ax_fpr.grid(True, alpha=0.3)
            if col == 0:
                ax_tpr.legend()

        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(self.data_dir, f"sample_comparison_plot_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {path}")
        return path


def main():
    distribution_type = 'gaussian'   # or 't'
    nu = 7
    selected_methods = ['stein', 'lasso', 'dfs', 'lassonet']    # e.g., ['stein', 'lasso', 'dfs', 'lassonet']
    selected_functions = ['quadratic', 'cos3', 'exp_poly', 'additive', 'multiplication']

    comparator = LowDimensionSampleSizeComparison(
        distribution_type=distribution_type,
        nu=nu,
        selected_methods=selected_methods,
        selected_functions=selected_functions
    )

    comparator.run_all_comparisons(n_experiments=99, base_seed=1, verbose=True)
    comparator.save_results()
    comparator.create_visualization()

if __name__ == "__main__":
    main()