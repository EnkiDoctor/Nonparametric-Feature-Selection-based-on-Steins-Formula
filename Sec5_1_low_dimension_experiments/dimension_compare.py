import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
from typing import Dict
import warnings
import time

# Add current and parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
for p in [current_dir, parent_dir]:
    if p not in sys.path:
        sys.path.append(p)

# Add custom module paths
for sub in ["deep_feature_selection", "prediction", "screening"]:
    path = os.path.join(parent_dir, sub)
    if path not in sys.path:
        sys.path.append(path)

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
    from sample_compare import (
        create_sigma_matrix, generate_samples_rho, quadratic_k, cos_func,
        cos_func2, cos_func3, exp_poly, func_choose, calculate_selection_metrics,
        SteinSelector, LassoSelector, LassoNetSelector, DFSSelector, LowDimensionComparator
    )
    SAMPLE_COMPARE_AVAILABLE = True
except ImportError:
    print("Warning: sample_compare module not available")
    SAMPLE_COMPARE_AVAILABLE = False


class LowDimensionComparison:
    """
    Compare feature selection methods under varying input dimension p
    (fixed sample size n = 2000).
    """

    def __init__(self, data_dir: str = "low_dimension_data", distribution_type: str = 'gaussian',
                 nu: float = 5, selected_methods: list = None, selected_functions: list = None):
        self.data_dir = data_dir
        self.distribution_type = distribution_type
        self.nu = nu
        self.dimensions = [200, 400, 600, 800, 1000]      # tested input dimensions
        self.n_samples = 2000                             # fixed sample size

        # Function types
        self.available_functions = ['quadratic', 'cos', 'cos2', 'cos3',
                                    'exp_poly', 'additive', 'multiplication']
        self.func_types = (self.available_functions if selected_functions is None
                          else [f for f in selected_functions if f in self.available_functions])

        # Methods
        self.available_methods = ['stein', 'lassonet', 'lasso', 'dfs']
        self.methods = (self.available_methods if selected_methods is None
                       else [m for m in selected_methods if m in self.available_methods])

        self.method_names = {'stein': 'Stein', 'lassonet': 'LassoNet',
                             'lasso': 'Lasso', 'dfs': 'DFS'}
        self.method_colors = {'stein': '#1f77b4', 'lassonet': '#2ca02c',
                              'lasso': '#d62728', 'dfs': '#9467bd'}

        os.makedirs(data_dir, exist_ok=True)
        self.results = {}

    def run_single_configuration(self, func_type: str, d: int, base_params: Dict) -> Dict:
        print(f"Running {func_type}, dimension = {d}")

        params = base_params.copy()
        params.update({'func_type': func_type, 'd': d, 'selected_methods': self.methods})

        try:
            comparator = LowDimensionComparator(**params)
            out = comparator.run_multiple_comparisons(
                n_experiments=params['n_experiments'], verbose=False
            )
            stats = out['summary_stats']

            result = {
                'func_type': func_type,
                'd': d,
                'timestamp': datetime.now().isoformat()
            }

            for method in self.methods:
                if method in stats and stats[method]['tpr']['mean'] is not None:
                    mstat = stats[method]
                    result[method] = {
                        'tpr': {'mean': mstat['tpr']['mean'], 'std': mstat['tpr']['std']},
                        'fpr': {'mean': mstat['fpr']['mean'], 'std': mstat['fpr']['std']},
                        'time': {'mean': mstat['time']['mean'], 'std': mstat['time']['std']}
                    }
                else:
                    result[method] = None
            return result

        except Exception as e:
            print(f"Error on {func_type}, d={d}: {e}")
            return None

    def run_all_comparisons(self, n_experiments: int = 10,
                           base_seed: int = 42, verbose: bool = True) -> None:
        print("=" * 80)
        print(f"Dimension comparison (n = {self.n_samples})")
        print("=" * 80)

        base_params = {
            'n_samples': self.n_samples,
            'k': 5,
            's': 5,
            'rho': 0,
            'n_experiments': n_experiments,
            'random_seed': base_seed,

            # LassoNet settings
            'lassonet_hidden_dims': (100, 50),
            'lassonet_standardize': True,
            'lassonet_lambda_start': 50,
            'lassonet_path_multiplier': 1.25,
            'lassonet_stable': False,

            # Lasso settings
            'lasso_alpha': 0.1,
            'lasso_standardize': True,

            # DFS settings
            'dfs_n_hidden1': 100,
            'dfs_n_hidden2': 50,
            'dfs_learning_rate': 0.1,
            'dfs_weight_decay_c': 1.0,
            'dfs_step': 5,
        }

        total = len(self.func_types) * len(self.dimensions)
        cur = 0
        for ft in self.func_types:
            self.results[ft] = {}
            for d in self.dimensions:
                cur += 1
                if verbose:
                    print(f"[{cur}/{total}] {ft}, p = {d}")
                self.results[ft][d] = self.run_single_configuration(ft, d, base_params)

        print("\nAll comparisons finished!")

    def save_results(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = f"_{self.distribution_type}" if self.distribution_type == 't' else ""
        path = os.path.join(self.data_dir,
                            f"dimension_comparison{suffix}_{timestamp}.json")
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

        case_map = {'quadratic': 1, 'cos': 2, 'cos2': 3, 'cos3': 4,
                    'exp_poly': 5, 'additive': 6, 'multiplication': 7}

        for col, ft in enumerate(self.func_types):
            ax_tpr, ax_fpr = axes[0, col], axes[1, col]

            for method in self.methods:
                tpr_vals, tpr_err = [], []
                fpr_vals, fpr_err = [], []
                dims = []

                for d in self.dimensions:
                    data = self.results.get(ft, {}).get(d, {}).get(method)
                    if data:
                        tpr_vals.append(data['tpr']['mean'])
                        tpr_err.append(data['tpr']['std'])
                        fpr_vals.append(data['fpr']['mean'])
                        fpr_err.append(data['fpr']['std'])
                        dims.append(d)

                if tpr_vals:
                    ax_tpr.errorbar(dims, tpr_vals, yerr=tpr_err,
                                    marker='o', label=self.method_names[method],
                                    color=self.method_colors[method], capsize=3)
                    ax_fpr.errorbar(dims, fpr_vals, yerr=fpr_err,
                                    marker='s', label=self.method_names[method],
                                    color=self.method_colors[method], capsize=3)

            case = case_map[ft]
            ax_tpr.set_title(f'TPR vs p: Case {case}')
            ax_fpr.set_title(f'FPR vs p: Case {case}')
            ax_tpr.set_xlabel('Input Dimension (p)')
            ax_fpr.set_xlabel('Input Dimension (p)')
            if col == 0:
                ax_tpr.set_ylabel('True Positive Rate (TPR)')
                ax_fpr.set_ylabel('False Positive Rate (FPR)')
                ax_tpr.legend(loc='lower right', fontsize=9)

            ax_tpr.set_ylim(0, 1.1)
            ax_fpr.set_ylim(0, 0.05)
            ax_tpr.grid(True, alpha=0.3)
            ax_fpr.grid(True, alpha=0.3)

        # Hide unused subplots
        if len(self.func_types) < cols:
            for c in range(len(self.func_types), cols):
                axes[0, c].set_visible(False)
                axes[1, c].set_visible(False)

        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = f"_{self.distribution_type}" if self.distribution_type == 't' else ""
        path = os.path.join(self.data_dir, f"dimension_comparison{suffix}_{timestamp}.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {path}")
        return path


def main():
    distribution_type = 'gaussian'          # 'gaussian' or 't'
    nu = 5

    # Select methods and functions (set to None for all)
    selected_methods = ['stein', 'dfs', 'lassonet', 'lasso']   # 'stein', 'lasso', 'dfs', 'lassonet'
    selected_functions = ['quadratic', 'cos3', 'exp_poly', 'additive', 'multiplication']

    comparator = LowDimensionComparison(
        distribution_type=distribution_type,
        nu=nu,
        selected_methods=selected_methods,
        selected_functions=selected_functions
    )

    comparator.run_all_comparisons(n_experiments=99, base_seed=1, verbose=True)
    results_file = comparator.save_results()
    plot_file = comparator.create_visualization()

    print("\n" + "=" * 50)
    print("Experiment finished!")
    print(f"Results: {results_file}")
    print(f"Plot   : {plot_file}")
    print("=" * 50)


if __name__ == "__main__":
    main()