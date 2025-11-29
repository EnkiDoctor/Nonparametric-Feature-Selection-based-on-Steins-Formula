import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys
from typing import Dict, List, Tuple
import warnings

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from screening_compare import run_gaussian_comparison

warnings.filterwarnings('ignore')

class RhoComparison:

    def __init__(self, data_dir: str = "screening_data", selected_methods: list = None, selected_functions: list = None):

        self.data_dir = data_dir
        self.distribution_type = 'gaussian'  # Use Gaussian distribution
        self.fixed_sample_size = 2000        # Fixed sample size
        self.fixed_dimension = 2000          # Fixed dimension
        self.rho_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]  # Correlation coefficients to evaluate
        
        self.available_functions = ['quadratic', 'cos', 'cos2', 'cos3', 'exp_poly']
        if selected_functions is None:
            self.func_types = self.available_functions.copy()
        else:
            valid_functions = [f for f in selected_functions if f in self.available_functions]
            if not valid_functions:
                raise ValueError(f"No valid function types were selected. Available types: {self.available_functions}")
            self.func_types = valid_functions
        
        self.available_methods = ['stein', 'stein_screening', 'lassonet', 'lasso', 'dfs']
        if selected_methods is None:
            self.methods = self.available_methods.copy()
        else:
            valid_methods = [m for m in selected_methods if m in self.available_methods]
            if not valid_methods:
                raise ValueError(f"No valid methods were selected. Available methods: {self.available_methods}")
            self.methods = valid_methods
        self.method_names = {
            'stein': 'Stein',
            'stein_screening': 'Stein+Screening',
            'lassonet': 'LassoNet',
            'lasso': 'Lasso',
            'dfs': 'DFS'
        }
        self.method_colors = {
            'stein': '#1f77b4',
            'stein_screening': '#ff7f0e', 
            'lassonet': '#2ca02c',
            'lasso': '#d62728',
            'dfs': '#9467bd'
        }
        
        os.makedirs(self.data_dir, exist_ok=True)
        
        self.results = {}
        
        print(f"Rho comparison initialized:")
        print(f"  Distribution: {self.distribution_type}")
        print(f"  Fixed sample size: {self.fixed_sample_size}")
        print(f"  Fixed dimension: {self.fixed_dimension}")
        print(f"  Rho values: {self.rho_values}")
        
        if len(self.func_types) == len(self.available_functions):
            print(f"  Function types: all functions ({self.func_types})")
        else:
            print(f"  Function types: selected subset ({self.func_types})")
        
        selected_method_names = [self.method_names[method] for method in self.methods]
        if len(self.methods) == len(self.available_methods):
            print(f"  Methods: all methods ({selected_method_names})")
        else:
            print(f"  Methods: selected subset ({selected_method_names})")
        print("-" * 50)
    
    def _print_filtered_results(self, summary_stats: dict, success_counts: dict, func_type: str, rho: float):
        """Print filtered summary for selected methods"""
        print(f"\n=== {func_type} function, rho={rho} results for selected methods ===")
        method_names = {
            'stein': 'Stein',
            'stein_screening': 'Stein+Screening',
            'lassonet': 'LassoNet',
            'lasso': 'Lasso',
            'dfs': 'DFS'
        }
        
        for method in self.methods:
            if method in summary_stats and summary_stats[method]['tpr']['mean'] is not None:
                stats = summary_stats[method]
                print(f"{method_names[method]}:")
                print(f"  TPR: {stats['tpr']['mean']:.4f} +/- {stats['tpr']['std']:.4f}")
                print(f"  FPR: {stats['fpr']['mean']:.4f} +/- {stats['fpr']['std']:.4f}")
                print(f"  Time: {stats['time']['mean']:.4f}s")
            else:
                print(f"{method_names[method]}: failed")
        print("-" * 50)
        
    def run_single_configuration(self, func_type: str, rho: float, 
                                base_params: Dict) -> Dict:
        """
        Run comparison for a single configuration
        
        Args:
            func_type: Function type
            rho: Correlation coefficient
            base_params: Base experiment parameters
            
        Returns:
            Dictionary containing the configuration results
        """
        print(f"Running {func_type} with rho={rho}")
        
        params = base_params.copy()
        params['func_type'] = func_type
        params['rho'] = rho
        
        try:
            results = run_gaussian_comparison(**params)
            
            summary_stats = results['summary_stats']
            success_counts = results['success_counts']
            
            filtered_success_counts = {method: success_counts.get(method, 0) for method in self.methods}
            
            self._print_filtered_results(summary_stats, filtered_success_counts, func_type, rho)
            
            config_results = {
                'func_type': func_type,
                'rho': rho,
                'success_counts': filtered_success_counts,
                'timestamp': datetime.now().isoformat()
            }
            
            for method in self.methods:
                if method in summary_stats:
                    method_stats = summary_stats[method]
                    config_results[method] = {
                        'tpr': {
                            'mean': method_stats['tpr']['mean'],
                            'std': method_stats['tpr']['std']
                        },
                        'fpr': {
                            'mean': method_stats['fpr']['mean'], 
                            'std': method_stats['fpr']['std']
                        },
                        'time': {
                            'mean': method_stats['time']['mean'],
                            'std': method_stats['time']['std']
                        }
                    } if method_stats['tpr']['mean'] is not None else None
                else:
                    config_results[method] = None
            
            return config_results
            
        except Exception as e:
            print(f"Error in {func_type} with rho={rho}: {e}")
            return None
    
    def run_all_comparisons(self, n_experiments: int = 10, 
                           base_seed: int = 42, verbose: bool = True) -> None:
        """
        Run all comparisons across rho values and function types
        
        Args:
            n_experiments: Experiments per configuration
            base_seed: Base random seed
            verbose: Whether to print progress
        """
        print("="*80)
        print("RHO COMPARISON ACROSS FUNCTION TYPES (Gaussian distribution)")
        print("="*80)
        print(f"Distribution type: {self.distribution_type}")
        print(f"Fixed sample size: {self.fixed_sample_size}")
        print(f"Fixed dimension: {self.fixed_dimension}")
        print(f"Rho values: {self.rho_values}")
        if len(self.func_types) == len(self.available_functions):
            print(f"Function types: all functions ({self.func_types})")
        else:
            print(f"Function types: selected subset ({self.func_types})")
        
        selected_method_names = [self.method_names[method] for method in self.methods]
        if len(self.methods) == len(self.available_methods):
            print(f"Methods: all methods ({selected_method_names})")
        else:
            print(f"Methods: selected subset ({selected_method_names})")
        print(f"Experiments per configuration: {n_experiments}")
        print("="*80)
        
        base_params = {
            'n_samples': self.fixed_sample_size,
            'd': self.fixed_dimension,
            'k': 5,
            's': 5,
            'distribution_type': self.distribution_type,
            'nu': 5,
            'n_experiments': n_experiments,
            'random_seed': base_seed,
            'use_gpu': True,
            'device_id': 0,
            
            'lassonet_hidden_dims': (100, 50),
            'lassonet_standardize': True,
            'lassonet_lambda_start': 100,
            'lassonet_path_multiplier': 1.25,
            'lassonet_stable': False,
            
            'lasso_alpha': 0.1,
            'lasso_standardize': True,
            
            'dfs_n_hidden1': 100,
            'dfs_n_hidden2': 50,
            'dfs_learning_rate': 0.1,
            'dfs_weight_decay_c': 1.0,
            'dfs_step': 5,
            
            'screening_m': 19,
            'screening_delta': 0.9
        }
        
        for func_type in self.func_types:
            self.results[func_type] = {}
        
        total_configs = len(self.func_types) * len(self.rho_values)
        current_config = 0
        
        for func_type in self.func_types:
            for rho in self.rho_values:
                current_config += 1
                if verbose:
                    print(f"\n[{current_config}/{total_configs}] Processing {func_type} with rho={rho}")
                
                config_results = self.run_single_configuration(
                    func_type, rho, base_params
                )
                
                if config_results is not None:
                    self.results[func_type][rho] = config_results
                else:
                    print(f"Failed: {func_type} with rho={rho}")
        
        print(f"\n{'='*80}")
        print("ALL COMPARISONS COMPLETED!")
        print(f"{'='*80}")
    
    def save_results(self) -> str:
        """
        Save results to JSON file
        
        Returns:
            File path of saved results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"rho_comparison_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def create_visualization(self) -> str:
        """
        Create visualization comparing TPR and FPR across rho values
        
        Returns:
            File path of saved plot
        """
        plt.style.use('default')
        sns.set_palette("husl")
        
        num_functions = len(self.func_types)
        if num_functions <= 5:
            cols = num_functions
            fig_width = 5 * cols
        else:
            cols = 5
            fig_width = 25
        
        fig, axes = plt.subplots(2, cols, figsize=(fig_width, 10))
        
        if num_functions == 1:
            axes = axes.reshape(2, 1)
        
        method_count = len(self.methods)
        selected_method_names = [self.method_names[method] for method in self.methods]
        func_desc = "all functions" if len(self.func_types) == len(self.available_functions) else f"selected functions ({len(self.func_types)})"
        method_desc = "all methods" if len(self.methods) == len(self.available_methods) else f"selected methods ({method_count})"
        fig.suptitle(f'TPR and FPR Comparison Across Rho Values ({method_desc}: {", ".join(selected_method_names)}, {func_desc}, n={self.fixed_sample_size}, d={self.fixed_dimension}, Gaussian)', 
                     fontsize=16, fontweight='bold')
        
        func_titles = {
            'quadratic': 'Quadratic',
            'cos': 'Cosine',
            'cos2': 'Cosine-2',
            'cos3': 'Cosine-3', 
            'exp_poly': 'Exp-Poly'
        }
        
        # Plot TPR (top row) and FPR (bottom row) for each function type
        for col, func_type in enumerate(self.func_types):
            ax_tpr = axes[0, col]
            ax_fpr = axes[1, col]
            
            if func_type in self.results:
                for method in self.methods:
                    tpr_values = []
                    tpr_stds = []
                    fpr_values = []
                    fpr_stds = []
                    valid_rho_values = []
                    
                    for rho in self.rho_values:
                        if (rho in self.results[func_type] and 
                            method in self.results[func_type][rho] and
                            self.results[func_type][rho][method] is not None):
                            
                            method_data = self.results[func_type][rho][method]
                            tpr_values.append(method_data['tpr']['mean'])
                            tpr_stds.append(method_data['tpr']['std'])
                            fpr_values.append(method_data['fpr']['mean'])
                            fpr_stds.append(method_data['fpr']['std'])
                            valid_rho_values.append(rho)
                    
                    if tpr_values:
                        ax_tpr.errorbar(valid_rho_values, tpr_values, yerr=tpr_stds,
                                      marker='o', linestyle='-', linewidth=2, markersize=4,
                                      color=self.method_colors[method], 
                                      label=self.method_names[method],
                                      capsize=3, capthick=1)
                    
                    if fpr_values:
                        ax_fpr.errorbar(valid_rho_values, fpr_values, yerr=fpr_stds,
                                      marker='s', linestyle='-', linewidth=2, markersize=4,
                                      color=self.method_colors[method],
                                      label=self.method_names[method],
                                      capsize=3, capthick=1)
            
            ax_tpr.set_title(f'{func_titles[func_type]} - TPR', fontsize=12, fontweight='bold')
            ax_tpr.set_xlabel('Correlation coefficient (rho)')
            if col == 0:
                ax_tpr.set_ylabel('True Positive Rate (TPR)')
            ax_tpr.grid(True, alpha=0.3)
            ax_tpr.set_ylim(0, 1.1)
            ax_tpr.set_xlim(-0.05, 0.55)
            if col == 0:
                ax_tpr.legend(loc='lower right', fontsize=9)
            
            ax_fpr.set_title(f'{func_titles[func_type]} - FPR', fontsize=12, fontweight='bold')
            ax_fpr.set_xlabel('Correlation coefficient (rho)')
            if col == 0:
                ax_fpr.set_ylabel('False Positive Rate (FPR)')
            ax_fpr.grid(True, alpha=0.3)
            ax_fpr.set_ylim(0, 0.05)
            ax_fpr.set_xlim(-0.05, 0.55)
            if col == 0:
                ax_fpr.legend(loc='upper right', fontsize=9)
        
        if num_functions < cols:
            for col in range(num_functions, cols):
                axes[0, col].set_visible(False)
                axes[1, col].set_visible(False)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f"rho_comparison_{timestamp}.png"
        plot_filepath = os.path.join(self.data_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to: {plot_filepath}")
        plt.close()
        
        return plot_filepath
    
    def create_time_comparison_plot(self) -> str:
        """
        Create separate plot for training time across rho values
        
        Returns:
            File path of saved plot
        """
        plt.style.use('default')
        sns.set_palette("husl")
        
        num_functions = len(self.func_types)
        if num_functions <= 5:
            cols = num_functions
            fig_width = 5 * cols
        else:
            cols = 5
            fig_width = 25
        
        fig, axes = plt.subplots(1, cols, figsize=(fig_width, 5))
        
        if num_functions == 1:
            axes = [axes]
        
        method_count = len(self.methods)
        selected_method_names = [self.method_names[method] for method in self.methods]
        func_desc = "all functions" if len(self.func_types) == len(self.available_functions) else f"selected functions ({len(self.func_types)})"
        method_desc = "all methods" if len(self.methods) == len(self.available_methods) else f"selected methods ({method_count})"
        fig.suptitle(f'Training Time Comparison Across Rho Values ({method_desc}: {", ".join(selected_method_names)}, {func_desc}, n={self.fixed_sample_size}, d={self.fixed_dimension}, Gaussian)', 
                     fontsize=16, fontweight='bold')
        
        func_titles = {
            'quadratic': 'Quadratic',
            'cos': 'Cosine',
            'cos2': 'Cosine-2',
            'cos3': 'Cosine-3', 
            'exp_poly': 'Exp-Poly'
        }
        
        for col, func_type in enumerate(self.func_types):
            ax = axes[col]
            
            if func_type in self.results:
                for method in self.methods:
                    time_values = []
                    time_stds = []
                    valid_rho_values = []
                    
                    for rho in self.rho_values:
                        if (rho in self.results[func_type] and 
                            method in self.results[func_type][rho] and
                            self.results[func_type][rho][method] is not None):
                            
                            method_data = self.results[func_type][rho][method]
                            time_values.append(method_data['time']['mean'])
                            time_stds.append(method_data['time']['std'])
                            valid_rho_values.append(rho)
                    
                    if time_values:
                        ax.errorbar(valid_rho_values, time_values, yerr=time_stds,
                                  marker='o', linestyle='-', linewidth=2, markersize=4,
                                  color=self.method_colors[method], 
                                  label=self.method_names[method],
                                  capsize=3, capthick=1)
            
            ax.set_title(f'{func_titles[func_type]} - Training Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Correlation coefficient (rho)')
            if col == 0:
                ax.set_ylabel('Training Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')
            ax.set_xlim(-0.05, 0.55)
            if col == 0:
                ax.legend(loc='upper left', fontsize=9)
        
        if num_functions < cols:
            for col in range(num_functions, cols):
                axes[col].set_visible(False)
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f"rho_time_comparison_{timestamp}.png"
        plot_filepath = os.path.join(self.data_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        print(f"Time comparison plot saved to: {plot_filepath}")
        plt.close()
        
        return plot_filepath
    
    def generate_summary_report(self) -> None:
        """
        Generate a summary report of the rho comparison
        """
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}")
        
        total_configs = len(self.func_types) * len(self.rho_values)
        successful_configs = 0
        
        for func_type in self.func_types:
            for rho in self.rho_values:
                if (func_type in self.results and 
                    rho in self.results[func_type]):
                    successful_configs += 1
        
        print("Distribution type: Gaussian")
        print(f"Successful configurations: {successful_configs}/{total_configs}")
        print(f"Fixed sample size: {self.fixed_sample_size}")
        print(f"Fixed dimension: {self.fixed_dimension}")
        print(f"Tested rho values: {self.rho_values}")
        
        selected_method_names = [self.method_names[method] for method in self.methods]
        if len(self.methods) == len(self.available_methods):
            print(f"Methods evaluated: all methods ({selected_method_names})")
        else:
            print(f"Methods evaluated: selected subset ({selected_method_names})")
        
        if len(self.func_types) == len(self.available_functions):
            print(f"Functions evaluated: all functions ({self.func_types})")
        else:
            print(f"Functions evaluated: selected subset ({self.func_types})")
        
        print(f"\nSuccess rate for selected methods:")
        for method in self.methods:
            success_count = 0
            for func_type in self.func_types:
                for rho in self.rho_values:
                    if (func_type in self.results and 
                        rho in self.results[func_type] and
                        method in self.results[func_type][rho] and
                        self.results[func_type][rho][method] is not None):
                        success_count += 1
            
            success_rate = success_count / total_configs * 100
            print(f"  {self.method_names[method]}: {success_count}/{total_configs} ({success_rate:.1f}%)")
        
        print(f"\nBest Method by Function Type (by average TPR):")
        for func_type in self.func_types:
            method_tpr_avg = {}
            
            for method in self.methods:
                tpr_values = []
                for rho in self.rho_values:
                    if (func_type in self.results and 
                        rho in self.results[func_type] and
                        method in self.results[func_type][rho] and
                        self.results[func_type][rho][method] is not None):
                        tpr_values.append(self.results[func_type][rho][method]['tpr']['mean'])
                
                if tpr_values:
                    method_tpr_avg[method] = np.mean(tpr_values)
            
            if method_tpr_avg:
                best_method = max(method_tpr_avg.items(), key=lambda x: x[1])
                print(f"  {func_type.upper()}: {self.method_names[best_method[0]]} (TPR={best_method[1]:.3f})")
        
        print(f"\nCorrelation Impact Analysis:")
        for method in self.methods:
            print(f"\n  {self.method_names[method]}:")
            for func_type in self.func_types:
                tpr_values = []
                valid_rhos = []
                
                for rho in self.rho_values:
                    if (func_type in self.results and 
                        rho in self.results[func_type] and
                        method in self.results[func_type][rho] and
                        self.results[func_type][rho][method] is not None):
                        tpr_values.append(self.results[func_type][rho][method]['tpr']['mean'])
                        valid_rhos.append(rho)
                
                if len(tpr_values) >= 2:
                    slope = np.polyfit(valid_rhos, tpr_values, 1)[0]
                    correlation_effect = "negative" if slope < -0.01 else "positive" if slope > 0.01 else "neutral"
                    print(f"    {func_type}: {correlation_effect} correlation effect (slope={slope:.4f})")
        
        print(f"\nOptimal Rho Analysis (by highest average TPR across all methods):")
        rho_performance = {}
        for rho in self.rho_values:
            total_tpr = 0
            count = 0
            for func_type in self.func_types:
                for method in self.methods:
                    if (func_type in self.results and 
                        rho in self.results[func_type] and
                        method in self.results[func_type][rho] and
                        self.results[func_type][rho][method] is not None):
                        total_tpr += self.results[func_type][rho][method]['tpr']['mean']
                        count += 1
            
            if count > 0:
                rho_performance[rho] = total_tpr / count
        
        if rho_performance:
            sorted_rho = sorted(rho_performance.items(), key=lambda x: x[1], reverse=True)
            print(f"  Ranking (rho: average TPR):")
            for i, (rho, avg_tpr) in enumerate(sorted_rho, 1):
                print(f"    {i}. rho={rho}: {avg_tpr:.3f}")
        
        print(f"{'='*80}")


def main():
    """
    Main entry point for rho comparison experiments
    """
    print("Starting Rho Comparison Experiment")
    print("="*50)
    selected_functions = ['quadratic', 'cos3', 'exp_poly', 'additive', 'multiplication']  
    
    method_names = {
        'stein': 'Stein',
        'stein_screening': 'Stein+Screening',
        'lassonet': 'LassoNet',
        'lasso': 'Lasso',
        'dfs': 'DFS'
    }
    function_names = {
        'quadratic': 'Quadratic',
        'cos': 'Cosine',
        'cos2': 'Cosine-2',
        'cos3': 'Cosine-3',
        'exp_poly': 'Exp-Poly'
    }
    
    print(f"Distribution type: gaussian")
    print(f"Fixed sample size: 2000")
    print(f"Fixed dimension: 2000")
    print(f"Rho values: [0, 0.1, 0.2, 0.3, 0.4, 0.5]")
    
    if selected_methods is None:
        print("Methods compared: all methods (Stein, Stein+Screening, LassoNet, Lasso, DFS)")
    else:
        selected_names = [method_names.get(m, m) for m in selected_methods]
        print(f"Methods compared: {', '.join(selected_names)}")
    
    if selected_functions is None:
        print("Function types: all functions (Quadratic, Cosine, Cosine-2, Cosine-3, Exp-Poly)")
    else:
        selected_func_names = [function_names.get(f, f) for f in selected_functions]
        print(f"Function types: {', '.join(selected_func_names)}")
    
    print("="*50)
    
    comparator = RhoComparison(
        selected_methods=selected_methods,
        selected_functions=selected_functions
    )
    
    comparator.run_all_comparisons(
        n_experiments=1,
        base_seed=1,
        verbose=True
    )
    
    results_file = comparator.save_results()
    plot_file = comparator.create_visualization()
    
    print(f"\n{'='*50}")
    print("EXPERIMENT COMPLETED!")
    print(f"Results saved to: {results_file}")
    print(f"Main visualization saved to: {plot_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
