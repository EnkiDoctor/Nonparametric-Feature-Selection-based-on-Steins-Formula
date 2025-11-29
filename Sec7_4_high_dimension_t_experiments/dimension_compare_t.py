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
parent_dir = os.path.dirname(current_dir)
screening_dir = os.path.join(parent_dir, "Sec7_2_high_dimension_experiments")
if screening_dir not in sys.path:
    sys.path.append(screening_dir)

from screening_compare import run_gaussian_comparison
warnings.filterwarnings('ignore')

class DimensionComparison:
    """
    Dimension comparison for different function types and feature selection methods
    Fixed sample size, varying dimension d
    """
    
    def __init__(self, data_dir: str = "screening_data", distribution_type: str = 'gaussian', nu: float = 5, 
                 selected_dimensions: List[int] = None, selected_functions: list = None):

        self.data_dir = data_dir
        self.distribution_type = distribution_type
        self.nu = nu
        self.all_dimensions = [200, 500, 800, 1000, 2000, 3000, 4000, 5000]
        self.dimensions = selected_dimensions if selected_dimensions is not None else self.all_dimensions
        self.fixed_sample_size = 2000  
        
        self.available_functions = ['quadratic', 'cos', 'cos2', 'cos3', 'exp_poly', 'additive', 'multiplication']
        if selected_functions is None:
            self.func_types = self.available_functions.copy()
        else:
            valid_functions = [f for f in selected_functions if f in self.available_functions]
            if not valid_functions:
                raise ValueError(f"No valid function types were selected. Available types: {self.available_functions}")
            self.func_types = valid_functions
        
        self.methods = ['stein', 'stein_screening', 'lassonet', 'lasso', 'dfs']
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
        
        self.screening_params = {
            200: {'m': 2, 'delta': 0.9},
            500: {'m': 5, 'delta': 0.9},
            800: {'m': 8, 'delta': 0.9},
            1000: {'m': 10, 'delta': 0.9},
            2000: {'m': 17, 'delta': 0.9},
            3000: {'m': 20, 'delta': 0.9},
            4000: {'m': 25, 'delta': 0.9},
            5000: {'m': 30, 'delta': 0.9}
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Results storage
        self.results = {}
        
        if selected_dimensions is not None:
            print(f"Selected dimensions: {self.dimensions}")
        else:
            print(f"Running all available dimensions: {self.dimensions}")
        
        if selected_functions is not None:
            print(f"Selected functions: {self.func_types}")
        else:
            print(f"Running all functions: {self.func_types}")
        
        print("Available dimensions and screening parameters:")
        for d in self.all_dimensions:
            params = self.get_screening_params(d)
            print(f"  d={d}: m={params['m']}, delta={params['delta']}")
        print("-" * 50)
    
    @classmethod
    def get_available_dimensions(cls) -> List[int]:
        return [200, 500, 800, 1000, 2000, 3000, 4000, 5000]
    
    @classmethod
    def get_dimension_info(cls) -> Dict[int, Dict[str, float]]:
        return {
            200: {'m': 2, 'delta': 0.9},
            500: {'m': 5, 'delta': 0.9},
            800: {'m': 8, 'delta': 0.9},
            1000: {'m': 10, 'delta': 0.9},
            2000: {'m': 17, 'delta': 0.9},
            3000: {'m': 20, 'delta': 0.9},
            4000: {'m': 25, 'delta': 0.9},
            5000: {'m': 30, 'delta': 0.9}
        }
        
    def get_screening_params(self, d: int) -> Dict[str, float]:
        if d in self.screening_params:
            return self.screening_params[d]
        else:
            # Interpolate or use nearest neighbor for unlisted dimensions
            nearest_d = min(self.screening_params.keys(), key=lambda x: abs(x - d))
            return self.screening_params[nearest_d]
    
    def run_single_configuration(self, func_type: str, d: int, 
                                base_params: Dict) -> Dict:
        print(f"Running {func_type} with d={d}")
        
        params = base_params.copy()
        params['func_type'] = func_type
        params['d'] = d

        screening_params = self.get_screening_params(d)
        params['screening_m'] = screening_params['m']
        params['screening_delta'] = screening_params['delta']
        
        print(f"  Screening params: m={screening_params['m']}, delta={screening_params['delta']}")
        
        try:
            results = run_gaussian_comparison(**params)
            
            summary_stats = results['summary_stats']
            success_counts = results['success_counts']
            
            config_results = {
                'func_type': func_type,
                'd': d,
                'screening_m': screening_params['m'],
                'screening_delta': screening_params['delta'],
                'success_counts': success_counts,
                'timestamp': datetime.now().isoformat()
            }
            
            for method in self.methods:
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
            
            return config_results
            
        except Exception as e:
            print(f"Error in {func_type} with d={d}: {e}")
            return None
    
    def run_all_comparisons(self, n_experiments: int = 10, 
                           base_seed: int = 42, verbose: bool = True) -> None:

        dist_name = "t-distribution" if self.distribution_type == 't' else "Gaussian"
        print("="*80)
        print(f"Dimension comparison experiment ({dist_name}, fixed sample size={self.fixed_sample_size})")
        print("="*80)
        print(f"Distribution type: {self.distribution_type}")
        if self.distribution_type == 't':
            print(f"Degrees of freedom: {self.nu}")
        print(f"Fixed sample size: {self.fixed_sample_size}")
        print(f"Test dimensions: {self.dimensions}")
        if len(self.func_types) == len(self.available_functions):
            print(f"Function types: all functions ({self.func_types})")
        else:
            print(f"Function types: selected subset ({self.func_types})")
        print(f"Methods compared: {list(self.method_names.values())}")
        print(f"Experiments per configuration: {n_experiments}")
        print(f"Screening params: {self.screening_params}")
        print("="*80)
        
        base_params = {
            'n_samples': self.fixed_sample_size,  
            'k': 5,       
            's': 5,            
            'rho': 0,         
            'distribution_type': self.distribution_type,  # Distribution type
            'nu': self.nu,      # t-distribution degrees of freedom
            'n_experiments': n_experiments,
            'random_seed': base_seed,
            'use_gpu': True,
            'device_id': 0,
            
            # LassoNet parameters
            'lassonet_hidden_dims': (100, 50),
            'lassonet_standardize': True,
            'lassonet_lambda_start': 100,
            'lassonet_path_multiplier': 1.25,
            'lassonet_stable': False,
            
            # Lasso parameters
            'lasso_alpha': 0.1,
            'lasso_standardize': True,
            
            # DFS parameters
            'dfs_n_hidden1': 100,
            'dfs_n_hidden2': 50,
            'dfs_learning_rate': 0.1,
            'dfs_weight_decay_c': 1.0,
            'dfs_step': 4,
            
            # Stein screening parameters (will be updated for each dimension)
            'screening_m': 15,
            'screening_delta': 0.9
        }
        
        for func_type in self.func_types:
            self.results[func_type] = {}
        
        total_configs = len(self.func_types) * len(self.dimensions)
        current_config = 0
        
        for func_type in self.func_types:
            for d in self.dimensions:
                current_config += 1
                if verbose:
                    print(f"\n[{current_config}/{total_configs}] Processing {func_type} with d={d}")
                

                config_results = self.run_single_configuration(
                    func_type, d, base_params
                )
                
                if config_results is not None:
                    self.results[func_type][d] = config_results
                else:
                    print(f"Failed: {func_type} with d={d}")
        
        print(f"\n{'='*80}")
        print("ALL COMPARISONS COMPLETED!")
        print(f"{'='*80}")
    
    def save_results(self) -> str:
        """
        Save results to JSON file
        
        Returns:
            Filepath of saved results
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dist_suffix = f"_{self.distribution_type}" if self.distribution_type == 't' else ""
        
        if len(self.func_types) < len(self.available_functions):
            func_suffix = "_" + "_".join(self.func_types)
        else:
            func_suffix = ""
        
        filename = f"dimension_comparison{dist_suffix}{func_suffix}_{timestamp}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {filepath}")
        return filepath
    
    def create_visualization(self) -> str:
        plt.style.use('default')
        sns.set_palette("husl")

        num_funcs = len(self.func_types)
        ncols = min(num_funcs, 7)
        
        fig, axes = plt.subplots(2, ncols, figsize=(5*ncols, 10), squeeze=False)
        dist_name = "t-distribution" if self.distribution_type == 't' else "Gaussian"
        
        if len(self.func_types) == len(self.available_functions):
            func_desc = "all functions"
        else:
            func_desc = f"selected functions ({len(self.func_types)})"
        
        fig.suptitle(f'TPR and FPR vs dimension ({func_desc}, n={self.fixed_sample_size}, {dist_name})', 
                     fontsize=16, fontweight='bold')
        func_titles = {
            'quadratic': 'Quadratic',
            'cos': 'Cosine',
            'cos2': 'Cosine-2',
            'cos3': 'Cosine-3', 
            'exp_poly': 'Exp-Poly',
            'additive': 'Additive',
            'multiplication': 'Multiplication'
        }
        
        for col, func_type in enumerate(self.func_types):
            ax_tpr = axes[0, col]
            ax_fpr = axes[1, col]
            if func_type in self.results:
                for method in self.methods:
                    tpr_values = []
                    tpr_stds = []
                    fpr_values = []
                    fpr_stds = []
                    valid_dimensions = []
                    
                    for d in self.dimensions:
                        if (d in self.results[func_type] and 
                            method in self.results[func_type][d] and
                            self.results[func_type][d][method] is not None):
                            
                            method_data = self.results[func_type][d][method]
                            tpr_values.append(method_data['tpr']['mean'])
                            tpr_stds.append(method_data['tpr']['std'])
                            fpr_values.append(method_data['fpr']['mean'])
                            fpr_stds.append(method_data['fpr']['std'])
                            valid_dimensions.append(d)

                    if tpr_values:
                        ax_tpr.errorbar(valid_dimensions, tpr_values, yerr=tpr_stds,
                                      marker='o', linestyle='-', linewidth=2, markersize=4,
                                      color=self.method_colors[method], 
                                      label=self.method_names[method],
                                      capsize=3, capthick=1)
                    if fpr_values:
                        ax_fpr.errorbar(valid_dimensions, fpr_values, yerr=fpr_stds,
                                      marker='s', linestyle='-', linewidth=2, markersize=4,
                                      color=self.method_colors[method],
                                      label=self.method_names[method],
                                      capsize=3, capthick=1)
            
            ax_tpr.set_title(f'{func_titles[func_type]} - TPR', fontsize=12, fontweight='bold')
            ax_tpr.set_xlabel('Dimension (d)')
            if col == 0:
                ax_tpr.set_ylabel('True Positive Rate (TPR)')
            ax_tpr.grid(True, alpha=0.3)
            ax_tpr.set_ylim(0, 1.1)
            ax_tpr.set_xlim(min(self.dimensions)*0.95, max(self.dimensions)*1.05)
            if col == 0:
                ax_tpr.legend(loc='lower right', fontsize=9)
            
            ax_fpr.set_title(f'{func_titles[func_type]} - FPR', fontsize=12, fontweight='bold')
            ax_fpr.set_xlabel('Dimension (d)')
            if col == 0:
                ax_fpr.set_ylabel('False Positive Rate (FPR)')
            ax_fpr.grid(True, alpha=0.3)
            ax_fpr.set_ylim(0, 0.05) 
            ax_fpr.set_xlim(min(self.dimensions)*0.95, max(self.dimensions)*1.05)
            if col == 0:
                ax_fpr.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dist_suffix = f"_{self.distribution_type}" if self.distribution_type == 't' else ""
        if len(self.func_types) < len(self.available_functions):
            func_suffix = "_" + "_".join(self.func_types)
        else:
            func_suffix = ""
        
        plot_filename = f"dimension_comparison{dist_suffix}{func_suffix}_{timestamp}.png"
        plot_filepath = os.path.join(self.data_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to: {plot_filepath}")
        plt.close()
        
        return plot_filepath
    
    def create_time_comparison_plot(self) -> str:
        plt.style.use('default')
        sns.set_palette("husl")
        num_funcs = len(self.func_types)
        ncols = min(num_funcs, 7)
        
        fig, axes = plt.subplots(1, ncols, figsize=(5*ncols, 5))
        
        if num_funcs == 1:
            axes = [axes]
        
        dist_name = "t-distribution" if self.distribution_type == 't' else "Gaussian"
        
        if len(self.func_types) == len(self.available_functions):
            func_desc = "all functions"
        else:
            func_desc = f"selected functions ({len(self.func_types)})"
        
        fig.suptitle(f'Training time vs dimension ({func_desc}, n={self.fixed_sample_size}, {dist_name})', 
                     fontsize=16, fontweight='bold')

        func_titles = {
            'quadratic': 'Quadratic',
            'cos': 'Cosine',
            'cos2': 'Cosine-2',
            'cos3': 'Cosine-3', 
            'exp_poly': 'Exp-Poly',
            'additive': 'Additive',
            'multiplication': 'Multiplication'
        }
        for col, func_type in enumerate(self.func_types):
            ax = axes[col]
            if func_type in self.results:
                for method in self.methods:
                    time_values = []
                    time_stds = []
                    valid_dimensions = []
                    
                    for d in self.dimensions:
                        if (d in self.results[func_type] and 
                            method in self.results[func_type][d] and
                            self.results[func_type][d][method] is not None):
                            
                            method_data = self.results[func_type][d][method]
                            time_values.append(method_data['time']['mean'])
                            time_stds.append(method_data['time']['std'])
                            valid_dimensions.append(d)
                    if time_values:
                        ax.errorbar(valid_dimensions, time_values, yerr=time_stds,
                                  marker='o', linestyle='-', linewidth=2, markersize=4,
                                  color=self.method_colors[method], 
                                  label=self.method_names[method],
                                  capsize=3, capthick=1)
            ax.set_title(f'{func_titles[func_type]} - Training Time', fontsize=12, fontweight='bold')
            ax.set_xlabel('Dimension (d)')
            if col == 0:
                ax.set_ylabel('Training Time (seconds)')
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # Log scale for better visibility
            ax.set_xlim(min(self.dimensions)*0.95, max(self.dimensions)*1.05)
            if col == 0:
                ax.legend(loc='upper left', fontsize=9)
        
        plt.tight_layout()
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        dist_suffix = f"_{self.distribution_type}" if self.distribution_type == 't' else ""
        
        if len(self.func_types) < len(self.available_functions):
            func_suffix = "_" + "_".join(self.func_types)
        else:
            func_suffix = ""
        
        plot_filename = f"dimension_time_comparison{dist_suffix}{func_suffix}_{timestamp}.png"
        plot_filepath = os.path.join(self.data_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        print(f"Time comparison plot saved to: {plot_filepath}")
        plt.close()
        
        return plot_filepath
    
    def generate_summary_report(self) -> None:
        print(f"\n{'='*80}")
        print("SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Count successful experiments
        total_configs = len(self.func_types) * len(self.dimensions)
        successful_configs = 0
        
        for func_type in self.func_types:
            for d in self.dimensions:
                if (func_type in self.results and 
                    d in self.results[func_type]):
                    successful_configs += 1
        
        dist_name = "t-distribution" if self.distribution_type == 't' else "Gaussian"
        print(f"Distribution: {dist_name}")
        print(f"Successful configurations: {successful_configs}/{total_configs}")
        print(f"Fixed sample size: {self.fixed_sample_size}")
        print(f"Available dimensions: {self.all_dimensions}")
        print(f"Selected dimensions: {self.dimensions}")
        if self.distribution_type == 't':
            print(f"Degrees of freedom: {self.nu}")
        if len(self.func_types) < len(self.available_functions):
            print(f"Selected functions: {self.func_types}")
        else:
            print("Function types: all functions")
        print(f"\nScreening parameters used:")
        for d in self.dimensions:
            params = self.get_screening_params(d)
            print(f"  d={d}: m={params['m']}, delta={params['delta']}")
        
        print(f"\nMethod Success Rates:")
        for method in self.methods:
            success_count = 0
            for func_type in self.func_types:
                for d in self.dimensions:
                    if (func_type in self.results and 
                        d in self.results[func_type] and
                        method in self.results[func_type][d] and
                        self.results[func_type][d][method] is not None):
                        success_count += 1
            
            success_rate = success_count / total_configs * 100
            print(f"  {self.method_names[method]}: {success_count}/{total_configs} ({success_rate:.1f}%)")

        print(f"\nBest Method by Function Type (by average TPR):")
        for func_type in self.func_types:
            method_tpr_avg = {}
            
            for method in self.methods:
                tpr_values = []
                for d in self.dimensions:
                    if (func_type in self.results and 
                        d in self.results[func_type] and
                        method in self.results[func_type][d] and
                        self.results[func_type][d][method] is not None):
                        tpr_values.append(self.results[func_type][d][method]['tpr']['mean'])
                
                if tpr_values:
                    method_tpr_avg[method] = np.mean(tpr_values)
            
            if method_tpr_avg:
                best_method = max(method_tpr_avg.items(), key=lambda x: x[1])
                print(f"  {func_type.upper()}: {self.method_names[best_method[0]]} (TPR={best_method[1]:.3f})")

        print(f"\nPerformance Trend Analysis:")
        for method in self.methods:
            print(f"\n  {self.method_names[method]}:")
            for func_type in self.func_types:
                tpr_values = []
                valid_dims = []
                
                for d in self.dimensions:
                    if (func_type in self.results and 
                        d in self.results[func_type] and
                        method in self.results[func_type][d] and
                        self.results[func_type][d][method] is not None):
                        tpr_values.append(self.results[func_type][d][method]['tpr']['mean'])
                        valid_dims.append(d)
                
                if len(tpr_values) >= 2:
                    slope = np.polyfit(valid_dims, tpr_values, 1)[0]
                    trend = "increasing" if slope > 0.001 else "decreasing" if slope < -0.001 else "stable"
                    print(f"    {func_type}: {trend} (slope={slope:.4f})")
        
        print(f"{'='*80}")


def main():
    print("Starting high-dimensional comparison experiment")
    print("="*50)
    
    distribution_type = 't'  
    nu = 7  # degrees of freedom (only used for t-distribution)
    
    selected_functions = ['quadratic', 'cos3', 'exp_poly', 'additive', 'multiplication']  # Only evaluate these
    selected_dims = [200, 500, 800, 1000, 2000, 3000, 4000, 5000]

    function_names = {
        'quadratic': 'Quadratic', 
        'cos': 'Cosine', 
        'cos2': 'Cosine2', 
        'cos3': 'Cosine3', 
        'exp_poly': 'ExpPoly', 
        'additive': 'Additive', 
        'multiplication': 'Multiplication'
    }
    
    print(f"Distribution type: {distribution_type}")
    if distribution_type == 't':
        print(f"Degrees of freedom: {nu}")
    
    if selected_functions is None:
        print("Function types: all functions (Quadratic, Cosine, Cosine2, Cosine3, ExpPoly, Additive, Multiplication)")
    else:
        selected_func_names = [function_names.get(f, f) for f in selected_functions]
        print(f"Function types: {', '.join(selected_func_names)}")
    
    print("="*50)
    
    print("Available dimensions and screening parameters:")
    for d, params in DimensionComparison.get_dimension_info().items():
        print(f"  d={d}: m={params['m']}, delta={params['delta']}")
    print()
    
    comparator = DimensionComparison(
        distribution_type=distribution_type, 
        nu=nu, 
        selected_dimensions=selected_dims,
        selected_functions=selected_functions
    )
    
    comparator.run_all_comparisons(
        n_experiments=49,  
        base_seed=1,      
        verbose=True     
    )
    
    results_file = comparator.save_results()
    plot_file = comparator.create_visualization()

    comparator.generate_summary_report()
    print(f"\n{'='*50}")
    print("Experiment finished!")
    print(f"Results saved to: {results_file}")
    print(f"Main visualization saved to: {plot_file}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main() 