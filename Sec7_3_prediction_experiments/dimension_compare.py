import sys
import os
import numpy as np
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import multiprocessing as mp
from functools import partial

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from prediction import run_multiple_simplified_experiments, get_device, clear_gpu_memory
warnings.filterwarnings('ignore')

def run_single_configuration_worker(config, runner_params):
    d, func_type = config
    print(f"\n{'='*60}")
    print(f"[Worker PID: {os.getpid()}] Running: d={d}, func_type={func_type}")
    print(f"{'='*60}")
    
    try:
        # Run multiple experiments for this configuration
        results = run_multiple_simplified_experiments(
            n_experiments=runner_params['n_experiments'],
            d=d,
            k=runner_params['k'],
            s=runner_params['s'],
            n_samples=runner_params['n_samples'],
            test_samples=runner_params['test_samples'],
            func_type=func_type,
            epochs=runner_params['epochs'],
            base_seed=runner_params['base_seed'],
            include_lassonet=True,
            include_dfs=True,
            use_gpu=runner_params['use_gpu'],
            device_id=runner_params['device_id'],
            use_cv=runner_params['use_cv'],
            cv_folds=runner_params['cv_folds'],
            
            # DFS-specific parameters
            dfs_Ts=runner_params['dfs_Ts'],
            dfs_epochs=runner_params['dfs_epochs'],
            dfs_n_hidden1=runner_params['dfs_n_hidden1'],
            dfs_n_hidden2=runner_params['dfs_n_hidden2'],
            dfs_learning_rate=runner_params['dfs_learning_rate'],
            dfs_step=runner_params['dfs_step'],
            dfs_weight_decay_c=runner_params['dfs_weight_decay_c'],
            
            # LassoNet-specific parameters
            lassonet_hidden_dims=runner_params['lassonet_hidden_dims'],
            lassonet_use_cv=runner_params['lassonet_use_cv'],
            lassonet_lambda_start=runner_params['lassonet_lambda_start'],
            lassonet_path_multiplier=runner_params['lassonet_path_multiplier'],
            lassonet_tol=runner_params['lassonet_tol'],
            lassonet_val_split=runner_params['lassonet_val_split'],
            lassonet_standardize_data=runner_params['lassonet_standardize_data'],
            lassonet_standardize_y=runner_params['lassonet_standardize_y'],
            lassonet_cv_folds=runner_params['lassonet_cv_folds'],
            lassonet_verbose=runner_params['lassonet_verbose']
        )
        
        # Extract key metrics for storage
        summary_stats = results['summary_stats']
        success_counts = results['success_counts']
        
        config_results = {
            'd': d,
            'func_type': func_type,
            'n_samples': runner_params['n_samples'],
            'success_counts': success_counts,
            'timestamp': datetime.now().isoformat(),
            
            # MSE metrics
            'mse_full': {
                'mean': summary_stats['mse_full']['mean'],
                'std': summary_stats['mse_full']['std'],
                'count': summary_stats['mse_full']['count']
            },
            'mse_stein': {
                'mean': summary_stats['mse_selected']['mean'],
                'std': summary_stats['mse_selected']['std'],
                'count': summary_stats['mse_selected']['count']
            },
            'mse_lassonet': {
                'mean': summary_stats['mse_lassonet']['mean'],
                'std': summary_stats['mse_lassonet']['std'],
                'count': summary_stats['mse_lassonet']['count']
            } if summary_stats['mse_lassonet']['mean'] is not None else None,
            'mse_dfs': {
                'mean': summary_stats['mse_dfs']['mean'],
                'std': summary_stats['mse_dfs']['std'],
                'count': summary_stats['mse_dfs']['count']
            } if summary_stats['mse_dfs']['mean'] is not None else None,
            
            # MSE improvements (relative to full model)
            'improvement_stein': {
                'mean': summary_stats['mse_improvement_selected']['mean'],
                'std': summary_stats['mse_improvement_selected']['std']
            },
            'improvement_lassonet': {
                'mean': summary_stats['mse_improvement_lassonet']['mean'],
                'std': summary_stats['mse_improvement_lassonet']['std']
            } if summary_stats['mse_improvement_lassonet']['mean'] is not None else None,
            'improvement_dfs': {
                'mean': summary_stats['mse_improvement_dfs']['mean'],
                'std': summary_stats['mse_improvement_dfs']['std']
            } if summary_stats['mse_improvement_dfs']['mean'] is not None else None,
            
            # Feature selection performance (TPR, FPR)
            'stein_selection': {
                'tpr': {'mean': summary_stats['tpr']['mean'], 'std': summary_stats['tpr']['std']},
                'fpr': {'mean': summary_stats['fpr']['mean'], 'std': summary_stats['fpr']['std']},
                'precision': {'mean': summary_stats['precision']['mean'], 'std': summary_stats['precision']['std']},
                'recall': {'mean': summary_stats['recall']['mean'], 'std': summary_stats['recall']['std']},
                'f1_score': {'mean': summary_stats['f1_score']['mean'], 'std': summary_stats['f1_score']['std']}
            },
            'lassonet_selection': {
                'tpr': {'mean': summary_stats['lassonet_tpr']['mean'], 'std': summary_stats['lassonet_tpr']['std']},
                'fpr': {'mean': summary_stats['lassonet_fpr']['mean'], 'std': summary_stats['lassonet_fpr']['std']},
                'precision': {'mean': summary_stats['lassonet_precision']['mean'], 'std': summary_stats['lassonet_precision']['std']},
                'recall': {'mean': summary_stats['lassonet_recall']['mean'], 'std': summary_stats['lassonet_recall']['std']},
                'f1_score': {'mean': summary_stats['lassonet_f1_score']['mean'], 'std': summary_stats['lassonet_f1_score']['std']}
            } if summary_stats['lassonet_tpr']['mean'] is not None else None,
            'dfs_selection': {
                'tpr': {'mean': summary_stats['dfs_tpr']['mean'], 'std': summary_stats['dfs_tpr']['std']},
                'fpr': {'mean': summary_stats['dfs_fpr']['mean'], 'std': summary_stats['dfs_fpr']['std']},
                'precision': {'mean': summary_stats['dfs_precision']['mean'], 'std': summary_stats['dfs_precision']['std']},
                'recall': {'mean': summary_stats['dfs_recall']['mean'], 'std': summary_stats['dfs_recall']['std']},
                'f1_score': {'mean': summary_stats['dfs_f1_score']['mean'], 'std': summary_stats['dfs_f1_score']['std']}
            } if summary_stats['dfs_tpr']['mean'] is not None else None,
            
            # Training times
            'train_time_full': {
                'mean': summary_stats['train_time_full']['mean'],
                'std': summary_stats['train_time_full']['std']
            },
            'train_time_stein': {
                'mean': summary_stats['train_time_selected']['mean'],
                'std': summary_stats['train_time_selected']['std']
            },
            'train_time_lassonet': {
                'mean': summary_stats['train_time_lassonet']['mean'],
                'std': summary_stats['train_time_lassonet']['std']
            } if summary_stats['train_time_lassonet']['mean'] is not None else None,
            'train_time_dfs': {
                'mean': summary_stats['train_time_dfs']['mean'],
                'std': summary_stats['train_time_dfs']['std']
            } if summary_stats['train_time_dfs']['mean'] is not None else None,
            'feature_selection_time': {
                'mean': summary_stats['feature_selection_time']['mean'],
                'std': summary_stats['feature_selection_time']['std']
            }
        }
        
        print(f"[Worker PID: {os.getpid()}] Completed: d={d}, func_type={func_type}")
        return (func_type, d, config_results)
        
    except Exception as e:
        print(f"[Worker PID: {os.getpid()}] Error in configuration d={d}, func_type={func_type}: {e}")
        import traceback
        traceback.print_exc()
        return (func_type, d, None)

class ServerExperimentRunner:
    """Server-side experiment runner for testing different feature dimensions and function types"""
    
    def __init__(self, base_seed=42, n_experiments=1, use_gpu=True, device_id=0, 
                 n_samples=2000, k=5, s=5, test_samples=2000, epochs=100, 
                 dfs_Ts=25, dfs_epochs=10, dfs_n_hidden1=100, dfs_n_hidden2=50,
                 dfs_learning_rate=0.01, dfs_step=4, dfs_weight_decay_c=1.0,
                 lassonet_hidden_dims=(100, 50), lassonet_use_cv=False,
                 lassonet_lambda_start=100, lassonet_path_multiplier=1.5,
                 lassonet_tol=1e-4, lassonet_val_split=0.2,
                 lassonet_standardize_data=True, lassonet_standardize_y=False,
                 lassonet_cv_folds=2, lassonet_verbose=False,
                 use_cv=False, cv_folds=5,
                 use_parallel=True, n_workers=None,
                 selected_functions=None):
        self.base_seed = base_seed
        self.n_experiments = n_experiments
        self.use_gpu = use_gpu
        self.device_id = device_id
        self.n_samples = n_samples
        self.k = k
        self.s = s
        self.test_samples = test_samples
        self.epochs = epochs
        
        # DFS parameter storage
        self.dfs_Ts = dfs_Ts
        self.dfs_epochs = dfs_epochs
        self.dfs_n_hidden1 = dfs_n_hidden1
        self.dfs_n_hidden2 = dfs_n_hidden2
        self.dfs_learning_rate = dfs_learning_rate
        self.dfs_step = dfs_step
        self.dfs_weight_decay_c = dfs_weight_decay_c
        
        # LassoNet parameter storage
        self.lassonet_hidden_dims = lassonet_hidden_dims
        self.lassonet_use_cv = lassonet_use_cv
        self.lassonet_lambda_start = lassonet_lambda_start
        self.lassonet_path_multiplier = lassonet_path_multiplier
        self.lassonet_tol = lassonet_tol
        self.lassonet_val_split = lassonet_val_split
        self.lassonet_standardize_data = lassonet_standardize_data
        self.lassonet_standardize_y = lassonet_standardize_y
        self.lassonet_cv_folds = lassonet_cv_folds
        self.lassonet_verbose = lassonet_verbose
        
        self.use_cv = use_cv
        self.cv_folds = cv_folds
        self.use_parallel = use_parallel
        self.n_workers = n_workers if n_workers is not None else mp.cpu_count()
        
        # Experiment configurations - testing different feature dimensions
        self.d_list = [100,200,400,600,800,1000]
        self.d_list = [100,600,1000]
        
        # Available function types list
        self.available_functions = ['quadratic', 'cos', 'cos2', 'cos3', 'exp_poly', 'additive', 'multiplication']
        if selected_functions is None:
            self.func_types = self.available_functions.copy()
        else:
            # Validate that selected function types are valid
            valid_functions = [f for f in selected_functions if f in self.available_functions]
            if not valid_functions:
                raise ValueError(f"No valid function types selected. Available function types: {self.available_functions}")
            self.func_types = valid_functions
        
        # Results storage
        self.results = {func_type: {} for func_type in self.func_types}
        
        # Create server_data directory if it doesn't exist
        self.data_dir = os.path.join(current_dir, 'prediction_data')
        os.makedirs(self.data_dir, exist_ok=True)
        
        print(f"Server experiment runner initialized (feature dimension testing)")
        print(f"Device: {get_device(prefer_gpu=use_gpu, device_id=device_id, verbose=False)}")
        print(f"Feature dimensions (d): {self.d_list}")
        if len(self.func_types) == len(self.available_functions):
            print(f"Function types: All functions ({self.func_types})")
        else:
            print(f"Function types: Selected functions ({self.func_types})")
        print(f"Fixed sample size: {n_samples}")
        print(f"Number of experiments per configuration: {n_experiments}")
        print(f"Cross validation: {'Enabled' if use_cv else 'Disabled'} ({cv_folds} folds)" if use_cv else "Cross validation: Disabled")
        print(f"Parallel processing: {'Enabled' if use_parallel else 'Disabled'} ({self.n_workers} worker processes)" if use_parallel else "Parallel processing: Disabled")
        print(f"Results will be saved to: {self.data_dir}")
    
    def get_runner_params(self):
        """Get all runner parameters as a dictionary for worker processes"""
        return {
            'base_seed': self.base_seed,
            'n_experiments': self.n_experiments,
            'use_gpu': self.use_gpu,
            'device_id': self.device_id,
            'n_samples': self.n_samples,
            'k': self.k,
            's': self.s,
            'test_samples': self.test_samples,
            'epochs': self.epochs,
            'dfs_Ts': self.dfs_Ts,
            'dfs_epochs': self.dfs_epochs,
            'dfs_n_hidden1': self.dfs_n_hidden1,
            'dfs_n_hidden2': self.dfs_n_hidden2,
            'dfs_learning_rate': self.dfs_learning_rate,
            'dfs_step': self.dfs_step,
            'dfs_weight_decay_c': self.dfs_weight_decay_c,
            'lassonet_hidden_dims': self.lassonet_hidden_dims,
            'lassonet_use_cv': self.lassonet_use_cv,
            'lassonet_lambda_start': self.lassonet_lambda_start,
            'lassonet_path_multiplier': self.lassonet_path_multiplier,
            'lassonet_tol': self.lassonet_tol,
            'lassonet_val_split': self.lassonet_val_split,
            'lassonet_standardize_data': self.lassonet_standardize_data,
            'lassonet_standardize_y': self.lassonet_standardize_y,
            'lassonet_cv_folds': self.lassonet_cv_folds,
            'lassonet_verbose': self.lassonet_verbose,
            'use_cv': self.use_cv,
            'cv_folds': self.cv_folds
        }

    def run_single_configuration(self, d, func_type):
        """Run experiments for a single configuration (d, func_type) - legacy method for backward compatibility"""
        print(f"\n{'='*60}")
        print(f"Running: d={d}, func_type={func_type}")
        print(f"{'='*60}")
        
        try:
            # Run multiple experiments for this configuration
            results = run_multiple_simplified_experiments(
                n_experiments=self.n_experiments,
                d=d,
                k=self.k,
                s=self.s,
                n_samples=self.n_samples,
                test_samples=self.test_samples,
                func_type=func_type,
                epochs=self.epochs,
                base_seed=self.base_seed,
                include_lassonet=True,
                include_dfs=True,
                use_gpu=self.use_gpu,
                device_id=self.device_id,
                use_cv=self.use_cv,
                cv_folds=self.cv_folds,
                
                # DFS-specific parameters
                dfs_Ts=self.dfs_Ts,
                dfs_epochs=self.dfs_epochs,
                dfs_n_hidden1=self.dfs_n_hidden1,
                dfs_n_hidden2=self.dfs_n_hidden2,
                dfs_learning_rate=self.dfs_learning_rate,
                dfs_step=self.dfs_step,
                dfs_weight_decay_c=self.dfs_weight_decay_c,
                
                # LassoNet-specific parameters
                lassonet_hidden_dims=self.lassonet_hidden_dims,
                lassonet_use_cv=self.lassonet_use_cv,
                lassonet_lambda_start=self.lassonet_lambda_start,
                lassonet_path_multiplier=self.lassonet_path_multiplier,
                lassonet_tol=self.lassonet_tol,
                lassonet_val_split=self.lassonet_val_split,
                lassonet_standardize_data=self.lassonet_standardize_data,
                lassonet_standardize_y=self.lassonet_standardize_y,
                lassonet_cv_folds=self.lassonet_cv_folds,
                lassonet_verbose=self.lassonet_verbose
            )
            
            # Extract key metrics for storage
            summary_stats = results['summary_stats']
            success_counts = results['success_counts']
            
            config_results = {
                'd': d,
                'func_type': func_type,
                'n_samples': self.n_samples,
                'success_counts': success_counts,
                'timestamp': datetime.now().isoformat(),
                
                # MSE metrics
                'mse_full': {
                    'mean': summary_stats['mse_full']['mean'],
                    'std': summary_stats['mse_full']['std'],
                    'count': summary_stats['mse_full']['count']
                },
                'mse_stein': {
                    'mean': summary_stats['mse_selected']['mean'],
                    'std': summary_stats['mse_selected']['std'],
                    'count': summary_stats['mse_selected']['count']
                },
                'mse_lassonet': {
                    'mean': summary_stats['mse_lassonet']['mean'],
                    'std': summary_stats['mse_lassonet']['std'],
                    'count': summary_stats['mse_lassonet']['count']
                } if summary_stats['mse_lassonet']['mean'] is not None else None,
                'mse_dfs': {
                    'mean': summary_stats['mse_dfs']['mean'],
                    'std': summary_stats['mse_dfs']['std'],
                    'count': summary_stats['mse_dfs']['count']
                } if summary_stats['mse_dfs']['mean'] is not None else None,
                
                # MSE improvements (relative to full model)
                'improvement_stein': {
                    'mean': summary_stats['mse_improvement_selected']['mean'],
                    'std': summary_stats['mse_improvement_selected']['std']
                },
                'improvement_lassonet': {
                    'mean': summary_stats['mse_improvement_lassonet']['mean'],
                    'std': summary_stats['mse_improvement_lassonet']['std']
                } if summary_stats['mse_improvement_lassonet']['mean'] is not None else None,
                'improvement_dfs': {
                    'mean': summary_stats['mse_improvement_dfs']['mean'],
                    'std': summary_stats['mse_improvement_dfs']['std']
                } if summary_stats['mse_improvement_dfs']['mean'] is not None else None,
                
                # Feature selection performance (TPR, FPR)
                'stein_selection': {
                    'tpr': {'mean': summary_stats['tpr']['mean'], 'std': summary_stats['tpr']['std']},
                    'fpr': {'mean': summary_stats['fpr']['mean'], 'std': summary_stats['fpr']['std']},
                    'precision': {'mean': summary_stats['precision']['mean'], 'std': summary_stats['precision']['std']},
                    'recall': {'mean': summary_stats['recall']['mean'], 'std': summary_stats['recall']['std']},
                    'f1_score': {'mean': summary_stats['f1_score']['mean'], 'std': summary_stats['f1_score']['std']}
                },
                'lassonet_selection': {
                    'tpr': {'mean': summary_stats['lassonet_tpr']['mean'], 'std': summary_stats['lassonet_tpr']['std']},
                    'fpr': {'mean': summary_stats['lassonet_fpr']['mean'], 'std': summary_stats['lassonet_fpr']['std']},
                    'precision': {'mean': summary_stats['lassonet_precision']['mean'], 'std': summary_stats['lassonet_precision']['std']},
                    'recall': {'mean': summary_stats['lassonet_recall']['mean'], 'std': summary_stats['lassonet_recall']['std']},
                    'f1_score': {'mean': summary_stats['lassonet_f1_score']['mean'], 'std': summary_stats['lassonet_f1_score']['std']}
                } if summary_stats['lassonet_tpr']['mean'] is not None else None,
                'dfs_selection': {
                    'tpr': {'mean': summary_stats['dfs_tpr']['mean'], 'std': summary_stats['dfs_tpr']['std']},
                    'fpr': {'mean': summary_stats['dfs_fpr']['mean'], 'std': summary_stats['dfs_fpr']['std']},
                    'precision': {'mean': summary_stats['dfs_precision']['mean'], 'std': summary_stats['dfs_precision']['std']},
                    'recall': {'mean': summary_stats['dfs_recall']['mean'], 'std': summary_stats['dfs_recall']['std']},
                    'f1_score': {'mean': summary_stats['dfs_f1_score']['mean'], 'std': summary_stats['dfs_f1_score']['std']}
                } if summary_stats['dfs_tpr']['mean'] is not None else None,
                
                # Training times
                'train_time_full': {
                    'mean': summary_stats['train_time_full']['mean'],
                    'std': summary_stats['train_time_full']['std']
                },
                'train_time_stein': {
                    'mean': summary_stats['train_time_selected']['mean'],
                    'std': summary_stats['train_time_selected']['std']
                },
                'train_time_lassonet': {
                    'mean': summary_stats['train_time_lassonet']['mean'],
                    'std': summary_stats['train_time_lassonet']['std']
                } if summary_stats['train_time_lassonet']['mean'] is not None else None,
                'train_time_dfs': {
                    'mean': summary_stats['train_time_dfs']['mean'],
                    'std': summary_stats['train_time_dfs']['std']
                } if summary_stats['train_time_dfs']['mean'] is not None else None,
                'feature_selection_time': {
                    'mean': summary_stats['feature_selection_time']['mean'],
                    'std': summary_stats['feature_selection_time']['std']
                }
            }
            
            return config_results
            
        except Exception as e:
            print(f"Error in configuration d={d}, func_type={func_type}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def run_all_experiments(self):
        """Run all experiments across different configurations with optional parallel processing"""
        total_configs = len(self.d_list) * len(self.func_types)
        
        print(f"\nStarting server experiments (feature dimension testing)...")
        print(f"Total configurations: {total_configs}")
        print(f"Total experiments: {total_configs * self.n_experiments}")
        print(f"Parallel processing: {'Enabled' if self.use_parallel else 'Disabled'}")
        
        start_time = time.time()
        
        if self.use_parallel:
            self._run_experiments_parallel()
        else:
            self._run_experiments_sequential()
        
        total_time = time.time() - start_time
        print(f"\nAll experiments completed!")
        print(f"Total time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
        
        # Save complete results
        self.save_complete_results()
        
        # Generate visualizations
        self.generate_visualization()  # MSE improvement comparison
        self.generate_mse_visualization()  # Absolute MSE values comparison
    
    def _run_experiments_parallel(self):
        """Run experiments in parallel using multiprocessing"""
        print(f"Using {self.n_workers} worker processes for parallel execution...")
        
        # Create list of all configurations
        configs = []
        for func_type in self.func_types:
            for d in self.d_list:
                configs.append((d, func_type))
        
        runner_params = self.get_runner_params()
        
        # Use multiprocessing Pool to run configurations in parallel
        with mp.Pool(processes=self.n_workers) as pool:
            worker_func = partial(run_single_configuration_worker, runner_params=runner_params)
            # Execute all configurations in parallel
            print(f"Submitting {len(configs)} configurations to worker pool...")
            results = pool.map(worker_func, configs)
        
        # Process results
        print("Processing parallel execution results...")
        for func_type, d, config_results in results:
            if config_results is not None:
                if func_type not in self.results:
                    self.results[func_type] = {}
                self.results[func_type][d] = config_results
            else:
                print(f"Warning: Failed to get results for d={d}, func_type={func_type}")
    
    def _run_experiments_sequential(self):
        """Run experiments sequentially (original behavior)"""
        current_config = 0
        total_configs = len(self.d_list) * len(self.func_types)
        
        for func_type in self.func_types:
            self.results[func_type] = {}
            
            for d in self.d_list:
                current_config += 1
                print(f"\nProgress: {current_config}/{total_configs}")
                
                # Run experiments for this configuration
                config_results = self.run_single_configuration(d, func_type)
                
                if config_results is not None:
                    self.results[func_type][d] = config_results
                
                # Clear GPU memory between configurations
                clear_gpu_memory()
    
    def save_complete_results(self):
        """Save complete results to file"""
        if len(self.func_types) < len(self.available_functions):
            func_suffix = "_" + "_".join(self.func_types)
        else:
            func_suffix = ""
        complete_filename = f"complete_results_s{func_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        complete_filepath = os.path.join(self.data_dir, complete_filename)
        
        with open(complete_filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Complete results saved: {complete_filename}")
        summary_filename = f"summary_stats_s{func_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_filepath = os.path.join(self.data_dir, summary_filename)
        
        summary_stats = self.calculate_summary_statistics()
        with open(summary_filepath, 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        
        print(f"Summary statistics saved: {summary_filename}")
    
    def calculate_summary_statistics(self):
        summary = {}
        for func_type in self.func_types:
            summary[func_type] = {}
            
            for d in self.d_list:
                if d in self.results[func_type]:
                    result = self.results[func_type][d]
                    
                    summary[func_type][d] = {
                        'mse_values': {
                            'full': result['mse_full']['mean'] if result['mse_full'] else None,
                            'stein': result['mse_stein']['mean'] if result['mse_stein'] else None,
                            'lassonet': result['mse_lassonet']['mean'] if result['mse_lassonet'] else None,
                            'dfs': result['mse_dfs']['mean'] if result['mse_dfs'] else None
                        },
                        'mse_improvements': {
                            # MSE improvement percentages relative to full model (positive = better)
                            'stein': result['improvement_stein']['mean'] if result['improvement_stein'] else None,
                            'lassonet': result['improvement_lassonet']['mean'] if result['improvement_lassonet'] else None,
                            'dfs': result['improvement_dfs']['mean'] if result['improvement_dfs'] else None
                        },
                        'feature_selection_performance': {
                            'stein_tpr': result['stein_selection']['tpr']['mean'],
                            'stein_fpr': result['stein_selection']['fpr']['mean'],
                            'lassonet_tpr': result['lassonet_selection']['tpr']['mean'] if result['lassonet_selection'] else None,
                            'lassonet_fpr': result['lassonet_selection']['fpr']['mean'] if result['lassonet_selection'] else None,
                            'dfs_tpr': result['dfs_selection']['tpr']['mean'] if result['dfs_selection'] else None,
                            'dfs_fpr': result['dfs_selection']['fpr']['mean'] if result['dfs_selection'] else None
                        },
                        'success_rates': result['success_counts']
                    }
        
        return summary
    
    def generate_visualization(self):
        """Generate visualization comparing MSE improvements across methods and function types vs feature dimensions"""
        print("\nGenerating visualization charts...")

        plt.style.use('default')
        sns.set_palette("husl")

        num_funcs = len(self.func_types)
        ncols = min(num_funcs, 7)  
        fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))
        if num_funcs == 1:
            axes = [axes]
        
        if len(self.func_types) == len(self.available_functions):
            func_desc = "All functions"
        else:
            func_desc = f"Selected functions ({len(self.func_types)})"
        
        fig.suptitle(f'MSE Improvement Relative to Full Model vs Feature Dimension ({func_desc})', fontsize=16, y=1.02)
        
        methods = ['stein', 'lassonet', 'dfs']
        method_colors = {'stein': '#1f77b4', 'lassonet': '#ff7f0e', 'dfs': '#2ca02c'}
        method_labels = {'stein': 'Stein', 'lassonet': 'LassoNet', 'dfs': 'DFS'}
        
        for i, func_type in enumerate(self.func_types):
            ax = axes[i]
            d_data = []
            improvements_data = {method: [] for method in methods}
            improvements_std = {method: [] for method in methods}
            
            for d in self.d_list:
                if d in self.results[func_type]:
                    result = self.results[func_type][d]
                    d_data.append(d)
                    
                    if result['improvement_stein']:
                        improvements_data['stein'].append(result['improvement_stein']['mean'])
                        improvements_std['stein'].append(result['improvement_stein']['std'])
                    else:
                        improvements_data['stein'].append(None)
                        improvements_std['stein'].append(None)
                    
                    if result['improvement_lassonet']:
                        improvements_data['lassonet'].append(result['improvement_lassonet']['mean'])
                        improvements_std['lassonet'].append(result['improvement_lassonet']['std'])
                    else:
                        improvements_data['lassonet'].append(None)
                        improvements_std['lassonet'].append(None)

                    if result['improvement_dfs']:
                        improvements_data['dfs'].append(result['improvement_dfs']['mean'])
                        improvements_std['dfs'].append(result['improvement_dfs']['std'])
                    else:
                        improvements_data['dfs'].append(None)
                        improvements_std['dfs'].append(None)

            for method in methods:
                if any(x is not None for x in improvements_data[method]):
                    valid_indices = [j for j, x in enumerate(improvements_data[method]) if x is not None]
                    valid_d = [d_data[j] for j in valid_indices]
                    valid_improvements = [improvements_data[method][j] for j in valid_indices]
                    valid_stds = [improvements_std[method][j] for j in valid_indices]
                    
                    if valid_d: 
                        ax.errorbar(valid_d, valid_improvements, yerr=valid_stds,
                                  marker='o', linestyle='-', linewidth=2, markersize=6,
                                  color=method_colors[method], label=method_labels[method],
                                  capsize=5, capthick=2)

            ax.set_title(f'{func_type.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Dimension (d)', fontsize=12)
            if i == 0:  
                ax.set_ylabel('MSE Improvement (%)', fontsize=12)
            
            ax.grid(True, alpha=0.3)

            ax.set_xticks(self.d_list[::2])  
            ax.set_xticklabels([str(d) for d in self.d_list[::2]], rotation=45)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            if i == 0:
                ax.legend(loc='best', fontsize=10)

        plt.tight_layout()
        if len(self.func_types) < len(self.available_functions):
            func_suffix = "_" + "_".join(self.func_types)
        else:
            func_suffix = ""

        plot_filename = f"mse_improvement_comparison_s{func_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_filepath = os.path.join(self.data_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        print(f"Visualization chart saved: {plot_filename}")
        #plt.show()

    
    def generate_mse_visualization(self):
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Dynamically calculate number of columns
        num_funcs = len(self.func_types)
        ncols = min(num_funcs, 7)  
        fig, axes = plt.subplots(1, ncols, figsize=(4*ncols, 4))

        if num_funcs == 1:
            axes = [axes]
        if len(self.func_types) == len(self.available_functions):
            func_desc = "All functions"
        else:
            func_desc = f"Selected functions ({len(self.func_types)})"
        
        fig.suptitle(f'Absolute MSE Values vs Feature Dimension ({func_desc})', fontsize=16, y=1.02)
        
        methods = ['full', 'stein', 'lassonet', 'dfs']
        method_colors = {'full': '#d62728', 'stein': '#1f77b4', 'lassonet': '#ff7f0e', 'dfs': '#2ca02c'}
        method_labels = {'full': 'Full Model', 'stein': 'Stein', 'lassonet': 'LassoNet', 'dfs': 'DFS'}
        method_markers = {'full': 's', 'stein': 'o', 'lassonet': '^', 'dfs': 'v'}
        
        for i, func_type in enumerate(self.func_types):
            ax = axes[i]
            
            # Collect data for this function type
            d_data = []
            mse_data = {method: [] for method in methods}
            mse_std = {method: [] for method in methods}
            
            for d in self.d_list:
                if d in self.results[func_type]:
                    result = self.results[func_type][d]
                    d_data.append(d)
                    
                    # Full model MSE
                    if result['mse_full']:
                        mse_data['full'].append(result['mse_full']['mean'])
                        mse_std['full'].append(result['mse_full']['std'])
                    else:
                        mse_data['full'].append(None)
                        mse_std['full'].append(None)
                    
                    # Stein MSE
                    if result['mse_stein']:
                        mse_data['stein'].append(result['mse_stein']['mean'])
                        mse_std['stein'].append(result['mse_stein']['std'])
                    else:
                        mse_data['stein'].append(None)
                        mse_std['stein'].append(None)
                    
                    # LassoNet MSE
                    if result['mse_lassonet']:
                        mse_data['lassonet'].append(result['mse_lassonet']['mean'])
                        mse_std['lassonet'].append(result['mse_lassonet']['std'])
                    else:
                        mse_data['lassonet'].append(None)
                        mse_std['lassonet'].append(None)
                    
                    # DFS MSE
                    if result['mse_dfs']:
                        mse_data['dfs'].append(result['mse_dfs']['mean'])
                        mse_std['dfs'].append(result['mse_dfs']['std'])
                    else:
                        mse_data['dfs'].append(None)
                        mse_std['dfs'].append(None)
            
            # Plot each method
            for method in methods:
                if any(x is not None for x in mse_data[method]):
                    # Filter out None values
                    valid_indices = [j for j, x in enumerate(mse_data[method]) if x is not None]
                    valid_d = [d_data[j] for j in valid_indices]
                    valid_mse = [mse_data[method][j] for j in valid_indices]
                    valid_stds = [mse_std[method][j] for j in valid_indices]
                    
                    if valid_d:  # Only plot if we have valid data
                        ax.errorbar(valid_d, valid_mse, yerr=valid_stds,
                                  marker=method_markers[method], linestyle='-', linewidth=2, markersize=6,
                                  color=method_colors[method], label=method_labels[method],
                                  capsize=5, capthick=2)

            ax.set_title(f'{func_type.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Feature Dimension (d)', fontsize=12)
            if i == 0:  
                ax.set_ylabel('MSE', fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(self.d_list[::2])  
            ax.set_xticklabels([str(d) for d in self.d_list[::2]], rotation=45)
            ax.set_yscale('log')
            if i == 0:
                ax.legend(loc='best', fontsize=10)
        plt.tight_layout()

        if len(self.func_types) < len(self.available_functions):
            func_suffix = "_" + "_".join(self.func_types)
        else:
            func_suffix = ""
        plot_filename = f"mse_values_comparison_s{func_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plot_filepath = os.path.join(self.data_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
        
        print(f"Absolute MSE values visualization chart saved: {plot_filename}")
        plt.show()


def main():

    # Function selection - modify these settings to select functions to compare
    # None means all functions, or specify a list
    selected_functions = ['quadratic', 'cos3', 'exp_poly', 'additive', 'multiplication']  # Only test this function

    function_names = {
        'quadratic': 'Quadratic', 
        'cos': 'Cosine', 
        'cos2': 'Cosine2', 
        'cos3': 'Cosine3', 
        'exp_poly': 'ExpPoly', 
        'additive': 'Additive', 
        'multiplication': 'Multiplication'
    }
    
    if selected_functions is None:
        print(f"Function types: All functions (Quadratic, Cosine, Cosine2, Cosine3, ExpPoly, Additive, Multiplication)")
    else:
        selected_func_names = [function_names.get(f, f) for f in selected_functions]
        print(f"Function types: {', '.join(selected_func_names)}")
    print("="*80)

    runner = ServerExperimentRunner(
        base_seed=1,
        n_experiments=99,  
        use_gpu=True,      
        device_id=3,       
        n_samples=2000,    
        k=5,               
        s=5,               
        test_samples=2000, 
        epochs=100,        
        
        # DFS-specific parameter configuration
        dfs_Ts=25,                    
        dfs_epochs=10,                
        dfs_n_hidden1=100,           
        dfs_n_hidden2=50,             
        dfs_learning_rate=0.01,       
        dfs_step=5,                   
        dfs_weight_decay_c=1.0,       
        
        # LassoNet-specific parameter configuration
        lassonet_hidden_dims=(100, 50),     
        lassonet_use_cv=False,               
        lassonet_lambda_start=10.0,         
        lassonet_path_multiplier=1.5,      
        lassonet_tol=1e-4,                  
        lassonet_val_split=0.1,             
        lassonet_standardize_data=True,      
        lassonet_standardize_y=False,        
        lassonet_cv_folds=2,                
        lassonet_verbose=False,           
        
        # Neural network training parameters
        use_cv=True,       
        cv_folds=5,        
        
        # Parallel processing parameters
        use_parallel=False,    
        n_workers=16,          
        selected_functions=selected_functions  
    )
    
    # Run all experiments
    runner.run_all_experiments()
    
    print("\nServer experiments completed successfully!")
    print(f"Results saved to: {runner.data_dir}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
