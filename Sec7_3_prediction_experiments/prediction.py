import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from variable_selection import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
import time
import warnings
import gc

dfs_path = os.path.join(parent_dir, "deep_feature_selection")
if dfs_path not in sys.path:
    sys.path.append(dfs_path)

from models import Net_linear, Net_nonlinear
from dfs import DFS_epoch, training_l
from utils import data_load_l, measure, mse
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from torch.nn.parameter import Parameter


try:
    from lassonet import LassoNetRegressor
    LASSONET_AVAILABLE = True
except ImportError:
    print("Warning: LassoNet not available. Install with: pip install lassonet")
    LASSONET_AVAILABLE = False

warnings.filterwarnings('ignore')


def get_device(prefer_gpu: bool = True, device_id: int = 0, verbose: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device(f'cuda:{device_id}')
        if verbose:
            gpu_name = torch.cuda.get_device_name(device_id)
            gpu_memory = torch.cuda.get_device_properties(device_id).total_memory / 1e9
            print(f"Using GPU device: {gpu_name} (Memory: {gpu_memory:.1f}GB)")
    else:
        device = torch.device('cpu')
        if verbose:
            if not torch.cuda.is_available():
                print("CUDA not available, using CPU device")
            else:
                print("Manually selected CPU device")
    
    return device

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class SimpleNN(nn.Module):
    """Neural network without bias terms for fair comparison with GPU support"""
    
    def __init__(self, input_dim: int, hidden_layers: list = [64, 32], 
                 random_seed: int = 42, device: torch.device = None):
        super(SimpleNN, self).__init__()
        
        self.random_seed = random_seed
        self.device = device if device is not None else get_device(verbose=False)
        self.layers = nn.ModuleList()
        self.relu = nn.ReLU()
        current_dim = input_dim
        for hidden_dim in hidden_layers:
            self.layers.append(nn.Linear(current_dim, hidden_dim, bias=True))
            current_dim = hidden_dim

        self.output_layer = nn.Linear(current_dim, 1, bias=True)
        self.initialize_weights()

        self.to(self.device)
        
    def initialize_weights(self):
        torch.manual_seed(self.random_seed)  
        
        for layer in self.layers:
            nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
        
    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)

        for layer in self.layers:
            x = self.relu(layer(x))

        x = self.output_layer(x)
        return x

class LassoNetTrainer:
    """Handles LassoNet model training and evaluation with customizable parameters"""
    
    def __init__(self, random_seed: int = 42, hidden_dims: tuple = (100, 50), use_cv: bool = False):
        self.random_seed = random_seed
        self.hidden_dims = hidden_dims
        self.use_cv = use_cv
        
    def train_and_evaluate(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray, 
                          num_features_to_select: int = 5) -> Tuple[float, float, np.ndarray, object]:
        """Train LassoNet and return test MSE, training time, selected feature indices, and model"""
        if not LASSONET_AVAILABLE:
            raise ImportError("LassoNet is not available. Please install it first.")
        
        scaler = StandardScaler()
        X_train_proc = scaler.fit_transform(X_train)
        y_train_proc = y_train.copy()  
        X_test_proc = scaler.transform(X_test)  
        
        start_time = time.time()
        if self.use_cv:
            try:
                from lassonet import LassoNetRegressorCV
                model = LassoNetRegressorCV(hidden_dims=self.hidden_dims, cv = 2, random_state=self.random_seed)
                model.fit(X_train_proc, y_train_proc) 
                training_time = time.time() - start_time
                feature_importances = model.feature_importances_
                
            except ImportError:
                raise ImportError("LassoNetRegressorCV not available. Please update lassonet to the latest version.")
        else:
            model = LassoNetRegressor(hidden_dims=self.hidden_dims, random_state=self.random_seed, lambda_start= 100, path_multiplier= 1.5)
            from sklearn.model_selection import train_test_split
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train_proc, y_train_proc, test_size=0.2, random_state=self.random_seed
            )

            path = model.path(X_train_split, y_train_split, return_state_dicts=True)
            training_time = time.time() - start_time
            best_val_mse = float('inf')
            best_model_id = -1
            
            for i, save in enumerate(path):
                model.load(save.state_dict)

                try:
                    y_val_pred = model.predict(X_val_split)
                    val_mse = mean_squared_error(y_val_split, y_val_pred)
                    if val_mse < best_val_mse:
                        best_val_mse = val_mse
                        best_model_id = i
                except:
                    continue

            if best_model_id >= 0:
                model.load(path[best_model_id].state_dict)
                print(f"Selected model {best_model_id+1} from path (total {len(path)} models), validation MSE: {best_val_mse:.6f}")
            else:
                middle_id = len(path) // 2
                model.load(path[middle_id].state_dict)
                print(f"Using middle model from path (model {middle_id+1} of {len(path)})")

            feature_importances = model.feature_importances_

        if hasattr(feature_importances, 'detach'):
            feature_importances = feature_importances.detach().cpu().numpy()
        elif hasattr(feature_importances, 'numpy'):
            feature_importances = feature_importances.numpy()
        
        selected_indices = np.argsort(feature_importances)[-num_features_to_select:]
        selected_indices = np.sort(selected_indices) 
        y_pred = model.predict(X_test_proc)
        mse = mean_squared_error(y_test, y_pred)
        
        return mse, training_time, selected_indices, model

class DFSSelector:
    """Handles DFS (Deep Feature Selection) method for feature selection using nonlinear model with GPU support"""
    
    def __init__(self, random_seed: int = 42, device: torch.device = None):
        self.random_seed = random_seed
        self.device = device if device is not None else get_device(verbose=False)
        
    def select_features(self, X_train: np.ndarray, y_train: np.ndarray, 
                       true_indices: np.ndarray, k: int = 5, s: int = 5, Ts: int = 25) -> Tuple[np.ndarray, float, object]:
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
        s = s  
        c = 1
        epochs = 10
        n_hidden1 = 100  
        n_hidden2 = 50  
        learning_rate = 0.01  
        step = 4  

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1).to(self.device)
        best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1).to(self.device)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate, weight_decay=0.0025*c)
        optimizer0 = torch.optim.Adam(model.hidden0.parameters(), lr=learning_rate, weight_decay=0.0005*c)

        lf = torch.nn.MSELoss()

        hist = []
        SUPP = []
        LOSSES = []  
        supp_x = list(range(p))  
        SUPP.append(supp_x)
        best_supp = supp_x.copy()  

        for i in range(epochs):
            model, supp_x, LOSS = DFS_epoch(model, s, supp_x, X, Y, lf, optimizer0, optimizer, Ts, step)
            LOSSES = LOSSES + LOSS  
            supp_x.sort()
            current_loss = lf(model(X), Y).cpu().data.numpy().tolist()
            hist.append(current_loss)
            SUPP.append(supp_x)
            if hist[-1] == min(hist):
                best_model.load_state_dict(model.state_dict())
                best_supp = supp_x.copy()
            if len(SUPP[-1]) == len(SUPP[-2]) and (np.array(SUPP[-1]) == np.array(SUPP[-2])).all():
                break
        
        training_time = time.time() - start_time
        final_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=1).to(self.device)
        final_model.load_state_dict(best_model.state_dict())
        del model, best_model, optimizer, optimizer0, X, Y
        
        clear_gpu_memory()
        gc.collect()
        
        return np.array(list(best_supp)), training_time, final_model

class ModelTrainer:
    """Handles model training and evaluation with GPU support and cross validation"""
    
    def __init__(self, epochs: int = 100, lr: float = 0.01, batch_size: int = None,
                 patience: int = 10, min_delta: float = 1e-4, adaptive_lr: bool = True,
                 device: torch.device = None, use_cv: bool = True, cv_folds: int = 5):
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size 
        self.patience = patience  
        self.min_delta = min_delta  
        self.adaptive_lr = adaptive_lr  
        self.device = device if device is not None else get_device(verbose=False)
        self.use_cv = use_cv  
        self.cv_folds = cv_folds  
        
    def train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                   evaluator: 'FeatureSelectionEvaluator', verbose: bool = False) -> Tuple[nn.Module, float]:
        """Train a neural network model and return trained model and training time"""
        start_time = time.time()
        model = model.to(self.device)

        n_samples = X_train.shape[0]
        if self.batch_size is None:
            if n_samples <= 500:
                effective_batch_size = 32
            elif n_samples <= 1000:
                effective_batch_size = 64
            elif n_samples <= 2000:
                effective_batch_size = 128
            elif n_samples <= 4000:
                effective_batch_size = 256
            else:
                effective_batch_size = 512
        else:
            effective_batch_size = self.batch_size

        if self.adaptive_lr:
            if n_samples <= 500:
                effective_lr = self.lr * 0.5
            elif n_samples <= 1000:
                effective_lr = self.lr * 0.75
            else:
                effective_lr = self.lr
        else:
            effective_lr = self.lr
        
        if self.use_cv:
            training_time = self._train_with_cv(model, X_train, y_train, effective_batch_size, effective_lr, verbose)
        else:
            training_time = self._train_with_holdout(model, X_train, y_train, effective_batch_size, effective_lr, n_samples, verbose)
        
        total_training_time = time.time() - start_time
        return model, total_training_time
    
    def _train_with_holdout(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                           effective_batch_size: int, effective_lr: float, n_samples: int, verbose: bool):

        val_ratio = 0.2  
        val_size = max(int(n_samples * val_ratio), 200)  
        val_size = min(val_size, n_samples // 2)  

        indices = np.random.permutation(n_samples)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        X_train_split = X_train[train_indices]
        y_train_split = y_train[train_indices]
        X_val_split = X_train[val_indices]
        y_val_split = y_train[val_indices]
        
        if verbose:
            print(f"Data split: Training set {len(train_indices)} samples, Validation set {len(val_indices)} samples")

        X_train_tensor = torch.FloatTensor(X_train_split).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_split).to(self.device)
        
        X_val_tensor = torch.FloatTensor(X_val_split).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_split).to(self.device)

        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=effective_batch_size, shuffle=False)
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=effective_lr)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                        patience=max(5, self.patience//2), 
                                                        min_lr=effective_lr/100, verbose=False)
        
        best_val_loss = float('inf')
        best_model_state = model.state_dict()
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            model.eval()
            val_loss_total = 0
            val_batches = 0
            
            with torch.no_grad():
                for val_batch_x, val_batch_y in val_loader:
                    val_batch_x = val_batch_x.to(self.device)
                    val_batch_y = val_batch_y.to(self.device)
                    val_outputs = model(val_batch_x)
                    val_batch_loss = criterion(val_outputs.squeeze(), val_batch_y)
                    val_loss_total += val_batch_loss.item()
                    val_batches += 1
            
            val_loss = val_loss_total / val_batches
            
            scheduler.step(val_loss)

            improvement_threshold = self.min_delta
            if len(X_train) > 2000:
                improvement_threshold = self.min_delta * 0.5  
            
            if val_loss < best_val_loss - improvement_threshold:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.patience:
                if verbose:
                    print(f'Early stopping triggered at epoch {epoch}')
                break
        
        # Load best model
        model.load_state_dict(best_model_state)
        
        return 0  
    
    def _train_with_cv(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                      effective_batch_size: int, effective_lr: float, verbose: bool):

        from sklearn.model_selection import KFold
        n_samples = X_train.shape[0]
        
        if verbose:
            print(f"Using {self.cv_folds}-fold cross-validation, total samples: {n_samples}")

        kfold = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        fold_val_losses = []
        best_epochs = []

        best_overall_loss = float('inf')
        best_model_state = None
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train)):
            if verbose:
                print(f"  Training fold {fold+1}/{self.cv_folds}...")

            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]

            X_fold_train_tensor = torch.FloatTensor(X_fold_train).to(self.device)
            y_fold_train_tensor = torch.FloatTensor(y_fold_train).to(self.device)
            X_fold_val_tensor = torch.FloatTensor(X_fold_val).to(self.device)
            y_fold_val_tensor = torch.FloatTensor(y_fold_val).to(self.device)

            fold_train_dataset = TensorDataset(X_fold_train_tensor, y_fold_train_tensor)
            fold_train_loader = DataLoader(fold_train_dataset, batch_size=effective_batch_size, shuffle=True)
            
            fold_val_dataset = TensorDataset(X_fold_val_tensor, y_fold_val_tensor)
            fold_val_loader = DataLoader(fold_val_dataset, batch_size=effective_batch_size, shuffle=False)

            model.initialize_weights()

            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=effective_lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                            patience=max(5, self.patience//2), 
                                                            min_lr=effective_lr/100, verbose=False)

            best_fold_val_loss = float('inf')
            best_fold_model_state = model.state_dict()
            patience_counter = 0
            
            for epoch in range(self.epochs):
                model.train()
                epoch_loss = 0
                for batch_x, batch_y in fold_train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()

                model.eval()
                val_loss_total = 0
                val_batches = 0
                
                with torch.no_grad():
                    for val_batch_x, val_batch_y in fold_val_loader:
                        val_batch_x = val_batch_x.to(self.device)
                        val_batch_y = val_batch_y.to(self.device)
                        val_outputs = model(val_batch_x)
                        val_batch_loss = criterion(val_outputs.squeeze(), val_batch_y)
                        val_loss_total += val_batch_loss.item()
                        val_batches += 1
                
                fold_val_loss = val_loss_total / val_batches

                scheduler.step(fold_val_loss)

                improvement_threshold = self.min_delta
                if n_samples > 2000:
                    improvement_threshold = self.min_delta * 0.5
                
                if fold_val_loss < best_fold_val_loss - improvement_threshold:
                    best_fold_val_loss = fold_val_loss
                    best_fold_model_state = model.state_dict()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"    Fold {fold+1} early stopped at epoch {epoch}")
                    break

            fold_val_losses.append(best_fold_val_loss)
            best_epochs.append(epoch - patience_counter)  

            if best_fold_val_loss < best_overall_loss:
                best_overall_loss = best_fold_val_loss
                best_model_state = best_fold_model_state.copy()
            
            if verbose:
                print(f"    Fold {fold+1} validation loss: {best_fold_val_loss:.6f}")

        mean_val_loss = np.mean(fold_val_losses)
        std_val_loss = np.std(fold_val_losses)
        mean_best_epoch = np.mean(best_epochs)
        
        if verbose:
            print(f"  Cross-validation results:")
            print(f"    Mean validation loss: {mean_val_loss:.6f} ± {std_val_loss:.6f}")
            print(f"    Mean best epoch: {mean_best_epoch:.1f}")
            print(f"    Best overall loss: {best_overall_loss:.6f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        if verbose:
            print(f"  Training on full training data...")

        final_epochs = min(int(mean_best_epoch * 1.1), self.epochs)  
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        final_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        final_loader = DataLoader(final_dataset, batch_size=effective_batch_size, shuffle=True)

        model.initialize_weights()
        final_optimizer = optim.Adam(model.parameters(), lr=effective_lr)
        final_criterion = nn.MSELoss()
        
        for epoch in range(final_epochs):
            model.train()
            for batch_x, batch_y in final_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                final_optimizer.zero_grad()
                outputs = model(batch_x)
                loss = final_criterion(outputs.squeeze(), batch_y)
                loss.backward()
                final_optimizer.step()
        
        if verbose:
            print(f"  Final training completed, trained for {final_epochs} epochs")
        
        return 0  
    
    def evaluate_model(self, model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Evaluate model and return MSE with GPU support"""
        model.eval()
        model = model.to(self.device)
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_tensor).squeeze().cpu().numpy()  
            mse = mean_squared_error(y_test, y_pred)
        return mse

class FeatureSelectionEvaluator:
    """Evaluates the effectiveness of feature selection with GPU support"""
    
    def __init__(self, d: int = 200, k: int = 5, s: int = 5, 
                 n_samples: int = 2000, val_samples: int = 500, test_samples: int = 500,
                 func_type: str = 'quadratic', random_seed: int = 42, selection_method: str = 'stein',
                 device: torch.device = None):
        self.d = d
        self.k = k
        self.s = s
        self.n_samples = n_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.func_type = func_type
        self.random_seed = random_seed
        self.selection_method = selection_method  # 'stein' or 'dfs'
        self.device = device if device is not None else get_device(verbose=False)
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Initialize data containers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.sigma = None
        self.A1 = None
        self.true_indices = None
        self.predicted_indices = None
        self.feature_selection_time = None
        self.func_params = None  # Store function parameters
        
    def generate_data(self):
        """Generate training, validation and test datasets"""
        
        # Generate training data
        self.X_train, self.sigma = generate_samples(self.d, self.n_samples, iid=True)
        self.A1 = np.zeros((self.d, self.k))
        self.true_indices = np.random.choice(self.d, self.s, replace=False)
        #self.A1[self.true_indices] = np.random.rand(self.s, self.k)
        self.A1[self.true_indices] = random_rotation_matrix(self.s)

        self.func_params = np.random.rand(self.k, 1) + 2  # notice
        self.y_train = func_choose(self.X_train, self.A1, self.k, self.func_type, self.func_params) 
        self.X_val, _ = generate_samples(self.d, self.val_samples, iid=True)
        self.y_val = func_choose(self.X_val, self.A1, self.k, self.func_type, self.func_params)
        self.X_test, _ = generate_samples(self.d, self.test_samples, iid=True)
        self.y_test = func_choose(self.X_test, self.A1, self.k, self.func_type, self.func_params)
        
    def perform_variable_selection(self):
        """Apply variable selection algorithm"""
        if self.X_train is None:
            raise ValueError("Please generate data first using generate_data()")
        
        start_time = time.time()
        if self.selection_method == 'stein':
            self.predicted_indices = test_num(self.X_train, self.y_train, self.sigma, self.k, 2*self.k)
            self.feature_selection_time = time.time() - start_time  
        elif self.selection_method == 'dfs':
            dfs_selector = DFSSelector(random_seed=self.random_seed, device=self.device)
            self.predicted_indices, self.feature_selection_time = dfs_selector.select_features(
                self.X_train, self.y_train, self.true_indices, self.k, self.s
            )
        else:
            raise ValueError(f"Unknown selection method: {self.selection_method}")
        
    def calculate_selection_metrics(self) -> dict:
        """Calculate TPR, FPR, precision, recall, and other selection metrics"""
        if self.predicted_indices is None or self.true_indices is None:
            raise ValueError("Please perform variable selection first")
            
        true_set = set(self.true_indices)
        pred_set = set(self.predicted_indices)
        
        # True positives, false positives, false negatives, true negatives
        tp = len(true_set.intersection(pred_set))  
        fp = len(pred_set - true_set)  
        fn = len(true_set - pred_set)  
        tn = self.d - len(true_set.union(pred_set))  

        precision = tp / len(pred_set) if len(pred_set) > 0 else 0
        recall = tp / len(true_set) if len(true_set) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tpr': tpr,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'correct_features': tp,
            'total_true_features': len(true_set),
            'false_positives': fp,
            'false_negatives': fn
        }

class SimplifiedComparator:
    
    def __init__(self, evaluator: FeatureSelectionEvaluator, trainer: ModelTrainer, 
                 hidden_layers: list = [64, 32], include_lassonet: bool = True, 
                 include_dfs: bool = True, device: torch.device = None,
                 dfs_Ts: int = 200, dfs_epochs: int = 10, 
                 dfs_n_hidden1: int = 100, dfs_n_hidden2: int = 50,
                 dfs_learning_rate: float = 0.01, dfs_step: int = 4,
                 dfs_weight_decay_c: float = 1.0,
                 lassonet_hidden_dims: tuple = (100, 50), lassonet_use_cv: bool = False,
                 lassonet_lambda_start: float = 100, lassonet_path_multiplier: float = 1.5,
                 lassonet_tol: float = 1e-4, lassonet_val_split: float = 0.2, 
                 lassonet_standardize_data: bool = True, lassonet_standardize_y: bool = False,
                 lassonet_cv_folds: int = 2, lassonet_verbose: bool = False):
        self.evaluator = evaluator
        self.trainer = trainer
        self.hidden_layers = hidden_layers
        self.include_lassonet = include_lassonet and LASSONET_AVAILABLE
        self.include_dfs = include_dfs
        self.device = device if device is not None else get_device(verbose=False)

        self.dfs_Ts = dfs_Ts
        self.dfs_epochs = dfs_epochs
        self.dfs_n_hidden1 = dfs_n_hidden1
        self.dfs_n_hidden2 = dfs_n_hidden2
        self.dfs_learning_rate = dfs_learning_rate
        self.dfs_step = dfs_step
        self.dfs_weight_decay_c = dfs_weight_decay_c

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
        
        if self.include_lassonet:
            self.lassonet_trainer = LassoNetTrainer(random_seed=evaluator.random_seed, 
                                                   hidden_dims=self.lassonet_hidden_dims,
                                                   use_cv=self.lassonet_use_cv)
        else:
            self.lassonet_trainer = None
            
        if self.include_dfs:
            self.dfs_selector = DFSSelector(random_seed=evaluator.random_seed, device=self.device)
        else:
            self.dfs_selector = None
            
        self.results = {}
        torch.manual_seed(self.evaluator.random_seed)
        np.random.seed(self.evaluator.random_seed)
        
    def _calculate_selection_metrics(self, true_indices: np.ndarray, predicted_indices: np.ndarray, d: int) -> dict:
        """Helper function to calculate feature selection performance metrics"""
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
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'tpr': tpr,
            'fpr': fpr,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        }
        
    def run_comparison(self, verbose: bool = False) -> dict:
    
        self.evaluator.generate_data()

        self.evaluator.perform_variable_selection()
        X_train_full = self.evaluator.X_train
        y_train = self.evaluator.y_train
        X_train_selected = X_train_full[:, self.evaluator.predicted_indices]

        X_test_full, _ = generate_samples(self.evaluator.d, self.evaluator.test_samples, iid=True)
        y_test = func_choose(X_test_full, self.evaluator.A1, self.evaluator.k, self.evaluator.func_type, self.evaluator.func_params)
        X_test_selected = X_test_full[:, self.evaluator.predicted_indices]
        
        if verbose:
            print(f"Training data shape: {X_train_full.shape}")
            print(f"Stein selected features: {len(self.evaluator.predicted_indices)}")
            print(f"Using device: {self.device}")
        
        if verbose:
            print("Training neural network with full features...")
        
        model_full = SimpleNN(input_dim=self.evaluator.d, hidden_layers=self.hidden_layers, 
                             random_seed=self.evaluator.random_seed, device=self.device)
        model_full, train_time_full = self.trainer.train_model(
            model_full, X_train_full, y_train, self.evaluator, verbose=False)
        test_mse_full = self.trainer.evaluate_model(model_full, X_test_full, y_test)
        
        if verbose:
            print("Training neural network with Stein selected features...")
            
        model_selected = SimpleNN(input_dim=len(self.evaluator.predicted_indices), 
                                 hidden_layers=self.hidden_layers, 
                                 random_seed=self.evaluator.random_seed, device=self.device)
        model_selected, train_time_selected = self.trainer.train_model(
            model_selected, X_train_selected, y_train, self.evaluator, verbose=False)
        test_mse_selected = self.trainer.evaluate_model(model_selected, X_test_selected, y_test)
        
        test_mse_lassonet = None
        lassonet_training_time = None
        lassonet_selected_features = None
        
        if self.include_lassonet:
            if verbose:
                print("Training LassoNet model...")
            try:
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                import time

                if self.lassonet_standardize_data:
                    scaler_X = StandardScaler()
                    X_train_proc = scaler_X.fit_transform(X_train_full)
                    X_test_proc = scaler_X.transform(X_test_full)
                else:
                    X_train_proc = X_train_full.copy()
                    X_test_proc = X_test_full.copy()
                
                if self.lassonet_standardize_y:
                    scaler_y = StandardScaler()
                    y_train_proc = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                else:
                    y_train_proc = y_train.copy()

                start_time = time.time()
                
                if self.lassonet_use_cv:
                    from lassonet import LassoNetRegressorCV
                    model = LassoNetRegressorCV(
                        hidden_dims=self.lassonet_hidden_dims, 
                        cv=self.lassonet_cv_folds, 
                        random_state=self.evaluator.random_seed,
                        tol=self.lassonet_tol,
                        verbose=self.lassonet_verbose
                    )
                    model.fit(X_train_proc, y_train_proc)
                    lassonet_training_time = time.time() - start_time
                    feature_importances = model.feature_importances_
                    
                else:
                    from lassonet import LassoNetRegressor

                    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                        X_train_proc, y_train_proc, test_size=self.lassonet_val_split, 
                        random_state=self.evaluator.random_seed
                    )
                    
                    model = LassoNetRegressor(
                        hidden_dims=self.lassonet_hidden_dims, 
                        random_state=self.evaluator.random_seed,
                        lambda_start=self.lassonet_lambda_start,
                        path_multiplier=self.lassonet_path_multiplier,
                        tol=self.lassonet_tol,
                        verbose=self.lassonet_verbose
                    )

                    path = model.path(X_train_split, y_train_split, return_state_dicts=True)
                    lassonet_training_time = time.time() - start_time

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

                lassonet_selected_features = np.argsort(feature_importances)[-self.evaluator.s:]
                lassonet_selected_features = np.sort(lassonet_selected_features)

                y_pred = model.predict(X_test_proc)

                if self.lassonet_standardize_y:
                    y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                
                test_mse_lassonet = mean_squared_error(y_test, y_pred)
                
                if verbose:
                    print(f"LassoNet MSE: {test_mse_lassonet:.6f}, Training time: {lassonet_training_time:.2f}s")
                    print(f"LassoNet selected {len(lassonet_selected_features)} features")
            except Exception as e:
                if verbose:
                    print(f"LassoNet training failed: {e}")
                test_mse_lassonet = None
                lassonet_training_time = None
                lassonet_selected_features = None

        test_mse_dfs = None
        dfs_training_time = None
        dfs_selected_features = None
        
        if self.include_dfs:
            if verbose:
                print("Performing DFS feature selection and model training...")
            try:
                start_time = time.time()
                
                np.random.seed(self.evaluator.random_seed)
                torch.manual_seed(self.evaluator.random_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(self.evaluator.random_seed)
                    torch.cuda.manual_seed_all(self.evaluator.random_seed)

                clear_gpu_memory()

                X_tensor = torch.tensor(X_train_full, dtype=torch.float32).to(self.device)
                Y_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(self.device)

                n_samples, p = X_train_full.shape
                s = self.evaluator.s
                c = 1

                torch.manual_seed(self.evaluator.random_seed)
                np.random.seed(self.evaluator.random_seed)
                model = Net_nonlinear(n_feature=p, n_hidden1=self.dfs_n_hidden1, 
                                     n_hidden2=self.dfs_n_hidden2, n_output=1).to(self.device)
                best_model = Net_nonlinear(n_feature=p, n_hidden1=self.dfs_n_hidden1, 
                                          n_hidden2=self.dfs_n_hidden2, n_output=1).to(self.device)

                optimizer = torch.optim.Adam(list(model.parameters()), lr=self.dfs_learning_rate, 
                                           weight_decay=0.0025*self.dfs_weight_decay_c)
                optimizer0 = torch.optim.Adam(model.hidden0.parameters(), lr=self.dfs_learning_rate, 
                                            weight_decay=0.0005*self.dfs_weight_decay_c)
                
                lf = torch.nn.MSELoss()
                
                hist = []
                SUPP = []
                LOSSES = []
                supp_x = list(range(p))  
                SUPP.append(supp_x.copy())
                best_supp = supp_x.copy()
                
                for epoch in range(self.dfs_epochs):
                    model, supp_x, LOSS = DFS_epoch(model, s, supp_x, X_tensor, Y_tensor, lf, 
                                                   optimizer0, optimizer, self.dfs_Ts, self.dfs_step)
                    LOSSES = LOSSES + LOSS
                    supp_x.sort()
                    
                    current_loss = lf(model(X_tensor), Y_tensor).cpu().data.numpy().tolist()
                    hist.append(current_loss)
                    SUPP.append(supp_x.copy())

                    if hist[-1] == min(hist):
                        best_model.load_state_dict(model.state_dict())
                        best_supp = supp_x.copy()
                    if len(SUPP[-1]) == len(SUPP[-2]) and (np.array(SUPP[-1]) == np.array(SUPP[-2])).all():
                        if verbose:
                            print(f"  DFS converged early at epoch {epoch+1}")
                        break
                
                dfs_training_time = time.time() - start_time
                dfs_selected_features = np.array(list(best_supp))

                X_test_tensor = torch.tensor(X_test_full, dtype=torch.float32).to(self.device)
                Y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(self.device)

                test_mse_dfs = mse(best_model, X_test_tensor, Y_test_tensor)
                
                if verbose:
                    print(f"DFS selected {len(dfs_selected_features)} features")
                    print(f"DFS MSE: {test_mse_dfs:.6f}, Training time: {dfs_training_time:.2f}s")
                    print(f"DFS convergence epochs: {len(hist)}/{self.dfs_epochs}")

                del model, best_model, optimizer, optimizer0
                del X_tensor, Y_tensor, X_test_tensor, Y_test_tensor
                clear_gpu_memory()
                gc.collect()
                    
            except Exception as e:
                print(f"DFS training failed: {e}")
                import traceback
                traceback.print_exc()
                test_mse_dfs = None
                dfs_training_time = None
                dfs_selected_features = None
        
        clear_gpu_memory()

        selection_metrics = self.evaluator.calculate_selection_metrics()

        lassonet_selection_metrics = None
        if lassonet_selected_features is not None:
            lassonet_selection_metrics = self._calculate_selection_metrics(
                self.evaluator.true_indices, lassonet_selected_features, self.evaluator.d
            )
        dfs_selection_metrics = None
        if dfs_selected_features is not None:
            dfs_selection_metrics = self._calculate_selection_metrics(
                self.evaluator.true_indices, dfs_selected_features, self.evaluator.d
            )
        
        self.results = {
            # MSE results for all four methods
            'mse_full': test_mse_full,
            'mse_selected': test_mse_selected,
            'mse_lassonet': test_mse_lassonet,
            'mse_dfs': test_mse_dfs,
            
            # MSE improvements (relative to full features)
            'mse_improvement_selected': ((test_mse_full - test_mse_selected) / test_mse_full * 100) if test_mse_full > 0 else 0,
            'mse_improvement_lassonet': ((test_mse_full - test_mse_lassonet) / test_mse_full * 100) if test_mse_full > 0 and test_mse_lassonet is not None else None,
            'mse_improvement_dfs': ((test_mse_full - test_mse_dfs) / test_mse_full * 100) if test_mse_full > 0 and test_mse_dfs is not None else None,
            
            # Training times
            'train_time_full': train_time_full,
            'train_time_selected': train_time_selected,
            'train_time_lassonet': lassonet_training_time,
            'train_time_dfs': dfs_training_time,
            'feature_selection_time': self.evaluator.feature_selection_time,
            'total_time_selected': train_time_selected + self.evaluator.feature_selection_time,
            
            # Feature selection performance (Stein method)
            'tpr': selection_metrics['tpr'],
            'fpr': selection_metrics['fpr'],
            'precision': selection_metrics['precision'],
            'recall': selection_metrics['recall'],
            'f1_score': selection_metrics['f1_score'],
            
            # Feature selection performance (LassoNet method)
            'lassonet_tpr': lassonet_selection_metrics['tpr'] if lassonet_selection_metrics else None,
            'lassonet_fpr': lassonet_selection_metrics['fpr'] if lassonet_selection_metrics else None,
            'lassonet_precision': lassonet_selection_metrics['precision'] if lassonet_selection_metrics else None,
            'lassonet_recall': lassonet_selection_metrics['recall'] if lassonet_selection_metrics else None,
            'lassonet_f1_score': lassonet_selection_metrics['f1_score'] if lassonet_selection_metrics else None,
            
            # Feature selection performance (DFS method)
            'dfs_tpr': dfs_selection_metrics['tpr'] if dfs_selection_metrics else None,
            'dfs_fpr': dfs_selection_metrics['fpr'] if dfs_selection_metrics else None,
            'dfs_precision': dfs_selection_metrics['precision'] if dfs_selection_metrics else None,
            'dfs_recall': dfs_selection_metrics['recall'] if dfs_selection_metrics else None,
            'dfs_f1_score': dfs_selection_metrics['f1_score'] if dfs_selection_metrics else None,
            
            # Feature counts and selected indices
            'selected_features_stein': len(self.evaluator.predicted_indices),
            'selected_features_lassonet': len(lassonet_selected_features) if lassonet_selected_features is not None else None,
            'selected_features_dfs': len(dfs_selected_features) if dfs_selected_features is not None else None,
            'true_features': len(self.evaluator.true_indices),
            
            # Selected feature indices
            'selected_indices_stein': self.evaluator.predicted_indices,
            'selected_indices_lassonet': lassonet_selected_features,
            'selected_indices_dfs': dfs_selected_features,
            'true_indices': self.evaluator.true_indices
        }
        
        if verbose:
            print(f"\n=== MSE Comparison of Four Methods ===")
            print(f"Full features MSE:      {test_mse_full:.6f}")
            print(f"Stein selected MSE:     {test_mse_selected:.6f} (improvement: {self.results['mse_improvement_selected']:.2f}%)")
            if test_mse_lassonet is not None:
                print(f"LassoNet MSE:     {test_mse_lassonet:.6f} (improvement: {self.results['mse_improvement_lassonet']:.2f}%)")
            if test_mse_dfs is not None:
                print(f"DFS MSE:          {test_mse_dfs:.6f} (improvement: {self.results['mse_improvement_dfs']:.2f}%)")
            
            print(f"\n=== Feature Selection Performance Comparison ===")
            print(f"Stein method - Precision: {selection_metrics['precision']:.4f}, Recall: {selection_metrics['recall']:.4f}, F1: {selection_metrics['f1_score']:.4f}")
            if lassonet_selection_metrics is not None:
                print(f"LassoNet - Precision: {lassonet_selection_metrics['precision']:.4f}, Recall: {lassonet_selection_metrics['recall']:.4f}, F1: {lassonet_selection_metrics['f1_score']:.4f}")
            if dfs_selection_metrics is not None:
                print(f"DFS - Precision: {dfs_selection_metrics['precision']:.4f}, Recall: {dfs_selection_metrics['recall']:.4f}, F1: {dfs_selection_metrics['f1_score']:.4f}")
            
            print(f"\n=== Training Time Comparison ===")
            print(f"Full features training:     {train_time_full:.2f}s")
            print(f"Stein selected training:    {train_time_selected:.2f}s")
            if lassonet_training_time is not None:
                print(f"LassoNet training:     {lassonet_training_time:.2f}s")
            if dfs_training_time is not None:
                print(f"DFS total training:        {dfs_training_time:.2f}s")
        
        return self.results

def run_simplified_comparison(d: int = 200, k: int = 5, s: int = 5, 
                            n_samples: int = 2000, test_samples: int = 500,
                            func_type: str = 'quadratic', epochs: int = 100,
                            hidden_layers: list = [64, 32], patience: int = 10,
                            lr: float = 0.01, min_delta: float = 1e-4,
                            random_seed: int = 42, selection_method: str = 'stein',
                            use_gpu: bool = True, device_id: int = 0,
                            dfs_Ts: int = 200, lassonet_hidden_dims: tuple = (100, 50),
                            use_cv: bool = False, cv_folds: int = 5) -> dict:
    # Get computing device
    device = get_device(prefer_gpu=use_gpu, device_id=device_id, verbose=True)
    
    # Initialize components with GPU support
    evaluator = FeatureSelectionEvaluator(d=d, k=k, s=s, n_samples=n_samples, 
                                        test_samples=test_samples, func_type=func_type, 
                                        random_seed=random_seed, selection_method=selection_method,
                                        device=device)
    trainer = ModelTrainer(epochs=epochs, patience=patience, lr=lr, min_delta=min_delta,
                          device=device, use_cv=use_cv, cv_folds=cv_folds)
    comparator = SimplifiedComparator(evaluator, trainer, hidden_layers=hidden_layers,
                                    device=device, dfs_Ts=dfs_Ts, 
                                    lassonet_hidden_dims=lassonet_hidden_dims)
    
    # Run comparison
    results = comparator.run_comparison()
    
    return results

def run_multiple_simplified_experiments(n_experiments: int = 10, d: int = 200, k: int = 5, s: int = 5, 
                                       n_samples: int = 2000, test_samples: int = 2000,
                                       func_type: str = 'quadratic', epochs: int = 100,
                                       hidden_layers: list = [64, 32], selection_method: str = 'stein',
                                       base_seed: int = 42, include_lassonet: bool = True, 
                                       include_dfs: bool = True, use_gpu: bool = True, 
                                       device_id: int = 0, use_cv: bool = False, cv_folds: int = 5,
                                       # DFS-specific parameters
                                       dfs_Ts: int = 100, dfs_epochs: int = 10, 
                                       dfs_n_hidden1: int = 100, dfs_n_hidden2: int = 50,
                                       dfs_learning_rate: float = 0.01, dfs_step: int = 4,
                                       dfs_weight_decay_c: float = 1.0,
                                       # LassoNet-specific parameters
                                       lassonet_hidden_dims: tuple = (100, 50), lassonet_use_cv: bool = False,
                                       lassonet_lambda_start: float = 100, lassonet_path_multiplier: float = 1.5,
                                       lassonet_tol: float = 1e-4, lassonet_val_split: float = 0.2, 
                                       lassonet_standardize_data: bool = True, lassonet_standardize_y: bool = False,
                                       lassonet_cv_folds: int = 2, lassonet_verbose: bool = False) -> dict:


    device = get_device(prefer_gpu=use_gpu, device_id=device_id, verbose=True)
    
    print(f"Running {n_experiments} four-method comparison experiments...")
    print(f"Using device: {device}")
    print(f"DFS inner iterations (Ts): {dfs_Ts}")
    print(f"LassoNet hidden layer dimensions: {lassonet_hidden_dims}")
    print(f"Using cross-validation: {'Yes' if use_cv else 'No'} ({cv_folds} folds)" if use_cv else "Using traditional train/validation split")
    print("="*60)
    
    # Store all metrics - extended to four methods
    metrics = {
        # MSE results
        'mse_full': [], 'mse_selected': [], 'mse_lassonet': [], 'mse_dfs': [],
        
        # MSE improvements
        'mse_improvement_selected': [], 'mse_improvement_lassonet': [], 'mse_improvement_dfs': [],
        
        # Training times
        'train_time_full': [], 'train_time_selected': [], 'train_time_lassonet': [], 'train_time_dfs': [],
        'feature_selection_time': [], 'total_time_selected': [],
        
        # Feature selection performance (Stein method)
        'tpr': [], 'fpr': [], 'precision': [], 'recall': [], 'f1_score': [],
        
        # Feature selection performance (LassoNet method)
        'lassonet_tpr': [], 'lassonet_fpr': [], 'lassonet_precision': [], 'lassonet_recall': [], 'lassonet_f1_score': [],
        
        # Feature selection performance (DFS method)
        'dfs_tpr': [], 'dfs_fpr': [], 'dfs_precision': [], 'dfs_recall': [], 'dfs_f1_score': [],
        
        # Feature counts
        'selected_features_stein': [], 'selected_features_lassonet': [], 'selected_features_dfs': [], 'true_features': []
    }
    
    # Record number of successful experiments
    successful_experiments = 0
    lassonet_successes = 0
    dfs_successes = 0
    
    for i in range(n_experiments):
        current_seed = base_seed + i * 1000
        
        print(f"Experiment {i+1}/{n_experiments} (seed: {current_seed})")
        
        try:
            # Run single experiment (GPU accelerated)
            evaluator = FeatureSelectionEvaluator(d=d, k=k, s=s, n_samples=n_samples, 
                                                test_samples=test_samples, func_type=func_type, 
                                                random_seed=current_seed, selection_method=selection_method,
                                                device=device)
            trainer = ModelTrainer(epochs=epochs, patience=10, lr=0.01, min_delta=1e-4,
                                 device=device, use_cv=use_cv, cv_folds=cv_folds)
            comparator = SimplifiedComparator(evaluator, trainer, hidden_layers=hidden_layers,
                                            include_lassonet=include_lassonet, include_dfs=include_dfs,
                                            device=device, 
                                            # DFS parameters
                                            dfs_Ts=dfs_Ts, dfs_epochs=dfs_epochs,
                                            dfs_n_hidden1=dfs_n_hidden1, dfs_n_hidden2=dfs_n_hidden2,
                                            dfs_learning_rate=dfs_learning_rate, dfs_step=dfs_step,
                                            dfs_weight_decay_c=dfs_weight_decay_c,
                                            # LassoNet parameters
                                            lassonet_hidden_dims=lassonet_hidden_dims, 
                                            lassonet_use_cv=lassonet_use_cv,
                                            lassonet_lambda_start=lassonet_lambda_start, 
                                            lassonet_path_multiplier=lassonet_path_multiplier,
                                            lassonet_tol=lassonet_tol, lassonet_val_split=lassonet_val_split, 
                                            lassonet_standardize_data=lassonet_standardize_data, 
                                            lassonet_standardize_y=lassonet_standardize_y,
                                            lassonet_cv_folds=lassonet_cv_folds, 
                                            lassonet_verbose=lassonet_verbose)
            
            results = comparator.run_comparison(verbose=False)
            
            # Collect basic metrics (always present)
            metrics['mse_full'].append(results['mse_full'])
            metrics['mse_selected'].append(results['mse_selected'])
            metrics['mse_improvement_selected'].append(results['mse_improvement_selected'])
            
            metrics['train_time_full'].append(results['train_time_full'])
            metrics['train_time_selected'].append(results['train_time_selected'])
            metrics['feature_selection_time'].append(results['feature_selection_time'])
            metrics['total_time_selected'].append(results['total_time_selected'])
            
            metrics['tpr'].append(results['tpr'])
            metrics['fpr'].append(results['fpr'])
            metrics['precision'].append(results['precision'])
            metrics['recall'].append(results['recall'])
            metrics['f1_score'].append(results['f1_score'])
            
            metrics['selected_features_stein'].append(results['selected_features_stein'])
            metrics['true_features'].append(results['true_features'])
            
            # Collect LassoNet metrics (if available)
            if results['mse_lassonet'] is not None:
                metrics['mse_lassonet'].append(results['mse_lassonet'])
                metrics['mse_improvement_lassonet'].append(results['mse_improvement_lassonet'])
                metrics['train_time_lassonet'].append(results['train_time_lassonet'])
                metrics['selected_features_lassonet'].append(results['selected_features_lassonet'])
                
                # LassoNet feature selection performance metrics
                metrics['lassonet_tpr'].append(results['lassonet_tpr'])
                metrics['lassonet_fpr'].append(results['lassonet_fpr'])
                metrics['lassonet_precision'].append(results['lassonet_precision'])
                metrics['lassonet_recall'].append(results['lassonet_recall'])
                metrics['lassonet_f1_score'].append(results['lassonet_f1_score'])
                
                lassonet_successes += 1
            else:
                metrics['mse_lassonet'].append(None)
                metrics['mse_improvement_lassonet'].append(None)
                metrics['train_time_lassonet'].append(None)
                metrics['selected_features_lassonet'].append(None)
                
                # LassoNet feature selection performance metrics (add None on failure)
                metrics['lassonet_tpr'].append(None)
                metrics['lassonet_fpr'].append(None)
                metrics['lassonet_precision'].append(None)
                metrics['lassonet_recall'].append(None)
                metrics['lassonet_f1_score'].append(None)
            
            # Collect DFS metrics (if available)
            if results['mse_dfs'] is not None:
                metrics['mse_dfs'].append(results['mse_dfs'])
                metrics['mse_improvement_dfs'].append(results['mse_improvement_dfs'])
                metrics['train_time_dfs'].append(results['train_time_dfs'])
                metrics['selected_features_dfs'].append(results['selected_features_dfs'])
                
                # DFS feature selection performance metrics
                metrics['dfs_tpr'].append(results['dfs_tpr'])
                metrics['dfs_fpr'].append(results['dfs_fpr'])
                metrics['dfs_precision'].append(results['dfs_precision'])
                metrics['dfs_recall'].append(results['dfs_recall'])
                metrics['dfs_f1_score'].append(results['dfs_f1_score'])
                
                dfs_successes += 1
            else:
                metrics['mse_dfs'].append(None)
                metrics['mse_improvement_dfs'].append(None)
                metrics['train_time_dfs'].append(None)
                metrics['selected_features_dfs'].append(None)
                
                # DFS feature selection performance metrics (add None on failure)
                metrics['dfs_tpr'].append(None)
                metrics['dfs_fpr'].append(None)
                metrics['dfs_precision'].append(None)
                metrics['dfs_recall'].append(None)
                metrics['dfs_f1_score'].append(None)
            
            successful_experiments += 1
            
            # Brief progress report - add improvement percentage and TPR
            full_mse = results['mse_full']
            selected_improvement = ((full_mse - results['mse_selected']) / full_mse * 100) if full_mse > 0 else 0
            stein_tpr = results['tpr']
            
            # LassoNet results and improvement percentage
            if results['mse_lassonet'] is not None:
                lassonet_improvement = ((full_mse - results['mse_lassonet']) / full_mse * 100) if full_mse > 0 else 0
                lassonet_result = f"{results['mse_lassonet']:.4f}({lassonet_improvement:+.1f}%)"
            else:
                lassonet_result = "Failed"
            
            # DFS results and improvement percentage
            if results['mse_dfs'] is not None:
                dfs_improvement = ((full_mse - results['mse_dfs']) / full_mse * 100) if full_mse > 0 else 0
                dfs_result = f"{results['mse_dfs']:.4f}({dfs_improvement:+.1f}%)"
            else:
                dfs_result = "Failed"
            
            print(f"  MSE: Full={results['mse_full']:.4f}, Selected={results['mse_selected']:.4f}({selected_improvement:+.1f}%), "
                  f"LassoNet={lassonet_result}, DFS={dfs_result}, Stein-TPR={stein_tpr:.3f}")
            
        except Exception as e:
            print(f"  Experiment {i+1} failed: {e}")
            continue
    
    print(f"\nSuccessfully completed {successful_experiments}/{n_experiments} experiments")
    print(f"LassoNet successes: {lassonet_successes}/{successful_experiments}")
    print(f"DFS successes: {dfs_successes}/{successful_experiments}")
    
    # Calculate statistics - handle None values
    def safe_stats(values):
        """Safely calculate statistics, filter None values"""
        clean_values = [v for v in values if v is not None]
        if len(clean_values) == 0:
            return {'mean': None, 'std': None, 'min': None, 'max': None, 'median': None, 'count': 0}
        return {
            'mean': np.mean(clean_values),
            'std': np.std(clean_values),
            'min': np.min(clean_values),
            'max': np.max(clean_values),
            'median': np.median(clean_values),
            'count': len(clean_values)
        }
    
    summary_stats = {}
    for metric, values in metrics.items():
        summary_stats[metric] = safe_stats(values)
    
    # Print four-method comparison results
    print(f"\nFour-Method Comparison Results ({successful_experiments} experiments):")
    print("="*80)
    
    print(f"MSE Performance Comparison:")
    print(f"  1. Full features MSE:     {summary_stats['mse_full']['mean']} ± {summary_stats['mse_full']['std']}")
    print(f"  2. Stein selected MSE:    {summary_stats['mse_selected']['mean']} ± {summary_stats['mse_selected']['std']}")
    
    if summary_stats['mse_lassonet']['count'] > 0:
        print(f"  3. LassoNet MSE:    {summary_stats['mse_lassonet']['mean']} ± {summary_stats['mse_lassonet']['std']} ({summary_stats['mse_lassonet']['count']} successes)")
    else:
        print(f"  3. LassoNet MSE:    Failed")
    
    if summary_stats['mse_dfs']['count'] > 0:
        print(f"  4. DFS MSE:         {summary_stats['mse_dfs']['mean']} ± {summary_stats['mse_dfs']['std']} ({summary_stats['mse_dfs']['count']} successes)")
    else:
        print(f"  4. DFS MSE:         Failed")
    
    print(f"\n=== MSE Improvement (relative to full features) ===")
    print(f"  Stein selected average improvement:  {summary_stats['mse_improvement_selected']['mean']:.2f}% ± {summary_stats['mse_improvement_selected']['std']:.2f}%")
    
    if summary_stats['mse_improvement_lassonet']['count'] > 0:
        print(f"  LassoNet average improvement:   {summary_stats['mse_improvement_lassonet']['mean']:.2f}% ± {summary_stats['mse_improvement_lassonet']['std']:.2f}%")
    else:
        print(f"  LassoNet average improvement:   Failed")
    
    if summary_stats['mse_improvement_dfs']['count'] > 0:
        print(f"  DFS average improvement:        {summary_stats['mse_improvement_dfs']['mean']:.2f}% ± {summary_stats['mse_improvement_dfs']['std']:.2f}%")
    else:
        print(f"  DFS average improvement:        Failed")
    
    print(f"\nFeature Selection Performance Comparison:")
    print(f"Stein method:")
    print(f"  TPR (Sensitivity):       {summary_stats['tpr']['mean']:.4f} ± {summary_stats['tpr']['std']:.4f}")
    print(f"  FPR (False Positive Rate):     {summary_stats['fpr']['mean']:.4f} ± {summary_stats['fpr']['std']:.4f}")
    print(f"  Precision:             {summary_stats['precision']['mean']:.4f} ± {summary_stats['precision']['std']:.4f}")
    print(f"  Recall:             {summary_stats['recall']['mean']:.4f} ± {summary_stats['recall']['std']:.4f}")
    print(f"  F1 Score:             {summary_stats['f1_score']['mean']:.4f} ± {summary_stats['f1_score']['std']:.4f}")
    
    if summary_stats['lassonet_precision']['count'] > 0:
        print(f"LassoNet method:")
        print(f"  TPR (Sensitivity):       {summary_stats['lassonet_tpr']['mean']:.4f} ± {summary_stats['lassonet_tpr']['std']:.4f}")
        print(f"  FPR (False Positive Rate):     {summary_stats['lassonet_fpr']['mean']:.4f} ± {summary_stats['lassonet_fpr']['std']:.4f}")
        print(f"  Precision:             {summary_stats['lassonet_precision']['mean']:.4f} ± {summary_stats['lassonet_precision']['std']:.4f}")
        print(f"  Recall:             {summary_stats['lassonet_recall']['mean']:.4f} ± {summary_stats['lassonet_recall']['std']:.4f}")
        print(f"  F1 Score:             {summary_stats['lassonet_f1_score']['mean']:.4f} ± {summary_stats['lassonet_f1_score']['std']:.4f}")
    
    if summary_stats['dfs_precision']['count'] > 0:
        print(f"DFS method:")
        print(f"  TPR (Sensitivity):       {summary_stats['dfs_tpr']['mean']:.4f} ± {summary_stats['dfs_tpr']['std']:.4f}")
        print(f"  FPR (False Positive Rate):     {summary_stats['dfs_fpr']['mean']:.4f} ± {summary_stats['dfs_fpr']['std']:.4f}")
        print(f"  Precision:             {summary_stats['dfs_precision']['mean']:.4f} ± {summary_stats['dfs_precision']['std']:.4f}")
        print(f"  Recall:             {summary_stats['dfs_recall']['mean']:.4f} ± {summary_stats['dfs_recall']['std']:.4f}")
        print(f"  F1 Score:             {summary_stats['dfs_f1_score']['mean']:.4f} ± {summary_stats['dfs_f1_score']['std']:.4f}")
    
    print(f"\n=== Training Time Comparison (seconds) ===")
    print(f"Model training time:")
    print(f"  Full features training:       {summary_stats['train_time_full']['mean']:.2f} ± {summary_stats['train_time_full']['std']:.2f}")
    print(f"  Stein selected training:      {summary_stats['train_time_selected']['mean']:.2f} ± {summary_stats['train_time_selected']['std']:.2f}")
    
    if summary_stats['train_time_lassonet']['count'] > 0:
        print(f"  LassoNet training:       {summary_stats['train_time_lassonet']['mean']:.2f} ± {summary_stats['train_time_lassonet']['std']:.2f}")
    
    if summary_stats['train_time_dfs']['count'] > 0:
        print(f"  DFS total training:          {summary_stats['train_time_dfs']['mean']:.2f} ± {summary_stats['train_time_dfs']['std']:.2f}")
    
    print(f"\nFeature selection time:")
    print(f"  Stein feature selection:      {summary_stats['feature_selection_time']['mean']:.4f} ± {summary_stats['feature_selection_time']['std']:.4f}")
    print(f"  Stein total time (selection+training): {summary_stats['total_time_selected']['mean']:.2f} ± {summary_stats['total_time_selected']['std']:.2f}")
    
    # Method ranking
    print(f"\n=== Method Ranking (by average MSE) ===")
    methods_mse = []
    if summary_stats['mse_full']['mean'] is not None:
        methods_mse.append(('Full features', summary_stats['mse_full']['mean']))
    if summary_stats['mse_selected']['mean'] is not None:
        methods_mse.append(('Stein selected', summary_stats['mse_selected']['mean']))
    if summary_stats['mse_lassonet']['mean'] is not None:
        methods_mse.append(('LassoNet', summary_stats['mse_lassonet']['mean']))
    if summary_stats['mse_dfs']['mean'] is not None:
        methods_mse.append(('DFS', summary_stats['mse_dfs']['mean']))
    
    # Sort by MSE (lower is better)
    methods_mse.sort(key=lambda x: x[1])
    
    for i, (method, mse) in enumerate(methods_mse, 1):
        print(f"  {i}. {method}: {mse:.6f}")
    
    return {
        'raw_metrics': metrics,
        'summary_stats': summary_stats,
        'success_counts': {
            'total': successful_experiments,
            'lassonet': lassonet_successes,
            'dfs': dfs_successes
        }
    }




