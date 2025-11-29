import os
import sys
from math import gamma
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from variable_selection import *
import warnings 
import numpy as np
import random
warnings.filterwarnings('ignore')

np.random.seed(45)        
random.seed(45)

d = 100
n_samples = 30000
rho = 0 
k = 5
s = 5
func_type = 'quadratic'
num_features_to_select = s 

def top_k_eigenvectors(matrix, k=5):
    eigenvalues, eigenvectors = eigh(matrix)
    indices = np.argsort(eigenvalues)[-k:]

    top_k_vectors = eigenvectors[:, indices]
    return top_k_vectors

X_train, sigma = generate_samples_rho(d, n_samples, rho)
A1 = np.zeros((d, k))
true_indices = np.random.choice(d, s, replace=False)
true_indices = np.arange(s)
A1[true_indices] = np.identity(s)
func_params = np.random.rand(k, 1) + 1

y_train = func_choose(X_train, A1, k, func_type, func_params).flatten()
r = EyTx(X_train, y_train, sigma)

eigenvalues = np.linalg.eigvals(r)
sorted_eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
eigengaps = []
for i in range(len(sorted_eigenvalues)-1):
    eigengap = np.abs(sorted_eigenvalues[i]) - np.abs(sorted_eigenvalues[i+1])
    eigengaps.append(eigengap)
eigengaps = np.array(eigengaps)
# calculate the true EyTX:
diag_matrix = np.zeros((len(func_params), len(func_params)))
np.fill_diagonal(diag_matrix, func_params*2)
EyTx_true = A1 @  diag_matrix @ A1.T
# calulate tao be the minimum of the true eigengaps of EyTx_true
eigenval_true = np.linalg.eigvals(EyTx_true)
sorted_eigenval_true = np.sort(np.abs(eigenval_true))[::-1]
eigengaps_true = []
for i in range(len(sorted_eigenval_true)-1):
    eigengap = np.abs(sorted_eigenval_true[i]) - np.abs(sorted_eigenval_true[i+1])
    eigengaps_true.append(eigengap)
eigengaps_true_larger0 = [gap for gap in eigengaps_true if gap > 0]
tao = np.min(eigengaps_true_larger0)/2 
print(tao)

# Plot eigenvalues and eigengaps with JRSSB journal quality
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set basic style without font changes
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.5,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False
})


fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))  

# Define professional color palette
colors = {
    'primary': '#2E86AB',    # Professional blue
    'secondary': '#E74C3C',  # Beautiful red
    'accent': '#F18F01',     # Orange
    'neutral': '#34495E'     # Dark blue-gray
}

cutoff_point_gaps = 30
gap_indices_display = np.arange(1, min(cutoff_point_gaps, len(eigengaps)) + 1)
eigengaps_display = eigengaps[:min(cutoff_point_gaps, len(eigengaps))]

ax2.plot(gap_indices_display, eigengaps_display, 
         color=colors['primary'], linewidth=2, alpha=0.8, zorder=2)
ax2.scatter(gap_indices_display, eigengaps_display, 
           color=colors['secondary'], s=15, alpha=0.9, zorder=3, edgecolors='white', linewidth=0.3)

# Add threshold line with better styling
if len(eigengaps) > cutoff_point_gaps:
    ax2.axhline(y=tao, color=colors['accent'], linestyle='--', linewidth=2, 
               alpha=0.8, zorder=1, label=rf'$\tau = {tao:.4f}$') 
else:
    ax2.axhline(y=tao, color=colors['accent'], linestyle='--', linewidth=2, 
               alpha=0.8, zorder=1, label=rf'$\tau = {tao:.4f}$')
if len(eigengaps) > cutoff_point_gaps:
    compressed_99_pos = 36
    final_gap_value = eigengaps_display[-1]
    x_connect = np.linspace(cutoff_point_gaps + 1, compressed_99_pos, 10) 
    y_connect = np.full_like(x_connect, final_gap_value)

    ax2.plot(x_connect, y_connect, color=colors['neutral'], linestyle=':', 
             linewidth=1.5, alpha=0.6, zorder=1)
    ax2.scatter([compressed_99_pos], [final_gap_value], color=colors['secondary'], s=15, 
               alpha=0.7, zorder=3, edgecolors='white', linewidth=0.3)

ax2.set_xlabel('Gap Index')
ax2.set_ylabel('Eigengap')
ax2.set_title(r'Eigengaps $|\hat{\lambda}_{k}| - |\hat{\lambda}_{k+1}|$')

ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax2.set_axisbelow(True)
if len(eigengaps) > cutoff_point_gaps:
    compressed_99_pos = 36
    ax2.set_xlim(0.5, compressed_99_pos + 2)
    xticks_final = list(range(1, cutoff_point_gaps+1, 10)) + [compressed_99_pos]
    xlabels_final = [str(x) for x in range(1, cutoff_point_gaps+1, 10)] + ['99']
    ax2.set_xticks(xticks_final)
    ax2.set_xticklabels(xlabels_final)
else:
    ax2.set_xlim(0.5, len(eigengaps) + 0.5)
y_min, y_max = ax2.get_ylim()
ax2.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
legend = ax2.legend(loc='best', frameon=True, fancybox=False, shadow=False, 
                   framealpha=0.9, edgecolor='gray', facecolor='white')
legend.get_frame().set_linewidth(0.5)
gap_ratios = []
for i in range(len(eigengaps)-1):
    if eigengaps[i+1] != 0: 
        ratio = (eigengaps[i] )/ (eigengaps[i+1] + 0.01)
        gap_ratios.append(ratio)
    else:
        gap_ratios.append(np.inf) 
gap_ratios = np.array(gap_ratios)
cutoff_point_ratios = 30
ratio_indices_display = np.arange(2, min(cutoff_point_ratios, len(gap_ratios)) + 2)
ratios_display = gap_ratios[:min(cutoff_point_ratios, len(gap_ratios))]

ax3.plot(ratio_indices_display, ratios_display, 
         color=colors['primary'], linewidth=2, alpha=0.8, zorder=2)
ax3.scatter(ratio_indices_display, ratios_display, 
           color=colors['secondary'], s=15, alpha=0.9, zorder=3, edgecolors='white', linewidth=0.3)
if len(gap_ratios) > cutoff_point_ratios:
    compressed_98_pos = 37
    final_ratio_value = ratios_display[-1] 
    x_connect = np.linspace(cutoff_point_ratios + 2, compressed_98_pos, 10)  
    y_connect = np.full_like(x_connect, final_ratio_value)
    ax3.plot(x_connect, y_connect, color=colors['neutral'], linestyle=':', 
             linewidth=1.5, alpha=0.6, zorder=1)
    ax3.scatter([compressed_98_pos], [final_ratio_value], color=colors['secondary'], s=15, 
               alpha=0.7, zorder=3, edgecolors='white', linewidth=0.3)

ax3.set_xlabel('Ratio Index')
ax3.set_ylabel('Gap Ratio')
ax3.set_title(r'Gap Ratios $r(k) = (|\hat{\lambda}_{k-1}| - |\hat{\lambda}_{k}|) / (|\hat{\lambda}_{k}| - |\hat{\lambda}_{k+1}| + \gamma_{reg})$')

ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax3.set_axisbelow(True)
if len(gap_ratios) > cutoff_point_ratios:
    compressed_98_pos = 37
    ax3.set_xlim(1.5, compressed_98_pos + 2)
    xticks_final = list(range(2, cutoff_point_ratios+2, 10)) + [compressed_98_pos]
    xlabels_final = [str(x) for x in range(2, cutoff_point_ratios+2, 10)] + ['98']
    ax3.set_xticks(xticks_final)
    ax3.set_xticklabels(xlabels_final)
else:
    ax3.set_xlim(1.5, len(gap_ratios) + 1.5)
y_min, y_max = ax3.get_ylim()
ax3.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
plt.tight_layout()

#plt.savefig('eigenanalysis_visualization.png', dpi=300, bbox_inches='tight', 
#           facecolor='white', edgecolor='none')

plt.show()