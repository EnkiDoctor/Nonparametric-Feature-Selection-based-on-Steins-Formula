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

# Set styling to match JRSSB (Times New Roman, clean layout)
plt.style.use('default')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',  # Best match for Times New Roman in math
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'lines.linewidth': 1.2,
    'lines.markersize': 5,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 5))

# Professional color palette (JRSSB acceptable colors)
# Using a high-contrast palette that looks good in color and converts well to grayscale
# Slightly darker colors for a more "printed journal" feel
c_line = '#003366'      # Navy Blue (Classic academic color)
c_marker = '#003366'    # Navy Blue
c_threshold = '#B22222' # Firebrick Red (Darker, more serious red)
c_connect = '#555555'   # Dark Grey
marker_style = 'o'

cutoff_point_gaps = 30
gap_indices_display = np.arange(1, min(cutoff_point_gaps, len(eigengaps)) + 1)
eigengaps_display = eigengaps[:min(cutoff_point_gaps, len(eigengaps))]

# Plot Eigengaps
ax2.plot(gap_indices_display, eigengaps_display, 
         color=c_line, linewidth=1.2, zorder=2)
ax2.scatter(gap_indices_display, eigengaps_display, 
           color=c_marker, s=15, zorder=3, marker=marker_style, facecolors='white', edgecolors=c_marker, linewidth=1.0)

# Add threshold line
ax2.axhline(y=tao, color=c_threshold, linestyle='--', linewidth=1.2, 
           alpha=0.8, zorder=1, label=rf'$\tau = {tao:.4f}$')

if len(eigengaps) > cutoff_point_gaps:
    compressed_99_pos = 36
    final_gap_value = eigengaps_display[-1]
    
    # Dotted connection to the "tail"
    x_connect = np.linspace(cutoff_point_gaps + 1, compressed_99_pos, 10) 
    y_connect = np.full_like(x_connect, final_gap_value)

    ax2.plot(x_connect, y_connect, color=c_connect, linestyle=':', 
             linewidth=1.0, alpha=0.6, zorder=1)
    ax2.scatter([compressed_99_pos], [final_gap_value], color=c_marker, s=15, 
               zorder=3, marker=marker_style, facecolors='white', edgecolors=c_marker, linewidth=1.0)

ax2.set_xlabel('Gap Index')
ax2.set_ylabel(r'Eigengap')
ax2.set_title(r'(a) Eigengaps $|\hat{\lambda}_{k}| - |\hat{\lambda}_{k+1}|$')

ax2.grid(True, axis='y', alpha=0.3, linestyle=':')
ax2.set_axisbelow(True)

if len(eigengaps) > cutoff_point_gaps:
    ax2.set_xlim(0, compressed_99_pos + 2)
    xticks_final = list(range(1, cutoff_point_gaps+1, 10)) + [compressed_99_pos]
    xlabels_final = [str(x) for x in range(1, cutoff_point_gaps+1, 10)] + [str(len(eigengaps))]
    ax2.set_xticks(xticks_final)
    ax2.set_xticklabels(xlabels_final)
else:
    ax2.set_xlim(0, len(eigengaps) + 1)

# Adjust y-limits slightly
y_min, y_max = ax2.get_ylim()
ax2.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

# Clean legend
legend = ax2.legend(loc='best', frameon=False)

# Gap Ratios Calculation
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

# Plot Gap Ratios
ax3.plot(ratio_indices_display, ratios_display, 
         color=c_line, linewidth=1.2, zorder=2)
ax3.scatter(ratio_indices_display, ratios_display, 
           color=c_marker, s=15, zorder=3, marker=marker_style, facecolors='white', edgecolors=c_marker, linewidth=1.0)

if len(gap_ratios) > cutoff_point_ratios:
    compressed_98_pos = 37
    final_ratio_value = ratios_display[-1] 
    x_connect = np.linspace(cutoff_point_ratios + 2, compressed_98_pos, 10)  
    y_connect = np.full_like(x_connect, final_ratio_value)
    
    ax3.plot(x_connect, y_connect, color=c_connect, linestyle=':', 
             linewidth=1.0, alpha=0.6, zorder=1)
    ax3.scatter([compressed_98_pos], [final_ratio_value], color=c_marker, s=15, 
               zorder=3, marker=marker_style, facecolors='white', edgecolors=c_marker, linewidth=1.0)

ax3.set_xlabel('Ratio Index')
ax3.set_ylabel(r'Gap Ratio')
ax3.set_title(r'(b) Gap Ratios $r(k) = \frac{|\hat{\lambda}_{k-1}| - |\hat{\lambda}_{k}|}{|\hat{\lambda}_{k}| - |\hat{\lambda}_{k+1}| + \gamma_{reg}}$')
# Note: Simplified the math in title for cleaner look, or use:
# ax3.set_title(r'(b) Gap Ratios $r(j)$')

ax3.grid(True, axis='y', alpha=0.3, linestyle=':')
ax3.set_axisbelow(True)

if len(gap_ratios) > cutoff_point_ratios:
    ax3.set_xlim(1, compressed_98_pos + 2)
    xticks_final = list(range(2, cutoff_point_ratios+2, 10)) + [compressed_98_pos]
    xlabels_final = [str(x) for x in range(2, cutoff_point_ratios+2, 10)] + [str(len(gap_ratios))]
    ax3.set_xticks(xticks_final)
    ax3.set_xticklabels(xlabels_final)
else:
    ax3.set_xlim(1, len(gap_ratios) + 2)

y_min, y_max = ax3.get_ylim()
ax3.set_ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))

plt.tight_layout()
plt.show()