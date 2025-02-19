import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

if len(sys.argv) != 2:
    print("Usage: python drawer.py <csv_file>")
    sys.exit(1)

# Read the CSV file
df = pd.read_csv(sys.argv[1])

# Calculate speedup and its uncertainty using error propagation
df['speedup'] = df['qiskit_time'] / df['cpp_time']
df['speedup_std'] = df['speedup'] * np.sqrt(
    (df['qiskit_std']/df['qiskit_time'])**2 + 
    (df['cpp_std']/df['cpp_time'])**2
)

# Check if any error is above threshold
error_threshold = 1e-9
if (df['error'] > error_threshold).any():
    print(f"Warning: Some errors are above {error_threshold}")
    print(df[df['error'] > error_threshold])

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Function to plot with aggregation and error bands
def plot_with_errorband(ax, x_data, y_data, yerr_data, xlabel):
    # Aggregate data by x value
    grouped = pd.DataFrame({'x': x_data, 'y': y_data, 'yerr': yerr_data}).groupby('x').agg({
        'y': 'mean',
        'yerr': 'mean'
    }).reset_index()
    
    # Sort by x for proper line plotting
    grouped = grouped.sort_values('x')
    
    # Plot line and points
    ax.plot(grouped['x'], grouped['y'], '-', color='blue', zorder=2)
    ax.scatter(grouped['x'], grouped['y'], color='blue', zorder=3)
    
    # Plot error band
    ax.fill_between(grouped['x'], 
                    grouped['y'] - grouped['yerr'],
                    grouped['y'] + grouped['yerr'],
                    alpha=0.2, color='blue', zorder=1)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Speedup (Qiskit/C++)')
    ax.grid(True)

# Plot speedup vs depth
plot_with_errorband(ax1, df['depth'], df['speedup'], df['speedup_std'], 'Circuit Depth')
ax1.set_title('Speedup vs Circuit Depth')

# Plot speedup vs number of qubits
plot_with_errorband(ax2, df['num_qubits'], df['speedup'], df['speedup_std'], 'Number of Qubits')
ax2.set_title('Speedup vs Number of Qubits')

plt.tight_layout()
plt.savefig('speedup.png')