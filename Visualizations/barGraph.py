import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ranksums

# Load the data from the CSV files
csv_file1 = 'Enter Path for file 1'
data1 = pd.read_csv(csv_file1)

# Load the data from the second CSV file
csv_file2 = 'Enter Path for file 2'
data2 = pd.read_csv(csv_file2)

# Set the column names
label_column = 'label'
score_column = 'hd'

# Mapping from numeric labels to descriptive labels - do according to your needs
label_mapping = {1: 'Fistula', 2: 'ISM', 3: 'ESM'}

# Replace numeric labels with descriptive labels in both datasets
data1[label_column] = data1[label_column].replace(label_mapping)
data2[label_column] = data2[label_column].replace(label_mapping)

# Extract unique labels and their indices
labels = data1[label_column].unique()
num_labels = len(labels)
label_positions = np.arange(num_labels)

# Calculate average scores and standard deviation (error bars) for each label in both datasets
average_scores1 = [data1[data1[label_column] == label][score_column].mean() for label in labels]
average_scores2 = [data2[data2[label_column] == label][score_column].mean() for label in labels]

error_bars1 = [data1[data1[label_column] == label][score_column].std() for label in labels]
error_bars2 = [data2[data2[label_column] == label][score_column].std() for label in labels]

# Create the figure
plt.figure(figsize=(12, 8))

# Define bar width and positions
bar_width = 0.35

# Plot the data from the first CSV file with error bars
plt.bar(label_positions - bar_width / 2, average_scores1, bar_width, yerr=error_bars1, capsize=3, label='nnU-Net', color='r')

# Plot the data from the second CSV file with error bars
plt.bar(label_positions + bar_width / 2, average_scores2, bar_width, yerr=error_bars2, capsize=3, label='MedSam', color='b')

# Xticks and labels with increased fontsize
plt.xticks(label_positions, labels, fontsize=24)

# Adding title and labels
plt.title('Bar Plot of Hausdorff Distance by Label with Error Bars', fontsize = 20)
plt.xlabel('Label', fontsize = 20)
plt.ylabel('Hausdorff Distance', fontsize = 20)
plt.ylim(bottom = 0)

# Perform Wilcoxon rank-sum test and add significance bars
for i, label in enumerate(labels):
    group1 = data1[data1[label_column] == label][score_column]
    group2 = data2[data2[label_column] == label][score_column]
    stat, p_value = ranksums(group1, group2)
    
    if p_value < 0.05:  # Significance level
        # Calculate the position of the asterisk
        y_max = max(average_scores1[i] + error_bars1[i], average_scores2[i] + error_bars2[i])
        plt.plot([i - bar_width / 2, i + bar_width / 2], [y_max + 0.02, y_max + 0.02], color='black')
        plt.text(i, y_max + 0.01, '*', ha='center', fontsize=24)

# Adding legend
plt.legend(fontsize=24)
# Display the plot
plt.tight_layout()
plt.show()
