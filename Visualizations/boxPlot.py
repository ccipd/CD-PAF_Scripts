
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from the two CSV files
csv_file1 = 'Enter file path ' 
csv_file2 = 'Enter file path' 
data1 = pd.read_csv(csv_file1)
data2 = pd.read_csv(csv_file2)

# Mapping from numeric labels to descriptive labels
label_mapping = {1: 'Fistula', 2: 'ISM', 3: 'ESM'}

# Replace numeric labels with descriptive labels in both datasets
data1['label'] = data1['label'].replace(label_mapping)
data2['label'] = data2['label'].replace(label_mapping)

# Add a column to each DataFrame to indicate the source
data1['Source'] = 'No Residual Encoder'
data2['Source'] = 'Residual Encoder'

# Concatenate the data into a single DataFrame
data = pd.concat([data1, data2], ignore_index=True)

# Preview the first few rows of the combined data
print(data.head())

# Set the column names
label_column = 'label'  # Replace with your label column name
score_column = 'precision'  # Replace with your dice score column name
source_column = 'Source'  # This indicates the source dataset

# Create the box plot using seaborn
plt.figure(figsize=(14, 8))

# Create a box plot with different border colors for each dataset
sns.boxplot(x=label_column, y=score_column, hue=source_column, data=data, palette="Set2", linewidth=2.5)

# Add a title and labels
plt.xlabel('Label', fontsize=16)
plt.ylabel('Precision Score', fontsize=16)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add a custom legend at the top
plt.legend(title='Dataset Source', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

# Show the plot
plt.tight_layout()
plt.show()
