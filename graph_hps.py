import pandas as pd
import matplotlib.pyplot as plt

file_path = 'alpha_tests.xlsx'
df = pd.read_excel(file_path)

# plot histogram with different colors for each column
plt.figure(figsize=(10, 6))

for i, column in enumerate(df.columns):
    color = plt.cm.tab10(i / float(len(df.columns)))
    plt.hist(df[column], bins=20, alpha=0.7, color=color, label=column)

plt.xlabel('Score')
plt.ylabel('Frequency')
plt.title('Scores for Varied Alpha at Constrant Layer Sizes')
plt.legend()
plt.grid(True)
plt.show()
