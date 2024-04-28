import matplotlib.pyplot as plt

# Data
data = [
    [3, 2], [7, 11], [6, 13], [6, 13], [2, 24], [4, 3], [1, 14], [1, 4], [5, 10], [8, 5],
    [8, 10], [1, 8], [8, 5], [8, 8], [3, 5], [8, 12], [16, 39], [9, 5], [4, 14], [5, 15],
    [7, 21], [16, 5], [6, 18], [11, 8], [4, 15], [7, 9], [6, 21], [3, 10], [7, 9], [4, 5],
    [5, 6], [7, 13], [24, 14], [11, 15], [4, 6], [4, 13], [11, 11], [12, 17], [7, 4], [8, 3],
    [12, 6], [14, 4], [3, 16], [7, 12], [14, 3], [10, 11], [5, 5], [8, 5], [7, 4], [12, 10],
    [9, 26], [6, 21], [7, 8], [9, 7], [12, 19], [3, 10], [2, 13], [5, 18], [5, 17], [12, 8],
    [5, 6], [13, 18], [10, 17], [9, 14], [10, 3], [2, 4], [12, 11], [5, 4], [6, 32], [3, 6],
    [6, 5], [3, 5], [3, 23], [2, 28], [5, 2], [8, 9], [12, 9], [6, 1], [13, 3], [3, 7],
    [8, 4], [4, 6], [8, 8], [5, 6], [5, 17], [15, 4], [6, 20], [5, 9], [11, 25], [27, 21],
    [8, 8], [5, 7], [5, 23], [16, 4], [3, 4], [6, 6], [3, 11], [3, 16], [9, 5], [5, 21]
]

# Extracting counts
count1 = [d[0] for d in data]
count2 = [d[1] for d in data]

# Create histograms
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(count1, bins=20, color='blue', alpha=0.7)
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('DQN Scores over 100 games')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.hist(count2, bins=20, color='green', alpha=0.7)
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.title('Double-DQN Scores over 100 games')
plt.grid(True)

plt.tight_layout()
plt.show()