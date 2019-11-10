import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib
print(matplotlib.matplotlib_fname())


round1 = pd.read_csv('round1_merge_noDup.csv')
round2 = pd.read_csv('round2_train.csv')

round1_labels = round1.iloc[:, 4:]
round2_labels = round2.iloc[:, 4:]

columns = round1_labels.columns
round1_freq = round1_labels.sum() / len(round1_labels)
round2_freq = round2_labels.sum() / len(round2_labels)

plt.figure()
plt.bar(np.arange(len(columns)) - 0.2, round1_freq, width=0.4, label='Round1')
plt.bar(np.arange(len(columns)) + 0.2, round2_freq, width=0.4, label='Round2')
plt.xticks(np.arange(len(columns)), columns, rotation='vertical')
plt.legend()
plt.show()

round2_vs_round1 = round2_freq / round1_freq
# round2_vs_round1[0] = 0
# round2_vs_round1 /= np.max(round2_vs_round1)

plt.figure()
plt.bar(np.arange(len(columns)), round2_vs_round1, width=0.5)
plt.xticks(np.arange(len(columns)), columns, rotation='vertical')
plt.show()

# set QRS低电压 to 0
# round1['QRS低电压'] = 0

round1_labels_weights = round1_labels * round2_vs_round1

sample_weights = []
for i, row in round1_labels_weights.iterrows():
    row_list = row.tolist()
    row_list = [item for item in row_list if item != 0.0]
    sample_weight = np.prod(row_list)
    sample_weights.append(sample_weight)

sample_weights.sort()
sample_weights = np.array(sample_weights)
sample_weights[sample_weights > 1] = 1

plt.figure()
plt.plot(np.arange(len(sample_weights)), sample_weights)
plt.show()
