import os
import csv
import numpy as np

os.makedirs('.', exist_ok=True)
data_path = '../1.2_data_preprocess/data.csv'
feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']

data = []
labels = []
with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) == 5:
            try:
                features = [float(x) for x in row[:4]]
                label = int(row[4])
                data.append(features)
                labels.append(label)
            except:
                continue

data = np.array(data)
labels = np.array(labels)

# 划分
np.random.seed(42)
indices = np.random.permutation(len(data))
test_size = int(len(data) * 0.3)
test_idx = indices[:test_size]
train_idx = indices[test_size:]

X_train, X_test = data[train_idx], data[test_idx]
y_train, y_test = labels[train_idx], labels[test_idx]

with open('train_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + ['Label'])
    for features, label in zip(X_train, y_train):
        writer.writerow(list(features) + [label])

with open('test_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + ['Label'])
    for features, label in zip(X_test, y_test):
        writer.writerow(list(features) + [label]) 