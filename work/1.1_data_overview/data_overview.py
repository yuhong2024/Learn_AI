import os
import csv
import numpy as np

os.makedirs('result', exist_ok=True)
data_path = '../data/iris.csv'
feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
label_name = 'Species'

data = []
labels = []
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines[1:]:
    if not line.strip():
        continue
    parts = [x.strip('"') for x in line.strip().split(',')]
    if len(parts) == 6:
        try:
            features = [float(x) for x in parts[1:5]]
            label = parts[5]
            data.append(features)
            labels.append(label)
        except:
            continue

data = np.array(data)
labels = np.array(labels)

# 保存原始数据csv
with open('result/raw_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + [label_name])
    for features, label in zip(data, labels):
        writer.writerow(list(features) + [label])

# 统计信息
stats = []
for i, name in enumerate(feature_names):
    stats.append([name, np.mean(data[:, i]), np.std(data[:, i]), np.min(data[:, i]), np.max(data[:, i])])

with open('result/feature_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Feature', 'Mean', 'Std', 'Min', 'Max'])
    writer.writerows(stats)

unique, counts = np.unique(labels, return_counts=True)
with open('result/class_distribution.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'Count'])
    for u, c in zip(unique, counts):
        writer.writerow([u, c]) 