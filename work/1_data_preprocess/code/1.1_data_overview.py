import os
import csv
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
os.makedirs('.', exist_ok=True)
data_path = '../../data/iris.csv'
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
with open('raw_data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + [label_name])
    for features, label in zip(data, labels):
        writer.writerow(list(features) + [label])

# 统计信息
stats = []
for i, name in enumerate(feature_names):
    stats.append([name, np.mean(data[:, i]), np.std(data[:, i]), np.min(data[:, i]), np.max(data[:, i])])

# 保存特征统计csv
with open('feature_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Feature', 'Mean', 'Std', 'Min', 'Max'])
    writer.writerows(stats)

# 类别分布
unique, counts = np.unique(labels, return_counts=True)
with open('class_distribution.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'Count'])
    for u, c in zip(unique, counts):
        writer.writerow([u, c])

# 可视化：类别分布条形图
plt.figure(figsize=(6,4))
plt.bar(unique, counts, color=['#4C72B0', '#55A868', '#C44E52'])
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.close()

# 可视化：每个特征的直方图
for i, name in enumerate(feature_names):
    plt.figure(figsize=(6,4))
    for u in unique:
        plt.hist(data[labels==u, i], bins=15, alpha=0.5, label=u)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f'{name} Distribution by Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{name}_hist.png')
    plt.close() 