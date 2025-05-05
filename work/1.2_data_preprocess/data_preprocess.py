import os
import csv
import numpy as np

os.makedirs('result', exist_ok=True)
data_path = '../1.1_data_overview/result/raw_data.csv'
feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
label_name = 'Species'
label_map = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

data = []
labels = []
with open(data_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if len(row) == 5:
            try:
                features = [float(x) for x in row[:4]]
                label = label_map[row[4]]
                data.append(features)
                labels.append(label)
            except:
                continue

data = np.array(data)
labels = np.array(labels)

# 标准化
means = np.mean(data, axis=0)
stds = np.std(data, axis=0)
data_std = (data - means) / stds

# 保存标准化后数据
with open('result/data.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(feature_names + ['Label'])
    for features, label in zip(data_std, labels):
        writer.writerow(list(features) + [label]) 