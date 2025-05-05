import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 创建结果目录
os.makedirs('result', exist_ok=True)
data_path = '../../1.1_data_overview/result/raw_data.csv'

# 读取数据
feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']
label_name = 'Species'
df = pd.read_csv(data_path)

# 1. 每个类别的特征分布统计（均值、标准差、最小值、最大值、分位数）
groups = df.groupby(label_name)
with open('result/class_feature_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'Feature', 'Mean', 'Std', 'Min', 'Max', 'Q1', 'Median', 'Q3'])
    for cls, group in groups:
        for name in feature_names:
            writer.writerow([
                cls, name,
                group[name].mean(),
                group[name].std(),
                group[name].min(),
                group[name].max(),
                group[name].quantile(0.25),
                group[name].quantile(0.5),
                group[name].quantile(0.75)
            ])

# 2. 分组直方图
for name in feature_names:
    plt.figure(figsize=(6,4))
    for cls in df[label_name].unique():
        plt.hist(df[df[label_name]==cls][name], bins=15, alpha=0.5, label=cls)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f'{name} Distribution by Class')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'result/{name}_class_hist.png')
    plt.close()

# 3. 箱线图
for name in feature_names:
    plt.figure(figsize=(6,4))
    sns.boxplot(x=label_name, y=name, data=df)
    plt.title(f'{name} Boxplot by Class')
    plt.tight_layout()
    plt.savefig(f'result/{name}_class_boxplot.png')
    plt.close()

# 组合分组直方图（2x2）
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, name in enumerate(feature_names):
    ax = axes[idx // 2, idx % 2]
    for cls in df[label_name].unique():
        ax.hist(df[df[label_name]==cls][name], bins=15, alpha=0.5, label=cls)
    ax.set_xlabel(name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} by Class')
    if idx == 0:
        ax.legend()
plt.tight_layout()
plt.savefig('result/all_features_class_hist.png')
plt.close()

# 组合箱线图（所有特征在一张图，x为特征，hue为类别）
plt.figure(figsize=(10, 6))
df_melt = df.melt(id_vars=label_name, value_vars=feature_names, var_name='Feature', value_name='Value')
sns.boxplot(x='Feature', y='Value', hue=label_name, data=df_melt)
plt.title('All Features Boxplot by Class')
plt.tight_layout()
plt.savefig('result/all_features_class_boxplot.png')
plt.close() 