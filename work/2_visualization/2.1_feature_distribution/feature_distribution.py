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

# 1. 每个特征的分布统计（均值、标准差、最小值、最大值、分位数）
stats = []
for name in feature_names:
    stats.append([
        name,
        df[name].mean(),
        df[name].std(),
        df[name].min(),
        df[name].max(),
        df[name].quantile(0.25),
        df[name].quantile(0.5),
        df[name].quantile(0.75)
    ])
with open('result/feature_distribution_stats.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Feature', 'Mean', 'Std', 'Min', 'Max', 'Q1', 'Median', 'Q3'])
    writer.writerows(stats)

# 2. 每个特征的直方图
for name in feature_names:
    plt.figure(figsize=(6,4))
    plt.hist(df[name], bins=20, color='#4C72B0', alpha=0.7)
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.title(f'{name} Distribution')
    plt.tight_layout()
    plt.savefig(f'result/{name}_hist.png')
    plt.close()

# 3. 相关性矩阵csv和热图
corr = df[feature_names].corr()
corr.to_csv('result/feature_correlation.csv')
plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('result/feature_correlation_heatmap.png')
plt.close()

# 组合直方图（2x2）
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for idx, name in enumerate(feature_names):
    ax = axes[idx // 2, idx % 2]
    ax.hist(df[name], bins=20, color='#4C72B0', alpha=0.7)
    ax.set_xlabel(name)
    ax.set_ylabel('Frequency')
    ax.set_title(f'{name} Distribution')
plt.tight_layout()
plt.savefig('result/all_features_hist.png')
plt.close()

# 特征散点图矩阵（pairplot，按类别分色，图注英文）
sns.pairplot(df, vars=feature_names, hue='Species', palette='Set1', diag_kind='hist')
plt.suptitle('Feature Pairplot', y=1.02, fontsize=16)
plt.tight_layout()
plt.savefig('result/feature_pairplot.png')
plt.close() 