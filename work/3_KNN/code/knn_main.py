import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from knn_core import KNNClassifier
import pandas as pd
import seaborn as sns

# 创建结果文件夹
os.makedirs('../result', exist_ok=True)
# 训练集和测试集路径
train_path = '../../1.3_data_split/result/train_data.csv'
test_path = '../../1.3_data_split/result/test_data.csv'
feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']

# 读取csv数据为numpy数组
def load_data(path):
    """
    读取csv文件，返回特征和标签
    :param path: csv文件路径
    :return: X, y
    """
    data = []
    labels = []
    with open(path, 'r', encoding='utf-8') as f:
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
    return np.array(data), np.array(labels)

# 加载训练集和测试集
X_train, y_train = load_data(train_path)
X_test, y_test = load_data(test_path)

# K值调参与评估
k_list = list(range(1, 21))
acc_list = []
for k in k_list:
    clf = KNNClassifier(k=k)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    acc_list.append(acc)

# 保存K-准确率csv
with open('../result/k_acc_curve.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['K', 'Accuracy'])
    for k, acc in zip(k_list, acc_list):
        writer.writerow([k, acc])

# 绘制K-准确率曲线
plt.figure(figsize=(8,5))
plt.plot(k_list, acc_list, marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.title('K vs Accuracy (KNN)')
plt.grid(True)
plt.savefig('../result/k_acc_curve.png')
plt.close()

# 选择最优K，重新训练并评估
best_k = k_list[np.argmax(acc_list)]
best_acc = max(acc_list)
best_clf = KNNClassifier(k=best_k)
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)

# 计算混淆矩阵
cm = np.zeros((3,3), dtype=int)
for t, p in zip(y_test, y_pred):
    cm[t, p] += 1
with open('../result/confusion_matrix.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([''] + [0,1,2])
    for i, row in enumerate(cm):
        writer.writerow([i] + list(row))

# 混淆矩阵可视化（热图）
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.title(f'Confusion Matrix (K={best_k})')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../result/confusion_matrix_heatmap.png')
plt.close()

# 保存最优K和准确率
with open('../result/best_k_acc.txt', 'w', encoding='utf-8') as f:
    f.write(f'Best K: {best_k}\n')
    f.write(f'Best Accuracy: {best_acc:.4f}\n')

# 计算多分类评估指标（精确率、召回率、特异度）
n_class = 3
precision = []
recall = []
specificity = []
for i in range(n_class):
    TP = cm[i, i]
    FP = sum(cm[:, i]) - TP
    FN = sum(cm[i, :]) - TP
    TN = cm.sum() - (TP + FP + FN)
    p = TP / (TP + FP) if (TP + FP) > 0 else 0
    r = TP / (TP + FN) if (TP + FN) > 0 else 0
    s = TN / (TN + FP) if (TN + FP) > 0 else 0
    precision.append(p)
    recall.append(r)
    specificity.append(s)

macro_precision = np.mean(precision)
macro_recall = np.mean(recall)
macro_specificity = np.mean(specificity)

# 保存评估指标csv
with open('../result/knn_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'Precision', 'Recall', 'Specificity'])
    for i in range(n_class):
        writer.writerow([i, precision[i], recall[i], specificity[i]])
    writer.writerow(['Macro Avg', macro_precision, macro_recall, macro_specificity])

# 评估指标条形图
metrics_df = pd.read_csv('../result/knn_metrics.csv')
metrics_df = metrics_df[metrics_df['Class'] != 'Macro Avg']
plt.figure(figsize=(8,5))
bar_width = 0.2
x = np.arange(n_class)
plt.bar(x-bar_width, metrics_df['Precision'], width=bar_width, label='Precision')
plt.bar(x, metrics_df['Recall'], width=bar_width, label='Recall')
plt.bar(x+bar_width, metrics_df['Specificity'], width=bar_width, label='Specificity')
plt.xticks(x, [f'Class {i}' for i in range(n_class)])
plt.ylim(0, 1.1)
plt.ylabel('Score')
plt.title('KNN Evaluation Metrics by Class')
plt.legend()
plt.tight_layout()
plt.savefig('../result/knn_metrics_bar.png')
plt.close()

# 评估指标三线表图片
fig, ax = plt.subplots(figsize=(7,2))
table_data = metrics_df.round(3).values.tolist()
col_labels = ['Class', 'Precision', 'Recall', 'Specificity']
table = ax.table(cellText=table_data, colLabels=col_labels, loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)
ax.axis('off')
plt.title('KNN Metrics Table')
plt.tight_layout()
plt.savefig('../result/knn_metrics_table.png')
plt.close() 