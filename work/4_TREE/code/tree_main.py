import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tree_core import DecisionTreeClassifier, print_tree

# 创建结果文件夹
os.makedirs('../result', exist_ok=True)
train_path = '../../1.3_data_split/result/train_data.csv'
test_path = '../../1.3_data_split/result/test_data.csv'
feature_names = ['Sepal.Length', 'Sepal.Width', 'Petal.Length', 'Petal.Width']

# 读取csv数据为numpy数组
def load_data(path):
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

# 训练决策树（推荐max_depth=2）
max_depth = 2  # 推荐值，实验表明此时模型性能最佳
clf = DecisionTreeClassifier(max_depth=max_depth)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 准确率
accuracy = np.mean(y_pred == y_test)
with open('../result/accuracy.txt', 'w', encoding='utf-8') as f:
    f.write(f'Accuracy: {accuracy:.4f}\n')

# 混淆矩阵
n_class = 3
cm = np.zeros((n_class, n_class), dtype=int)
for t, p in zip(y_test, y_pred):
    cm[t, p] += 1
with open('../result/confusion_matrix.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow([''] + [0,1,2])
    for i, row in enumerate(cm):
        writer.writerow([i] + list(row))

# 混淆矩阵热图
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=[0,1,2], yticklabels=[0,1,2])
plt.title('Confusion Matrix (Decision Tree)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('../result/confusion_matrix_heatmap.png')
plt.close()

# 评估指标（精确率、召回率、特异度）
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

with open('../result/tree_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Class', 'Precision', 'Recall', 'Specificity'])
    for i in range(n_class):
        writer.writerow([i, precision[i], recall[i], specificity[i]])
    writer.writerow(['Macro Avg', macro_precision, macro_recall, macro_specificity])

# 评估指标条形图
metrics_df = pd.read_csv('../result/tree_metrics.csv')
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
plt.title('Decision Tree Evaluation Metrics by Class')
plt.legend()
plt.tight_layout()
plt.savefig('../result/tree_metrics_bar.png')
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
plt.title('Decision Tree Metrics Table')
plt.tight_layout()
plt.savefig('../result/tree_metrics_table.png')
plt.close()

# ==========================
# 批量训练不同max_depth下的决策树，输出对比结果
# ==========================
max_depth_list = [1, 2, 3, 4, 5, None]
results = []
for max_depth in max_depth_list:
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    # 混淆矩阵
    cm = np.zeros((n_class, n_class), dtype=int)
    for t, p in zip(y_test, y_pred):
        cm[t, p] += 1
    # 评估指标
    precision, recall, specificity = [], [], []
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
    results.append([max_depth if max_depth is not None else 'None', accuracy, macro_precision, macro_recall, macro_specificity])

with open('../result/depth_metrics.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['max_depth', 'Accuracy', 'Macro_Precision', 'Macro_Recall', 'Macro_Specificity'])
    for row in results:
        writer.writerow(row)

# 可视化准确率随max_depth变化
depth_labels = [str(d) for d in max_depth_list]
accs = [row[1] for row in results]
plt.figure(figsize=(8,5))
plt.plot(depth_labels, accs, marker='o')
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Decision Tree Accuracy vs max_depth')
plt.grid(True)
plt.savefig('../result/depth_acc_curve.png')
plt.close()

# 打印并保存决策树结构
print("决策树结构：")
import sys
from io import StringIO
old_stdout = sys.stdout
sys.stdout = mystdout = StringIO()
print_tree(clf.tree_, feature_names)
sys.stdout = old_stdout
print(mystdout.getvalue())
with open('../result/tree_structure.txt', 'w', encoding='utf-8') as f:
    f.write(mystdout.getvalue()) 