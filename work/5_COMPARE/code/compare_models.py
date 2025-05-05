import os
import sys

# 优先导入科学包，若缺失则报错提示
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
except ImportError as e:
    print("Required package not found:", e)
    print("Please install the missing package and rerun this script.")
    sys.exit(1)

# 配色和风格
sns.set_style("whitegrid")
colors = ['#2E86C1', '#E74C3C']

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNN_METRICS = os.path.join(BASE_DIR, '../../3_KNN/result/knn_metrics.csv')
TREE_METRICS = os.path.join(BASE_DIR, '../../4_TREE/result/tree_metrics.csv')
KNN_CONF = os.path.join(BASE_DIR, '../../3_KNN/result/confusion_matrix.csv')
TREE_CONF = os.path.join(BASE_DIR, '../../4_TREE/result/confusion_matrix.csv')
OUT_DIR = os.path.join(BASE_DIR, '../result')
os.makedirs(OUT_DIR, exist_ok=True)


def read_metrics_csv(path):
    """Read metrics csv and return macro avg row as dict."""
    try:
        df = pd.read_csv(path)
        print(f"Loaded {path}, columns: {list(df.columns)}")
        macro_row = df[df['Class'].str.lower().str.contains('macro')].iloc[0]
        return {
            'Precision': float(macro_row['Precision']),
            'Recall': float(macro_row['Recall']),
            'Specificity': float(macro_row['Specificity'])
        }
    except Exception as e:
        print(f"Error reading metrics from {path}: {e}")
        sys.exit(1)

def read_confusion_csv(path):
    """Read confusion matrix csv and return DataFrame."""
    try:
        df = pd.read_csv(path, index_col=0)
        print(f"Loaded {path}, shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error reading confusion matrix from {path}: {e}")
        sys.exit(1)

def make_comparison_table(knn_macro, tree_macro):
    """Create comparison DataFrame for macro metrics, round to 2 decimals."""
    comparison = pd.DataFrame({
        'Model': ['KNN', 'Decision Tree'],
        'Macro Precision': [round(knn_macro['Precision'], 2), round(tree_macro['Precision'], 2)],
        'Macro Recall': [round(knn_macro['Recall'], 2), round(tree_macro['Recall'], 2)],
        'Macro Specificity': [round(knn_macro['Specificity'], 2), round(tree_macro['Specificity'], 2)]
    })
    return comparison

def plot_metrics_bar(comparison, out_path):
    """Plot bar chart for macro metrics comparison."""
    plt.figure(figsize=(8, 5))
    metrics = ['Macro Precision', 'Macro Recall', 'Macro Specificity']
    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, [comparison[m].iloc[0] for m in metrics], width, label='KNN', color=colors[0])
    plt.bar(x + width/2, [comparison[m].iloc[1] for m in metrics], width, label='Decision Tree', color=colors[1])
    plt.ylabel('Score')
    plt.title('Macro Metrics Comparison', fontsize=14, fontweight='bold')
    plt.xticks(x, metrics)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=2, frameon=False)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrices(knn_conf, tree_conf, out_path):
    """Plot side-by-side confusion matrices with different color maps and improved annotation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    sns.heatmap(knn_conf, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False, annot_kws={"size":14, "weight":"bold"})
    ax1.set_title('KNN Confusion Matrix', fontsize=13, fontweight='bold', color=colors[0])
    ax1.set_xlabel('Predicted', fontsize=11)
    ax1.set_ylabel('Actual', fontsize=11)
    ax1.tick_params(axis='both', labelsize=10)
    sns.heatmap(tree_conf, annot=True, fmt='d', cmap='Oranges', ax=ax2, cbar=False, annot_kws={"size":14, "weight":"bold"})
    ax2.set_title('Decision Tree Confusion Matrix', fontsize=13, fontweight='bold', color=colors[1])
    ax2.set_xlabel('Predicted', fontsize=11)
    ax2.set_ylabel('Actual', fontsize=11)
    ax2.tick_params(axis='both', labelsize=10)
    plt.suptitle('Confusion Matrices Comparison', fontsize=15, fontweight='bold', y=1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_table_image(comparison, out_path):
    """Save comparison table as image, round to 2 decimals."""
    plt.figure(figsize=(8, 2.5))
    plt.axis('off')
    # 格式化所有数值为两位小数
    cell_text = [[f'{v:.2f}' if isinstance(v, float) else v for v in row] for row in comparison.values]
    table = plt.table(cellText=cell_text,
                     colLabels=comparison.columns,
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    plt.title('Macro Metrics Comparison Table', fontsize=13, fontweight='bold', pad=10)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 读取宏平均指标
    knn_macro = read_metrics_csv(KNN_METRICS)
    tree_macro = read_metrics_csv(TREE_METRICS)
    # 读取混淆矩阵
    knn_conf = read_confusion_csv(KNN_CONF)
    tree_conf = read_confusion_csv(TREE_CONF)
    # 生成对比表
    comparison = make_comparison_table(knn_macro, tree_macro)
    # 输出csv
    comparison.to_csv(os.path.join(OUT_DIR, 'models_comparison.csv'), index=False)
    # 输出三线表图片
    save_table_image(comparison, os.path.join(OUT_DIR, 'models_comparison_table.png'))
    # 输出条形图
    plot_metrics_bar(comparison, os.path.join(OUT_DIR, 'metrics_comparison.png'))
    # 输出混淆矩阵对比图
    plot_confusion_matrices(knn_conf, tree_conf, os.path.join(OUT_DIR, 'confusion_matrices_comparison.png'))
    print('Model comparison finished! All results saved in result folder.')

if __name__ == '__main__':
    main() 