import numpy as np
from collections import Counter

class TreeNode:
    """
    决策树节点类
    """
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini  # 当前节点的基尼指数
        self.num_samples = num_samples  # 当前节点的样本数
        self.num_samples_per_class = num_samples_per_class  # 每个类别的样本数
        self.predicted_class = predicted_class  # 当前节点的预测类别
        self.feature_index = None  # 最优分裂特征索引
        self.threshold = None  # 最优分裂阈值
        self.left = None  # 左子树
        self.right = None  # 右子树

class DecisionTreeClassifier:
    """
    手写决策树分类器，支持基尼指数分裂
    """
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.n_classes_ = None
        self.n_features_ = None
        self.tree_ = None

    def fit(self, X, y):
        """
        训练决策树分类器
        :param X: 训练集特征，shape=(n_samples, n_features)
        :param y: 训练集标签，shape=(n_samples,)
        """
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        """
        对测试集进行预测
        :param X: 测试集特征，shape=(n_samples, n_features)
        :return: 预测标签，shape=(n_samples,)
        """
        return np.array([self._predict(inputs) for inputs in X])

    def _gini(self, y):
        """
        计算基尼指数
        """
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def _best_split(self, X, y):
        """
        寻找最优分裂特征和阈值
        """
        m, n = X.shape
        if m <= 1:
            return None, None
        best_gini = 1.0
        best_idx, best_thr = None, None
        for idx in range(n):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
            num_left = [0] * self.n_classes_
            num_right = Counter(classes)
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_) if i > 0)
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_) if (m - i) > 0)
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr

    def _grow_tree(self, X, y, depth=0):
        """
        递归生长决策树
        """
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = TreeNode(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )
        # 停止条件：纯节点或达到最大深度
        if depth < (self.max_depth if self.max_depth is not None else np.inf):
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node

    def _predict(self, inputs):
        """
        单样本预测
        """
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

def print_tree(node, feature_names, depth=0):
    """递归打印决策树结构，便于可解释性分析"""
    indent = "    " * depth
    if node.left is None and node.right is None:
        label_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        print(f"{indent}类别: {label_map.get(node.predicted_class, node.predicted_class)}")
    else:
        feature = feature_names[node.feature_index]
        print(f"{indent}{feature} <= {node.threshold}?")
        print(f"{indent}├─ 是 →", end=" ")
        print_tree(node.left, feature_names, depth + 1)
        print(f"{indent}└─ 否 →", end=" ")
        print_tree(node.right, feature_names, depth + 1) 