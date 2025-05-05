import numpy as np
from collections import Counter

def euclidean_distance(x1, x2):
    """
    计算两个样本之间的欧氏距离
    :param x1: 第一个样本（特征向量）
    :param x2: 第二个样本（特征向量）
    :return: 欧氏距离
    """
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNNClassifier:
    """
    手写KNN分类器，支持欧氏距离和K值调参
    """
    def __init__(self, k=3):
        """
        初始化KNN分类器
        :param k: 最近邻个数
        """
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        训练KNN分类器（实际为记忆训练集）
        :param X: 训练集特征，shape=(n_samples, n_features)
        :param y: 训练集标签，shape=(n_samples,)
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        对测试集进行预测
        :param X: 测试集特征，shape=(n_samples, n_features)
        :return: 预测标签，shape=(n_samples,)
        """
        predictions = []
        for x in X:
            # 计算所有训练样本到当前测试样本x的欧氏距离
            distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
            # 取距离最近的k个训练样本的索引
            k_indices = np.argsort(distances)[:self.k]
            # 获取这k个邻居的标签
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            # 多数表决，统计出现最多的类别作为预测结果
            most_common = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(most_common)
        return np.array(predictions) 