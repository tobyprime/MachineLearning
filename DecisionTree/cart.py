from copy import copy
from typing import TypeVar, Any

import numpy as np


def get_subsets(dataset: np.ndarray, feature_idx: int, value) -> (np.ndarray, np.ndarray):
    """
    根据给定特征的值，划分为等于与不等于的两个子集合
    """
    p_subset = []
    n_subset = []
    for sample in dataset:
        # 去掉已经遍历过的特征，并加进新集合
        if sample[feature_idx] == value:
            p_subset.append(list(sample)[:feature_idx] + list(sample)[feature_idx + 1:])
        else:
            n_subset.append(list(sample)[:feature_idx] + list(sample)[feature_idx + 1:])
    return np.array(p_subset), np.array(n_subset)


def prob(dataset: np.ndarray, sample: np.ndarray) -> float:
    """
    计算数据集（随机变量）中某个样本（随机变量的值）的占比（概率）
    :param dataset: 数据集（随机变量）
    :param sample: 样本（随机变量的值）
    :return: 占比（概率）
    """
    return len(dataset) / list(dataset[:, -1]).count(list(sample)[-1])


def gini(dataset):
    acc = 0
    for sample in dataset:
        p = prob(dataset, sample)
        acc += 1 - p ** 2
    return acc


def cond_gini(dataset, feature_idx: int, value) -> float:
    """
    计算按照给定特征值二分后的基尼系数
    """
    positive_subset, negative_subset = get_subsets(dataset, feature_idx, value)
    return len(positive_subset) / len(dataset) * gini(positive_subset) + len(negative_subset) / len(dataset) * gini(
        negative_subset)


def get_best_seg_point(dataset: np.ndarray, labels: list[str]) -> (str, Any):
    """
    获取最优的划分点
    :param dataset: 数据集
    :param labels: 标签
    :return: 特征以及最优分割值
    """
    # 选择最好的划分特征的label
    best_gini_value = 1
    best_feature = None
    best_value = None
    for i in range(len(labels)):
        for feature_value in dataset[i, :-1]:
            gini_value = cond_gini(dataset, i, feature_value)
            if gini_value < best_gini_value:
                best_feature = labels[i]
                best_value = feature_value

    return best_feature, best_value


def create_decision_tree(dataset, labels: list) -> dict[str, dict | Any]:
    classes = dataset[:, -1]

    if len(set(classes)) == 1:  # 集合内所有元素类别相同
        return classes[0]  # 终止递归，返回类别
    if len(labels) == 0:  # 所有特征都遍历完了
        return max(list(classes), key=list(classes).count)  # 返回出现次数最多的类别

    # 选择最优划分点
    best_feature, best_value = get_best_seg_point(dataset, labels)
    best_feature_idx = labels.index(best_feature)

    decision_tree = {
        best_feature: {},
    }

    # 构建树
    del (labels[best_feature_idx])  # 删除已经访问过的特征
    positive_subset, negative_subset = get_subsets(dataset, best_feature_idx, best_value)  # 按最优划分点划分为两个子集

    decision_tree[best_feature][best_value] = create_decision_tree(positive_subset, copy(labels))
    decision_tree[best_feature]['others'] = create_decision_tree(negative_subset, copy(labels))

    return decision_tree


def test():
    dataset = [['youth', 'no', 'no', 'just so-so', 'no'],
               ['youth', 'no', 'no', 'good', 'no'],
               ['youth', 'yes', 'no', 'good', 'yes'],
               ['youth', 'yes', 'yes', 'just so-so', 'yes'],
               ['youth', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'just so-so', 'no'],
               ['midlife', 'no', 'no', 'good', 'no'],
               ['midlife', 'yes', 'yes', 'good', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['midlife', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'great', 'yes'],
               ['geriatric', 'no', 'yes', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'good', 'yes'],
               ['geriatric', 'yes', 'no', 'great', 'yes'],
               ['geriatric', 'no', 'no', 'just so-so', 'no']]
    features = ['age', 'work', 'house', 'credit']

    print(create_decision_tree(np.array(dataset, dtype=str), features))


if __name__ == '__main__':
    test()
