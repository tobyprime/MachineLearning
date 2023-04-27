from copy import copy
from typing import TypeVar, Any

import numpy as np


def get_subsets_decision(dataset: np.ndarray, feature_idx: int, value) -> (np.ndarray, np.ndarray):
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


def get_subset_reg(dataset: np.ndarray, feature_idx: int, spilt_value):
    p_subset = []
    n_subset = []
    for sample in dataset:
        # 去掉已经遍历过的特征，并加进新集合
        if sample[feature_idx] < spilt_value:
            p_subset.append(list(sample)[:feature_idx] + list(sample)[feature_idx + 1:])
        else:
            n_subset.append(list(sample)[:feature_idx] + list(sample)[feature_idx + 1:])
    return np.array(p_subset), np.array(n_subset)


def prob(dataset: np.ndarray, sample: np.ndarray) -> float:
    """
    计算数据集中某个样本的占比（概率）
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
    positive_subset, negative_subset = get_subsets_decision(dataset, feature_idx, value)
    return len(positive_subset) / len(dataset) * gini(positive_subset) + len(negative_subset) / len(dataset) * gini(
        negative_subset)


def dataset_mse_loss(dataset: np.ndarray):
    """
    计算集合内的 MSE 损失，（将集合所有类别的平均值作为预测值）
    """
    mean = dataset[-1].mean()
    loss = 0
    for sample in dataset:
        loss += (sample[-1] - mean) ** 2
    return loss


def get_best_seg_point_decision(dataset: np.ndarray, labels: list[str]) -> (str, Any):
    """
    获取最优的分类划分点
    :param dataset: 数据集
    :param labels: 标签
    :return: 特征以及最优分割值
    """
    # 选择最好的划分特征的label
    best_gini_value = 1
    best_feature_idx = None
    best_value = None
    for i in range(len(labels)):
        for feature_value in dataset[i, :-1]:
            gini_value = cond_gini(dataset, i, feature_value)
            if gini_value < best_gini_value:
                best_feature_idx = i
                best_value = feature_value

    return labels[best_feature_idx], best_value


def get_dataset_spilt_value_decision(dataset: np.ndarray, feature_idx):
    """
    计算给定集合在某个连续特征下所有可能的划分点
    """
    values = dataset[:, feature_idx]
    values.sort()
    sections = np.concatenate(([values[:-1]], [values[1:]]), axis=0)
    spilt_points = sections.mean(axis=-1)
    return spilt_points


def get_dataset_spilt_value_reg(dataset: np.ndarray, feature_idx):
    """
    计算给定集合在某个连续特征下所有可能的划分点
    """
    split_values = get_dataset_spilt_value_decision(dataset, feature_idx)
    best_loss = np.inf
    best_value = 0
    for value in split_values:
        positive_subset, negative_subset = get_subset_reg(dataset, feature_idx, value)  # 按最优划分点划分为两个子集
        loss = dataset_mse_loss(positive_subset) + dataset_mse_loss(negative_subset)
        if loss < best_loss:
            best_loss = loss
            best_value = value
    return best_value, best_loss


def get_dataset_spilt_point_reg(dataset: np.ndarray, labels: list):
    """
    计算给定集合在某个连续特征下所有可能的划分点
    """
    best_loss = np.inf
    best_feature_idx = 0
    best_value = 0
    for i in range(len(labels)):
        split_value, loss = get_dataset_spilt_value_reg(dataset, i)
        if loss < best_loss:
            best_loss = loss
            best_value = split_value
            best_feature_idx = i
    return labels[best_feature_idx], best_value


def create_reg_tree(dataset, labels: list, min_setsize=0) -> dict[str, dict | Any]:
    labels = copy(labels)
    classes = dataset[:, -1]

    if len(labels) == 0:  # 达到预剪枝的最小子集条件，默认是 0
        return dataset[-1].mean()  # 返回出现次数最多的类别

    # 选择最优划分点
    best_feature, best_value = get_dataset_spilt_point_reg(dataset, labels)
    best_feature_idx = labels.index(best_feature)

    decision_tree = {
        best_feature: {
            best_value: {},
            '@else': {}
        }
    }

    # 构建树
    del (labels[best_feature_idx])  # 删除已经访问过的特征
    positive_subset, negative_subset = get_subset_reg(dataset, best_feature_idx, best_value)  # 按最优划分点划分为两个子集

    decision_tree[best_feature][best_value] = create_reg_tree(positive_subset, copy(labels))
    decision_tree[best_feature]['@else'] = create_reg_tree(negative_subset, copy(labels))

    return decision_tree


def create_decision_tree(dataset, labels: list, min_setsize=0) -> dict[str, dict | Any]:
    labels = copy(labels)
    classes = dataset[:, -1]

    if len(set(classes)) == 1:  # 集合内所有元素类别相同
        return classes[0]  # 终止递归，返回类别
    if len(labels) == 0 or len(dataset) <= min_setsize:  # 达到预剪枝的最小子集条件，默认是 0
        return max(list(classes), key=list(classes).count)  # 返回出现次数最多的类别

    # 选择最优划分点
    best_feature, best_value = get_best_seg_point_decision(dataset, labels)
    best_feature_idx = labels.index(best_feature)

    decision_tree = {
        best_feature: {
            best_value: {},
            '@else': {}
        }
    }

    # 构建树
    del (labels[best_feature_idx])  # 删除已经访问过的特征
    positive_subset, negative_subset = get_subsets_decision(dataset, best_feature_idx, best_value)  # 按最优划分点划分为两个子集

    decision_tree[best_feature][best_value] = create_decision_tree(positive_subset, copy(labels))
    decision_tree[best_feature]['@else'] = create_decision_tree(negative_subset, copy(labels))

    return decision_tree


def pred_decision(tree: dict[str, dict | Any], sample: np.ndarray, labels: list):
    assert len(tree.items()) == 1
    feature, tree = [*tree.items()][0]
    feature_idx = labels.index(feature)
    value = sample[feature_idx]
    if value in tree.keys():
        if isinstance(tree[value], dict):
            return pred_decision(tree[value], sample, labels)
        else:
            return tree[value]
    else:
        if isinstance(tree['@else'], dict):
            return pred_decision(tree['@else'], sample, labels)
        else:
            return tree['@else']


def test_decision():
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
    labels = ['age', 'work', 'house', 'credit']

    tree = create_decision_tree(np.array(dataset, dtype=str), labels)
    true_count = 0
    for sample in dataset:
        if sample[-1] == pred_decision(tree, sample, labels):
            true_count += 1
    print(true_count / len(dataset))


def test_reg():
    dataset = np.array([[1, 1], [2, 1], [3, 1], [100, 0], [110, 0]])
    labels = ['value']

    tree = create_reg_tree(dataset,labels)
    print(tree)

if __name__ == '__main__':
    test_reg()
    test_decision()
