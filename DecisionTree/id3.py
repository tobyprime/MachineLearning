from math import log

import numpy as np
from numpy import ndarray


def prob(dataset: ndarray, sample: ndarray) -> float:
    """
    计算数据集（随机变量）中某个样本（随机变量的值）的占比（概率）
    :param dataset: 数据集（随机变量）
    :param sample: 样本（随机变量的值）
    :return: 占比（概率）
    """
    return len(dataset) / list(dataset[:, -1]).count(list(sample)[-1])


def info_ent(dataset: ndarray) -> float:
    """
    计算数据集的信息熵（香农熵）
    """
    ent = 0
    for sample in dataset:
        p = prob(dataset, sample)
        ent -= p * log(p, 2)
    return ent


def cond_info_ent(dataset: ndarray, feature_idx: int) -> float:
    """
    计算数据集在给定特征下的交叉熵
    """
    feature_values = set(dataset[:, feature_idx])
    ent = 0
    for feature_value in feature_values:
        sub_set = np.array([sample for sample in dataset if sample[feature_idx] == feature_value])
        p = len(sub_set) / len(dataset)
        ent += p * info_ent(sub_set)
    return ent


def get_subset(dataset: ndarray, feature_idx: int, value) -> ndarray:
    """
    获取某个特征等于给定值下的子数据集
    :param dataset: 数据集
    :param feature_idx: 特征下标
    :param value: 特征所对应的值
    :return: 特征的值与给定值相等的子集
    """
    new_dataset = []
    for sample in dataset:
        if sample[feature_idx] == value:
            # 去掉已经遍历过的特征，并加进新集合
            new_dataset.append(list(sample)[:feature_idx] + list(sample)[feature_idx + 1:])
    return np.array(new_dataset)


def get_best_seg_feature(dataset: ndarray, labels: list[str]) -> str:
    """
    获取最优的划分特征
    :param dataset: 数据集
    :param labels: 标签
    :return: 最优的特征对应的标签
    """
    # 选择最好的划分特征的label
    ent = info_ent(dataset)
    gains = []
    for i in range(len(labels)):
        # 计算按 feature=i 划分下的条件熵，并计算信息增益
        cond_ent = cond_info_ent(dataset, i)
        gain = ent - cond_ent
        gains.append(gain)
    best_feature = gains.index(max(gains))
    return labels[best_feature]


def create_tree(dataset: ndarray, labels: list[str]) -> dict:
    """
    创建树
    :param dataset: 数据集 [sample * n]: sample -> [feature_0, feature_1, ..., feature_m, cls]
    :param labels: 所有特征所对应的标签 [label_0,...,label_m]
    :return: 树
    {
        label_a: {
            label_b: pred_cls,
            ...,
            label_c: pred_cls
            },
        ...,
        label_d: pred_cls
    }
    """
    classes = dataset[:, -1]

    if len(set(classes)) == 1:  # 集合内所有元素类别相同
        return classes[0]  # 终止递归，返回类别
    if len(labels) == 0:  # 所有特征都遍历完了
        return max(list(classes), key=list(classes).count)  # 返回出现次数最多的类别

    # 选择最优分割特征
    best_label = get_best_seg_feature(dataset, labels)
    best_label_id = labels.index(best_label)

    # 初始化树
    tree = {best_label: {}}

    # 构建树
    del labels[best_label_id]  # 删除已经访问过的特征
    feature_value = set([sample[best_label_id] for sample in dataset])  # 最优划分特征所有可能的值（不重复）
    for value in feature_value:
        # 对按值分割后的子集合继续构造树
        subset = get_subset(dataset, best_label_id, value)
        tree[best_label][value] = create_tree(subset, labels)
    return tree


def test():
    """
    简易数据集：
    [sample * n]:sample -> [first, second, (first + second) 奇数为 1,偶数为 2]
    """

    dataset = [[1, 1, 2],
               [1, 2, 1],
               [2, 2, 2],
               [1, 3, 2],
               [2, 1, 1],
               [3, 1, 2],
               [3, 2, 1],
               [2, 3, 1],
               [3, 3, 1]]

    arr = np.array(dataset)

    print(create_tree(arr, ['first', 'second']))


if __name__ == '__main__':
    test()
