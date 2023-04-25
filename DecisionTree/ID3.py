from math import log

import numpy as np
from numpy import ndarray


def prob(dataset: ndarray, sample: ndarray):
    return len(dataset) / list(dataset[:, -1]).count(list(sample)[-1])


def info_ent(dataset: ndarray) -> float:  # 计算集合dataset的信息熵
    ent = 0
    for sample in dataset:
        p = prob(dataset, sample)
        ent -= p * log(p, 2)
    return ent


def cond_info_ent(dataset: ndarray, feature_id) -> float:  # 计算feature_id下的条件熵
    feature_values = set(dataset[:, feature_id])
    ent = 0
    for feature_value in feature_values:
        sub_set = np.array([sample for sample in dataset if sample[feature_id] == feature_value])
        p = len(sub_set) / len(dataset)
        ent += p * info_ent(sub_set)
    return ent


def split_set(dataset, label_id, value):
    new_dataset = []
    for sample in dataset:
        if sample[label_id] == value:
            # 去掉已经遍历过的特征，并加进新集合
            new_dataset.append(list(sample)[:label_id] + list(sample)[label_id + 1:])
    return np.array(new_dataset)


def get_best_feature_label(dataset, labels) -> str:
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


def create_tree(dataset, labels):
    classes = dataset[:, -1]
    if len(set(classes)) == 1:  # 集合内所有元素类别相同
        return classes[0]  # 终止递归，返回类别
    if len(labels) == 0:  # 所有特征都遍历完了
        return max(list(classes), key=list(classes).count)  # 返回出现次数最多的类别
    best_label = get_best_feature_label(dataset, labels)
    best_label_id = labels.index(best_label)
    tree = {best_label: {}}
    # 删除已经访问过的特征
    del labels[best_label_id]
    feature_value = set([sample[best_label_id] for sample in dataset])
    for value in feature_value:
        tree[best_label][value] = create_tree(split_set(dataset, best_label_id, value), labels)
    return tree


def test():
    # first + second = singular or plural
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