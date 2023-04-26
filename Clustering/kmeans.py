import random

import numpy as np


def distance(a: np.ndarray, b: np.ndarray):
    """
    计算两个向量的欧式距离
    """
    return np.sqrt(np.sum((a - b) ** 2))


def get_prototypes(dataset: np.ndarray, k: int):
    # 从数据集中随机无放回的随机采样 k 个样本作为原型
    rand_idx = random.sample(range(len(dataset)), k)
    prototypes = [dataset[idx] for idx in rand_idx]

    # 迭代下降
    while True:
        clusters: list[list[np.ndarray]] = [[] for _ in range(k)]  # k个簇，初始为空

        # step: 计算簇
        for sample in dataset:
            d = [distance(sample, prototypes[j]) for j in range(k)]  # 计算该样本与各原型的距离
            idx = d.index(min(d))  # 取得与该样本距离最小的原型下标
            clusters[idx].append(sample)  # 将该样本放入距离最小原型对应的簇中

        # step: 更新原型
        updated = False
        for i in range(k):
            prototype: np.ndarray = np.average(clusters[i], axis=0)  # 计算新的原型
            if all(prototype != prototypes[i]):  # 如果原型有变化
                updated = True
                prototypes[i] = prototype  # 更新原型
        print(prototypes)
        if not updated:  # 如果所有原型都没有变化
            return prototypes  # 如果原型均没有更新，则说明收敛。


def test():
    print(get_prototypes(np.array([[100, 100], [50, 50], [150, 150], [-100, -100], [-50, -50], [-150, -150]]),
                         k=2))


if __name__ == '__main__':
    test()