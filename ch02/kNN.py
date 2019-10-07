import numpy as np


def create_data_set():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify_self(input_data, data_set: np.ndarray, label_list: np.ndarray, k: int):
    # 获取数据集的数量大小
    data_set_size = data_set.shape[0]
    # 将数据复制成与数据集相同的大小 容易相减
    input_data_format = np.tile(input_data, (data_set_size, 1))
    # 获取与数据集之间的差距
    diff_mat = input_data_format - data_set
    # 欧式距离计算
    diff_square = np.square(diff_mat)  # x^2 y^2
    # print(diff_square)
    diff_square_add = np.sum(diff_square, axis=1)  # 相加
    # print(diff_square_add)
    distance = np.sqrt(diff_square_add)  # 开方
    distance_arg_sort = distance.argsort()  # 按照数据所在的位置排序
    # print(distance_arg_sort)
    type_count = {}
    for i in range(k):
        key_label = label_list[distance_arg_sort[i]]
        type_count[key_label] = type_count.get(key_label, 0) + 1
    type_count_zip = zip(type_count.values(), type_count.keys())
    res = sorted(type_count_zip, reverse=True)
    # print(res[0][1])
    return res[0][1]
