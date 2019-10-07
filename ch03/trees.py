from math import log
import operator
import matplotlib


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    # change to discrete values
    return data_set, labels


def calc_shannon_ent(data_set):
    num_entries = len(data_set)  # 数据集中的元素个数
    label_counts = {}  # 将同类标签中的数据统计
    for feat_vec in data_set:
        current_label = feat_vec[-1]  # 获取标签
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1
    shannon_ent = 0.0
    for key in label_counts:
        prob = float(label_counts[key]) / num_entries  # 概率P(x_i)
        shannon_ent -= prob * log(prob, 2)  # 计算香农熵
    return shannon_ent


if __name__ == '__main__':
    data_set, labels = create_data_set()
    shannon_ent = calc_shannon_ent(data_set)
    print(shannon_ent)
