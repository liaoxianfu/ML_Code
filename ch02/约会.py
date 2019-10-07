import numpy as np
import matplotlib.pyplot as plt


def file2matrix(file_name: str) -> (np.ndarray, np.ndarray):
    """
    文本数据转换成numpy的数据\n
    :param file_name: 文件名
    :return: 数据集和labels
    """
    with open(file_name) as f:
        # 获取数据集的数量
        file_len = len(f.readlines())
        return_mat = np.zeros((file_len, 3))  # 创建一个全为0的矩阵
        labels_list = []
    with open(file_name) as f:
        index = 0
        for line in f.readlines():
            line = line.strip()
            split = line.split("\t")
            return_mat[index] = split[0:3]
            labels_list.append(split[-1])
            index += 1
    return return_mat, np.array(labels_list)


def show_pic(return_mat: np.ndarray, labels_list: np.ndarray):
    """
    数据展示
    :param return_mat:
    :param labels_list:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels_list = labels_list.astype('float64')
    ax.scatter(return_mat[:, 1], return_mat[:, 2], 15.0 * labels_list, 15.0 * labels_list)
    plt.show()


def auto_norm(data_set: np.ndarray):
    """
    不同的参数之间的数据相差可能过大 例如玩游戏时间占比在0~100之间 而
    飞行距离则普遍超过1000 这两则的占比权重本身应该一致，但是如果不处理就会使得
    玩游戏时间占比的权重非常小，所以应该将所有的数据均匀的分布在0-1之间
    通常使用归一化的方法
    公式：
    new_value = (old_value-min)/(max-min)
    :param data_set:
    :return:
    """
    '''
    
    print(a.min()) #无参，所有中的最小值
    print(a.min(0)) # axis=0; 每列的最小值
    print(a.min(1)) # axis=1；每行的最小值
    '''
    min_value = data_set.min(0)  # 数据中每一列的最小值
    max_value = data_set.max(0)
    ranges = max_value - min_value  # 最大最小值得差距
    norm_data_set = (data_set - min_value) / ranges  # 归一化的公式
    return norm_data_set, min_value, ranges


def classify_knn(input_data: np.ndarray, data_set: np.ndarray, labels: np.ndarray, k: int):
    """
    kNN预测算法
    :param input_data:
    :param data_set:
    :param labels:
    :param k:
    :return:
    """
    sub_dis = input_data - data_set
    distance = np.sqrt(np.sum(np.square(sub_dis), axis=1))
    label_arg_sort = distance.argsort()
    type_count = {}
    index = 0
    for i in range(k):
        type_key = labels[label_arg_sort[i]]
        type_count[type_key] = type_count.get(type_key, 0) + 1
        index += 1
    type_zip = zip(type_count.values(), type_count.keys())
    aim = sorted(type_zip, reverse=True)
    return aim[0][1]


def train_test():
    test_rate = 0.1  # 测试的比例
    k = 5  # k值
    data_set, labels = file2matrix("resources/datingTestSet2.txt")
    data_set, _, _ = auto_norm(data_set)  # 归一化
    num = int(test_rate * data_set.shape[0])
    train_data = data_set[num:]
    test_data = data_set[:num]
    train_data_labels = labels[num:]
    test_data_labels = labels[:num]
    error_num = 0
    for data, real_label in zip(test_data, test_data_labels):
        pre_label = classify_knn(data, train_data, train_data_labels, k)
        print("预测结果为%s,真实结果为%s" % (pre_label, real_label))
        if pre_label != real_label:
            error_num += 1
            print("预测错啦")
    print("预测错误个数为%d", error_num)
    print("准确率为%f" % (1 - (error_num / num)))


if __name__ == '__main__':
    # return_mats, labels_list = file2matrix("resources/datingTestSet2.txt")
    # return_mat, _, _ = auto_norm(return_mats)
    # classify_knn(return_mat[0], return_mat, labels_list, 5)
    train_test()
