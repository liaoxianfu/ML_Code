import numpy as np
import os


def img2vector(file_name):
    vec = []
    with open(file_name) as f:
        readlines = f.readlines()
        for line in readlines:
            line = line.replace("\n", "")
            for c in list(line):
                vec.append(int(c))
    vec_arr = np.array(vec).reshape((1, 1024))
    return vec_arr


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


def hand_writing():
    label_list = []
    train_path = "resources/trainingDigits"
    train_file_list = os.listdir(train_path)
    train_file_len = len(train_file_list)
    train_set_mat = np.zeros((train_file_len, 1024))
    for i in range(train_file_len):
        file_name_str = train_file_list[i]
        label = file_name_str.split("_")[0]
        label_list.append(label)
        train_set_mat[i] = img2vector("resources/trainingDigits/%s" % file_name_str)
    test_path = 'resources/testDigits'
    test_file_list = os.listdir(test_path)
    test_file_len = len(test_file_list)
    error_num = 0
    for i in range(test_file_len):
        file_name_str = test_file_list[i]
        label = file_name_str.split("_")[0]
        test_data = img2vector('resources/testDigits/%s' % file_name_str)
        pre_label = classify_knn(test_data, train_set_mat, label_list, 9)
        print("预测结果为%s,真实结果为%s" % (pre_label, label))
        if label != pre_label:
            error_num += 1
            print("error")
    print("预测错误数量为%d" % error_num)
    print("准确率为%f" % (1 - (error_num / test_file_len)))


if __name__ == '__main__':
    hand_writing()
