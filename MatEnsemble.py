import torch
from tqdm import tqdm
import numpy as np
import time
from tools.cal_metrics import weighted_model_metric
from tools.utils import read_data



def _cal_W(val_pred, val_labels):
    _size = val_pred.shape[0]
    val_pred = val_pred.permute(1, 2, 0)  # [4, 7508, 102] -> [7508, 102, 4]
    val_labels = val_labels.unsqueeze(1)  # [7508] -> [7508, 1]
    b = torch.zeros(val_labels.shape[0], val_pred.shape[1]).scatter_(1, val_labels, 1)  # [7508, 102]

    # calculate w1, w2, ..., w102
    class_num = b.shape[1]
    W = torch.zeros(_size, class_num)
    for i in range(class_num):
        # simplify
        tmp_X = val_pred[:, i, :]
        X = tmp_X.unsqueeze(dim=2) @ tmp_X.unsqueeze(dim=1)
        X = torch.sum(X, dim=0)
        tmp_Y = b[:, i].unsqueeze(1).unsqueeze(2)
        Y = tmp_Y @ tmp_X.unsqueeze(dim=1)
        Y = torch.sum(Y, dim=0)

        Y *= 2
        w = 0.5 * (torch.inverse(X) @ Y.t())  # [4, 1]
        W[:, i] = w.squeeze(1)
    return W


if __name__ == '__main__':
    val_pred, val_labels = read_data('IP102', 'val')
    test_pred, test_labels = read_data('IP102', 'test')

    W = _cal_W(val_pred, val_labels)
    # np.save('../output/weights/Deng_W.npy', W)
    print('val: ')
    weighted_model_metric(val_pred, val_labels, W)
    print('test: ')
    weighted_model_metric(test_pred, test_labels, W)

    # ==== Compution time ====
    # time_list = np.zeros(10)
    # for i in range(10):
    #     start_time = time.time()
    #     W = _cal_W(val_pred, val_labels)
    #     end_time = time.time()
    #     time_list[i] = end_time - start_time
    # print("Runtime: %.4f seconds" % (np.mean(time_list)))
    # ====
