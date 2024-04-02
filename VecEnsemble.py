import time

import torch
import numpy as np
import datetime
from tools.cal_metrics import weighted_model_metric
from tools.utils import read_data


def _cal_w(val_pred, val_labels):
    _size = val_pred.shape[0]
    val_pred = val_pred.permute(1, 2, 0)  # [4, 7508, 102] -> [7508, 102, 4]
    val_labels = val_labels.unsqueeze(1)  # [7508] -> [7508, 1]
    Y = torch.zeros(val_labels.shape[0], val_pred.shape[1]).scatter_(1, val_labels, 1)  # [7508, 102]

    # simplify
    A = torch.sum(val_pred.permute(0, 2, 1) @ val_pred, dim=0)
    b = torch.sum(Y.unsqueeze(dim=1) @ val_pred, dim=0)

    b *= 2
    w = 0.5 * torch.mm(torch.inverse(A), b.t())  # [4, 1]
    return w


if __name__ == '__main__':
    val_pred, val_labels = read_data('IP102', 'val')
    test_pred, test_labels = read_data('IP102', 'test')

    w = _cal_w(val_pred, val_labels)
    print('w: ', w)
    # np.save('../output/weights/IP102_vec.npy', w)
    print('val: ')
    weighted_model_metric(val_pred, val_labels, w)
    print('test: ')
    weighted_model_metric(test_pred, test_labels, w)

    # ==== Compution time ====
    # time_list = np.zeros(10)
    # for i in range(10):
    #     start_time = time.time()
    #     W = _cal_w(val_pred, val_labels)
    #     end_time = time.time()
    #     time_list[i] = end_time - start_time
    # print("Runtime: %.4f seconds" % (np.mean(time_list)))
    # ====
