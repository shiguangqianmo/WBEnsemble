import torch
from timm.utils import accuracy
import numpy as np
from tqdm import tqdm
from tools.model_list import model_list
from VecEnsemble import _cal_w
from MatEnsemble import _cal_W
from tools.cal_metrics import cal_acc
from collections import Counter
import random
from sklearn.metrics import accuracy_score
from tools.utils import get_ensemble_combination, read_data


random.seed(10)


def hard_voting_acc(pred_list, labels):
    pred_list = pred_list.permute(1, 0, 2)
    _, preds = pred_list.max(dim=2)

    num = labels.shape[0]
    final_pred = torch.zeros(num)

    for i in range(num):
        counter = Counter(preds[i].tolist())
        votes = counter.most_common()
        final_pred[i] = votes[random.randint(0, len(votes) - 1)][0]
    acc = accuracy_score(labels, final_pred) * 100
    # print('acc:', acc)
    return acc


if __name__ == '__main__':
    val_pred, val_labels = read_data('IP102', 'val')
    test_pred, test_labels = read_data('IP102', 'test')
    ensemble_combination = get_ensemble_combination(val_pred.shape[0])

    for cur_comb in ensemble_combination:
        model_info = ''
        for cur_model in cur_comb:
            model_info = model_info + '+' + model_list[cur_model]['model_name']
        print(model_info[1:])

        # VecEnsemble
        w = _cal_w(val_pred[cur_comb], val_labels)
        print('VecEnsemble: ')
        print('val: ', cal_acc(w, val_pred[cur_comb], val_labels))
        print('test: ', cal_acc(w, test_pred[cur_comb], test_labels))

        # MatEnsemble
        print('MatEnsemble: ')
        W = _cal_W(val_pred[cur_comb], val_labels)
        print('val: ', cal_acc(W, val_pred[cur_comb], val_labels))
        print('val: ', cal_acc(W, test_pred[cur_comb], test_labels))

        # Soft Voting
        print('Soft Voting')
        w_S = torch.ones(len(cur_comb), 1) * (1 / len(cur_comb))
        print('val: ', cal_acc(w_S, val_pred[cur_comb], val_labels))
        print('test: ', cal_acc(w_S, test_pred[cur_comb], test_labels))

        # Hard Voting
        print('Hard Voting')
        print('val: ', hard_voting_acc(val_pred[cur_comb], val_labels))
        print('test: ', hard_voting_acc(test_pred[cur_comb], test_labels))




