import torch
from timm.utils import accuracy
import numpy as np
from tqdm import tqdm
from model_list import model_list
from VecEnsemble import _cal_w
from MatEnsemble import _cal_W
from cal_metrics import cal_acc
from collections import Counter
import random
from sklearn.metrics import accuracy_score
from utils import get_ensemble_combination, read_data


def hard_voting_2model_acc(pred_list, labels):
    pred_list = pred_list.permute(1, 0, 2)
    _, preds = pred_list.max(dim=2)

    num = labels.shape[0]
    final_pred = torch.zeros(num)

    for i in range(num):
        counter = Counter(preds[i].tolist())
        if counter.__len__() == 2:
            final_pred[i] = random.choice(list(counter.keys()))
        else:
            final_pred[i] = counter.most_common()[0][0]
    acc = accuracy_score(labels, final_pred) * 100
    # print('acc:', acc)
    return acc


def hard_voting_3model_acc(pred_list, labels):
    pred_list = pred_list.permute(1, 0, 2)
    _, preds = pred_list.max(dim=2)

    num = labels.shape[0]
    final_pred = torch.zeros(num)

    for i in range(num):
        counter = Counter(preds[i].tolist())
        if counter.__len__() == 3:
            final_pred[i] = random.choice(list(counter.keys()))
        else:
            final_pred[i] = counter.most_common()[0][0]
    acc = accuracy_score(labels, final_pred) * 100
    # print('acc:', acc)
    return acc


def hard_voting_4model_acc(pred_list, labels):
    pred_list = pred_list.permute(1, 0, 2)
    _, preds = pred_list.max(dim=2)

    num = labels.shape[0]
    final_pred = torch.zeros(num)

    for i in range(num):
        counter = Counter(preds[i].tolist())
        if counter.__len__() == 4:
            final_pred[i] = random.choice(list(counter.keys()))
        elif counter.__len__() == 3:
            final_pred[i] = counter.most_common()[0][0]
        elif counter.__len__() == 2:
            if list(counter.values())[0] == 2:
                final_pred[i] = random.choice(list(counter.keys()))
            else:
                final_pred[i] = counter.most_common()[0][0]
        else:
            final_pred[i] = counter.most_common()[0][0]
    acc = accuracy_score(labels, final_pred) * 100
    # print('acc:', acc)
    return acc


if __name__ == '__main__':
    val_pred = torch.load('../output/IP102_val_pred_list.pth')
    val_labels = torch.load('../output/IP102_val_labels.pth')
    test_pred = torch.load('../output/IP102_test_pred_list.pth')
    test_labels = torch.load('../output/IP102_test_labels.pth')
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
        if len(cur_comb) == 2:
            print('val: ', hard_voting_2model_acc(val_pred[cur_comb], val_labels))
            print('test: ', hard_voting_2model_acc(test_pred[cur_comb], test_labels))
        elif len(cur_comb) == 3:
            print('val: ', hard_voting_2model_acc(val_pred[cur_comb], val_labels))
            print('test: ', hard_voting_3model_acc(test_pred[cur_comb], test_labels))
        elif len(cur_comb) == 4:
            print('val: ', hard_voting_2model_acc(val_pred[cur_comb], val_labels))
            print('test: ', hard_voting_3model_acc(test_pred[cur_comb], test_labels))



