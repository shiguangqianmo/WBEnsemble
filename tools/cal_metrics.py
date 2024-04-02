from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from timm.utils import accuracy


def cal_acc(weight, preds, labels):
    preds = preds.permute(1, 0, 2)
    weighted_pred = preds * weight
    pred = weighted_pred.sum(axis=1)
    acc1, acc5 = accuracy(pred, labels, topk=(1, 5))
    # print('acc1:', acc1, ', acc5:', acc5)
    return acc1, acc5


def weighted_model_metric(pred_list, labels, weight):
    pred_list = pred_list.permute(1, 0, 2)  # [4, 7508, 102] -> [7508, 4, 102]
    weighted_pred = pred_list * weight  # dot
    pred = weighted_pred.sum(axis=1)
    _, pred = pred.topk(1, 1, True, True)
    pred = pred.squeeze()

    acc = accuracy_score(labels, pred) * 100
    prec = precision_score(labels, pred, average='macro') * 100
    rec = recall_score(labels, pred, average='macro') * 100
    f1 = f1_score(labels, pred, average='macro') * 100

    print('acc: %.4f, prec: %.4f, rec: %.4f, f1: %.4f' % (acc, prec, rec, f1))
    return acc, prec, rec, f1