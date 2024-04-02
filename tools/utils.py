import os
import sys
import time
import torch
import math

import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict
import torch.nn.functional as F
from itertools import combinations

_logger = logging.getLogger(__name__)


def read_data(dataset, split):
    pred_path = './output/' + dataset + '_' + split + '_pred_list.pth'
    labels_path = './output/' + dataset + '_' + split + '_labels.pth'
    pred_list = torch.load(pred_path)
    labels = torch.load(labels_path)
    return pred_list, labels


def get_ensemble_combination(n):
    res = []
    cur_num = 2
    while(cur_num <= n):
        for i in combinations(range(0, n), cur_num):
            res.append(list(i))
        cur_num = cur_num + 1
    return res


def load_state_dict(checkpoint_path, model, use_ema=False, num_classes=1000):
    if checkpoint_path and os.path.isfile(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict_key = 'state_dict'
        if isinstance(checkpoint, dict):
            if use_ema and 'state_dict_ema' in checkpoint:
                state_dict_key = 'state_dict_ema'
        if state_dict_key and state_dict_key in checkpoint:
            new_state_dict = OrderedDict()
            for k, v in checkpoint[state_dict_key].items():
                # strip `module.` prefix
                name = k[7:] if k.startswith('module') else k
                new_state_dict[name] = v
            state_dict = new_state_dict
        else:
            state_dict = checkpoint
        _logger.info("Loaded {} from checkpoint '{}'".format(state_dict_key, checkpoint_path))
        if num_classes != state_dict['head.bias'].shape[0]:
            # completely discard fully connected for all other differences between pretrained and created model
            del state_dict['head.weight']
            del state_dict['head.bias']

        return state_dict
    else:
        _logger.error("No checkpoint found at '{}'".format(checkpoint_path))
        raise FileNotFoundError()


def load_pretrained_weights(model, checkpoint_path, use_ema=False, strict=True, num_classes=1000):
    state_dict = load_state_dict(checkpoint_path, model, use_ema, num_classes)
    model.load_state_dict(state_dict, strict=strict)