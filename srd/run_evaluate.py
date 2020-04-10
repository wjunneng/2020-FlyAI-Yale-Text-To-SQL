# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('..'))
os.chdir(sys.path[0])

import copy
import torch
import json
import argparse
import pandas as pd

from srd.libs import utils
from srd.cores.model import IRNet
from srd.libs import semQL
from srd.confs.arguments import init_arg_parser, init_config
from srd.run import Main

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluate(args):
    """
    :param args:
    :return:
    """

    tables = json.load(fp=open(os.path.join(args.data_path, 'tables.json')))
    tables = dict(zip([i['db_id'] for i in tables], tables))

    # 加载数据
    data = pd.read_csv(os.path.join(args.data_path, 'test.csv'))
    data = Main.generate_sample_1(input_data=data, input_tables=tables)

    grammar = semQL.Grammar()

    model = IRNet(args, grammar)

    if args.cuda:
        model.cuda(device=DEVICE)

    pretrained_model = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    pretrained_modeled = copy.deepcopy(pretrained_model)
    for k in pretrained_model.keys():
        if k not in model.state_dict().keys():
            del pretrained_modeled[k]

    model.load_state_dict(pretrained_modeled)

    model.word_emb = utils.load_word_emb(args.glove_embed_path)

    json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, data, tables, beam_size=args.beam_size)

    question = []
    query = []
    for item in json_datas:
        query.append(item['query'].lower())
        question.append(item['question'])

    result = pd.DataFrame(data=question, columns=['question'])
    result['query'] = query

    result.to_csv(path_or_buf=os.path.join(args.data_path, 'result.csv'), encoding='utf-8', index=False)


if __name__ == '__main__':
    arg_parse = init_arg_parser()
    args = init_config(arg_parse)

    # 项目的超参，不使用可以删除
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=10, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=128, type=int, help="batch size")
    args.epoch = parser.parse_args().EPOCHS
    args.batch_size = parser.parse_args().BATCH
    args.cuda = True
    args.loss_epoch_threshold = 50
    args.sketch_loss_coefficient = 1.0
    args.beam_size = 5
    args.seed = 90
    args.embed_size = 300
    args.sentence_features = True
    args.column_pointer = True
    args.hidden_size = 300
    args.lr_scheduler = True
    args.lr_scheduler_gammar = 0.5
    args.att_vec_size = 300
    args.glove_embed_path = './data/glove.42B.300d.txt'
    args.data_path = './data'
    args.load_model = './saved_model/1586480029/best_model.model'
    args.input_path = 'predict_lf.json'
    args.output_path = 'irnet'

    evaluate(args)
