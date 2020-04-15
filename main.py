# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('..'))
os.chdir(sys.path[0])

import time
import argparse
import pandas as pd
import copy
import torch
import tqdm
import traceback

from torch import optim
from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper

from path import TEXTSQL_DIR
from src.confs import arguments
from src.libs import utils
from src.cores.model import IRNet
from src.libs import semQL
from src.libs.sample import Sample

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Main(FlyAI):
    """
    项目中必须继承FlyAI类，否则线上运行会报错。
    """

    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        data_helper.download_from_ids("TextSQL")
        # 必须使用该方法下载模型，然后加载
        from flyai.utils import remote_helper
        remote_helper.get_remote_date('https://www.flyai.com/m/glove.42B.300d.zip')
        # 二选一或者根据app.json的配置下载文件
        # data_helper.download_from_json()
        print('=*=数据下载完成=*=')

    def deal_with_data(self):
        """
        处理数据，没有可不写。
        :return:
        """
        # 加载数据
        data = pd.read_csv(os.path.join(TEXTSQL_DIR, 'train.csv'))

        sql_data = []
        self.tables = []

        for index in range(data.shape[0]):
            tmp = eval(data.iloc[index, 0])
            tmp['sql'] = eval(data.iloc[index, 2])
            sql_data.append(tmp)
            self.tables.append(eval(data.iloc[index, 1]))

        # 划分训练集、测试集
        self.train_data, self.valid_data = train_test_split(sql_data, test_size=0.2, random_state=6, shuffle=True)

        print('=*=数据处理完成=*=')

    def train(self):
        """
        训练
        :return:
        """
        grammar = semQL.Grammar()
        model = IRNet(args, grammar)
        # 训练集
        self.train_data = Sample.generate_sample_std(input_data=self.train_data, table_data=self.tables,
                                                     input_contrast_question=args.contrast_question_json_path)
        # 测试集
        self.valid_data = Sample.generate_sample_std(input_data=self.valid_data, table_data=self.tables,
                                                     input_contrast_question=args.contrast_question_json_path)

        if args.cuda:
            model.cuda(DEVICE)

        # now get the optimizer
        optimizer_cls = eval('torch.optim.%s' % args.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
        if args.lr_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=[21, 41],
                                                       gamma=args.lr_scheduler_gammar)
        else:
            scheduler = None

        print('Loss epoch threshold: %d' % args.loss_epoch_threshold)
        print('Sketch loss coefficient: %f' % args.sketch_loss_coefficient)

        if args.load_model:
            pretrained_model = torch.load(args.load_model, map_location=lambda storage, loc: storage)
            pretrained_modeled = copy.deepcopy(pretrained_model)
            for k in pretrained_model.keys():
                if k not in model.state_dict().keys():
                    del pretrained_modeled[k]

            model.load_state_dict(pretrained_modeled)

        model.word_emb = utils.load_word_emb(args.glove_embed_path)

        # begin train
        model_save_path = utils.init_log_checkpoint_path(args)
        utils.save_args(args, os.path.join(model_save_path, 'config.json'))
        best_dev_acc = .0

        try:
            with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
                for epoch in tqdm.tqdm(range(args.epoch)):
                    if args.lr_scheduler:
                        scheduler.step(epoch=epoch)
                    epoch_begin = time.time()
                    loss = utils.epoch_train(model, optimizer, args.batch_size, self.train_data, self.tables,
                                             args,
                                             loss_epoch_threshold=args.loss_epoch_threshold,
                                             sketch_loss_coefficient=args.sketch_loss_coefficient)
                    epoch_end = time.time()
                    json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, self.valid_data,
                                                                  self.tables,
                                                                  beam_size=args.beam_size)
                    acc = utils.eval_acc(json_datas, self.valid_data)

                    if acc > best_dev_acc:
                        utils.save_checkpoint(model, os.path.join(model_save_path, 'best_model.model'))
                        best_dev_acc = acc
                    utils.save_checkpoint(model, os.path.join(model_save_path, '{%s}_{%s}.model') % (epoch, acc))

                    log_str = 'Epoch: %d, Loss: %f, Sketch Acc: %f, Acc: %f, time: %f\n' % (
                        epoch + 1, loss, sketch_acc, acc, epoch_end - epoch_begin)
                    tqdm.tqdm.write(log_str)
                    epoch_fd.write(log_str)
                    epoch_fd.flush()
        except Exception as e:
            # Save model
            utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
            print(e)
            tb = traceback.format_exc()
            print(tb)
        else:
            utils.save_checkpoint(model, os.path.join(model_save_path, 'end_model.model'))
            json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, self.valid_data, self.tables,
                                                          beam_size=args.beam_size)
            acc = utils.eval_acc(json_datas, self.valid_data)

            print("Sketch Acc: %f, Acc: %f, Beam Acc: %f" % (sketch_acc, acc, acc,))

    def evaluate(self):
        """
        :param args:
        :return:
        """
        # 加载数据
        data = pd.read_csv(args.test_csv_path, encoding='utf-8')
        data = Sample.generate_sample_std(input_data=data,
                                          table_data=self.tables,
                                          input_contrast_question=args.contrast_question_json_path)
        print('=*=数据处理完成=*=')
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
        json_datas, sketch_acc, acc = utils.epoch_acc(model=model, batch_size=args.batch_size, sql_data=data,
                                                      table_data=self.tables, beam_size=args.beam_size)

        question = []
        query = []
        for item in json_datas:
            query.append(item['query'].lower())
            question.append(item['question'])

        result = pd.DataFrame(data=question, columns=['question'])
        result['query'] = query

        result.to_csv(path_or_buf=args.result_csv_path, encoding='utf-8', index=False)


if __name__ == '__main__':
    # 项目的超参，不使用可以删除
    arg_parse = arguments.init_arg_parser()
    args = arguments.init_config(arg_parse)

    # 项目的超参，不使用可以删除
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=3, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    args.epoch = parser.parse_args().EPOCHS
    args.batch_size = parser.parse_args().BATCH

    args.project_dir = os.path.abspath('.')
    args.data_yan_dir = os.path.join(args.project_dir, 'data_yan')
    args.contrast_question_json_path = os.path.join(args.data_yan_dir, 'contrast_question.json')

    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)
