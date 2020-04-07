# -*- coding: utf-8 -*-
import os
import sys

os.chdir(sys.path[0])
import argparse
import copy
import time
import tqdm
import traceback

from sklearn.model_selection import train_test_split
from customer import *
from torch import optim

import json
from srd.confs.arguments import init_arg_parser, init_config
from srd.libs import utils
from srd.cores.model import IRNet
from srd.libs import semQL

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Main(object):
    def deal_with_data(self):
        """
        处理数据，没有可不写。
        :return:
        """
        # 表数据
        self.tables = json.load(fp=open('../data/tables.json'))
        self.tables = dict(zip([i['db_id'] for i in self.tables], self.tables))
        # 加载数据
        self.data = pd.read_csv(os.path.join('../data/train.csv'))
        # 划分训练集、测试集
        train_data, valid_data = train_test_split(self.data, test_size=0.2, random_state=6, shuffle=True)

        self.train_data = []
        for index in range(train_data.shape[0]):
            sample = {}
            # eg.'program_share'
            db_id = train_data.iloc[0, 0]
            # eg.'what is the number of different channel owners?'
            question = train_data.iloc[0, 1]
            # eg.'select count(distinct owner) from channel'
            query = train_data.iloc[0, 2]

            sample['db_id'] = db_id
            sample['query'] = query
            sample['question'] = question
            self.train_data.append(sample)

        self.vaild_data = []
        for index in range(valid_data.shape[0]):
            sample = {}
            # eg.'program_share'
            db_id = valid_data.iloc[0, 0]
            # eg.'what is the number of different channel owners?'
            question = valid_data.iloc[0, 1]
            # eg.'select count(distinct owner) from channel'
            query = valid_data.iloc[0, 2]

            sample['db_id'] = db_id
            sample['query'] = query
            sample['question'] = question
            self.vaild_data.append(sample)

        print('=*=数据处理完成=*=')

    def train(self):
        """
        训练
        :return:
        """
        grammar = semQL.Grammar()
        model = IRNet(args, grammar)

        if args.cuda:
            model.cuda(DEVICE)

        # now get the optimizer
        optimizer_cls = eval('torch.optim.%s' % args.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)
        print('Enable Learning Rate Scheduler: ', args.lr_scheduler)
        if args.lr_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21, 41], gamma=args.lr_scheduler_gammar)
        else:
            scheduler = None

        print('Loss epoch threshold: %d' % args.loss_epoch_threshold)
        print('Sketch loss coefficient: %f' % args.sketch_loss_coefficient)

        if args.load_model:
            print('load pretrained model from %s' % args.load_model)
            pretrained_model = torch.load(args.load_model, map_location=lambda storage, loc: storage)
            pretrained_modeled = copy.deepcopy(pretrained_model)
            for k in pretrained_model.keys():
                if k not in model.state_dict().keys():
                    del pretrained_modeled[k]

            model.load_state_dict(pretrained_modeled)

        # model.word_emb = utils.load_word_emb(args.glove_embed_path)
        # begin train

        model_save_path = utils.init_log_checkpoint_path(args)
        utils.save_args(args, os.path.join(model_save_path, 'config.json'))
        best_dev_acc = .0

        try:
            with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
                for epoch in tqdm.tqdm(range(args.epoch)):
                    if args.lr_scheduler:
                        scheduler.step(epoch)
                    epoch_begin = time.time()
                    loss = utils.epoch_train(model, optimizer, args.batch_size, self.train_data, self.tables, args,
                                             loss_epoch_threshold=args.loss_epoch_threshold,
                                             sketch_loss_coefficient=args.sketch_loss_coefficient)
                    epoch_end = time.time()
                    json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, self.vaild_data, self.tables,
                                                                  beam_size=args.beam_size)
                    # acc = utils.eval_acc(json_datas, val_sql_data)

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
            json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, self.vaild_data, self.tables,
                                                          beam_size=args.beam_size)
            # acc = utils.eval_acc(json_datas, val_sql_data)

            print("Sketch Acc: %f, Acc: %f, Beam Acc: %f" % (sketch_acc, acc, acc,))


if __name__ == '__main__':
    arg_parse = init_arg_parser()
    args = init_config(arg_parse)

    # 项目的超参，不使用可以删除
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
    args.EPOCHS = parser.parse_args().EPOCHS
    args.BATCH = parser.parse_args().BATCH

    main = Main()
    main.deal_with_data()
    main.train()

    exit(0)
