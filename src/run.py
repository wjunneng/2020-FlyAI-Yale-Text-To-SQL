# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

sys.path.append(os.path.abspath('..'))
os.chdir(sys.path[0])
import argparse
import copy
import time
import tqdm
import torch
import json
import traceback
import pandas as pd

from torch import optim
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from src.confs.arguments import init_arg_parser, init_config
from src.libs import utils
from src.cores.model import IRNet
from src.libs import semQL
from src.libs.sample import Sample

wordnet_lemmatizer = WordNetLemmatizer()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Main(object):
    def __init__(self, args):
        self.args = args
        # 表数据
        self.tables = json.load(fp=open(self.args.tables_json_path), encoding='utf-8')
        self.tables = dict(zip([i['db_id'] for i in self.tables], self.tables))

    def train(self):
        """
        训练
        :return:
        """
        # 加载数据
        data = pd.read_csv(self.args.train_csv_path, encoding='utf-8')
        # 划分训练集、测试集
        train_data, valid_data = train_test_split(data, test_size=0.2, random_state=6,
                                                  shuffle=True)
        # 训练集
        self.train_data = Sample.generate_sample_std(input_data=train_data,
                                                     input_contrast_question=self.args.contrast_question_json_path)
        # 测试集
        self.vaild_data = Sample.generate_sample_std(input_data=valid_data,
                                                     input_contrast_question=self.args.contrast_question_json_path)
        print('=*=数据处理完成=*=')

        grammar = semQL.Grammar()
        model = IRNet(self.args, grammar)

        if self.args.cuda:
            model.cuda(DEVICE)

        # now get the optimizer
        optimizer_cls = eval('torch.optim.%s' % self.args.optimizer)
        optimizer = optimizer_cls(model.parameters(), lr=self.args.lr)
        if self.args.lr_scheduler:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[21, 41],
                                                       gamma=self.args.lr_scheduler_gammar)
        else:
            scheduler = None

        print('Loss epoch threshold: %d' % self.args.loss_epoch_threshold)
        print('Sketch loss coefficient: %f' % self.args.sketch_loss_coefficient)

        if self.args.load_model:
            pretrained_model = torch.load(self.args.load_model, map_location=lambda storage, loc: storage)
            pretrained_modeled = copy.deepcopy(pretrained_model)
            for k in pretrained_model.keys():
                if k not in model.state_dict().keys():
                    del pretrained_modeled[k]

            model.load_state_dict(pretrained_modeled)

        model.word_emb = utils.load_word_emb(self.args.glove_embed_path)

        # begin train
        model_save_path = utils.init_log_checkpoint_path(self.args)
        utils.save_args(self.args, os.path.join(model_save_path, 'config.json'))
        best_dev_acc = .0

        try:
            with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
                for epoch in tqdm.tqdm(range(self.args.epoch)):
                    if self.args.lr_scheduler:
                        scheduler.step(epoch=epoch)
                    epoch_begin = time.time()
                    loss = utils.epoch_train(model, optimizer, self.args.batch_size, self.train_data, self.tables,
                                             self.args,
                                             loss_epoch_threshold=self.args.loss_epoch_threshold,
                                             sketch_loss_coefficient=self.args.sketch_loss_coefficient)
                    epoch_end = time.time()
                    json_datas, sketch_acc, acc = utils.epoch_acc(model, self.args.batch_size, self.vaild_data,
                                                                  self.tables,
                                                                  beam_size=self.args.beam_size)
                    acc = utils.eval_acc(json_datas, self.vaild_data)

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
            json_datas, sketch_acc, acc = utils.epoch_acc(model, self.args.batch_size, self.vaild_data, self.tables,
                                                          beam_size=self.args.beam_size)
            acc = utils.eval_acc(json_datas, self.vaild_data)

            print("Sketch Acc: %f, Acc: %f, Beam Acc: %f" % (sketch_acc, acc, acc,))

    def evaluate(self):
        """
        :param args:
        :return:
        """
        # 加载数据
        data = pd.read_csv(self.args.test_csv_path, encoding='utf-8')
        data = Sample.generate_sample_std(input_data=data,
                                          input_contrast_question=self.args.contrast_question_json_path)
        print('=*=数据处理完成=*=')
        grammar = semQL.Grammar()
        model = IRNet(self.args, grammar)

        if self.args.cuda:
            model.cuda(device=DEVICE)

        pretrained_model = torch.load(self.args.load_model, map_location=lambda storage, loc: storage)
        pretrained_modeled = copy.deepcopy(pretrained_model)
        for k in pretrained_model.keys():
            if k not in model.state_dict().keys():
                del pretrained_modeled[k]

        model.load_state_dict(pretrained_modeled)
        model.word_emb = utils.load_word_emb(self.args.glove_embed_path)
        json_datas, sketch_acc, acc = utils.epoch_acc(model=model, batch_size=self.args.batch_size, sql_data=data,
                                                      table_data=self.tables, beam_size=self.args.beam_size)

        question = []
        query = []
        for item in json_datas:
            query.append(item['query'].lower())
            question.append(item['question'])

        result = pd.DataFrame(data=question, columns=['question'])
        result['query'] = query

        result.to_csv(path_or_buf=self.args.result_csv_path, encoding='utf-8', index=False)


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
    args.beam_size = 1
    args.seed = 90
    args.embed_size = 300
    args.sentence_features = True
    args.column_pointer = True
    args.hidden_size = 300
    args.lr_scheduler = True
    args.lr_scheduler_gammar = 0.5
    args.att_vec_size = 300

    args.project_dir = os.path.abspath('..')
    args.data_yan_dir = os.path.join(args.project_dir, 'data_yan')
    args.glove_embed_path = os.path.join(args.data_yan_dir, 'glove.42B.300d.txt')
    args.tables_json_path = os.path.join(args.data_yan_dir, 'tables.json')
    args.train_csv_path = os.path.join(args.data_yan_dir, 'train.csv')
    args.test_csv_path = os.path.join(args.data_yan_dir, 'test.csv')
    args.result_csv_path = os.path.join(args.data_yan_dir, 'result.csv')
    args.save = os.path.join(args.data_yan_dir, 'saved_model')
    args.contrast_question_json_path = os.path.join(args.data_yan_dir, 'contrast_question.json')
    # args.pretrained_model = os.path.join(args.data_yan_dir, 'IRNet_pretrained.model')
    args.load_model = os.path.join(args.save, 'best_model.model')

    main = Main(args=args)
    # main.train()

    main.evaluate()
