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
import re
import torch
import json
import traceback
import pandas as pd

from torch import optim
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

from srd.confs.arguments import init_arg_parser, init_config
from srd.libs import utils
from srd.cores.model import IRNet
from srd.libs import semQL

wordnet_lemmatizer = WordNetLemmatizer()
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Main(object):
    @staticmethod
    def delete_space(item):
        if ' ( ' in item:
            item = item.replace(' ( ', '(')
        elif '( ' in item:
            item = item.replace('( ', '(')
        elif ' (' in item:
            item = item.replace(' (', '(')

        if ' ) ' in item:
            item = item.replace(' ) ', ')')
        elif ') ' in item:
            item = item.replace(') ', ')')
        elif ' )' in item:
            item = item.replace(' )', ')')

        return item

    @staticmethod
    def deal_order_action(rule_able, current_query_str):
        # Order
        if current_query_str.endswith(' desc'):
            rule_able += 'Order(0) '
            char = 'desc'
        else:
            rule_able += 'Order(1) '
            char = ' asc'

        item = current_query_str[
               current_query_str.index(' order by ') + len(' order by '): current_query_str.index(
                   char)].strip()
        item = Main.delete_space(item=item)
        return rule_able, item

    @staticmethod
    def deal_sup_action(rule_able, current_query_str):
        # Sup
        if ' desc limit ' in current_query_str:
            rule_able += 'Sup(0) '
            char = ' desc limit '
        else:
            rule_able += 'Sup(1) '
            char = ' asc limit '

        item = current_query_str[
               current_query_str.index(' order by ') + len(' order by '): current_query_str.index(
                   char)].strip()
        item = Main.delete_space(item=item)
        return rule_able, item

    @staticmethod
    def deal_filter_action(rule_able, current_query_str):
        # Filter
        if ' = ' in current_query_str:
            rule_able += 'Filter(2) '

            char = '='
        elif ' != ' in current_query_str:
            rule_able += 'Filter(3) '

            char = '!='
        elif ' < ' in current_query_str:
            rule_able += 'Filter(4) '

            char = '<'
        elif ' > ' in current_query_str:
            rule_able += 'Filter(5) '

            char = '>'
        elif ' <= ' in current_query_str:
            rule_able += 'Filter(6) '

            char = '<='
        elif ' >= ' in current_query_str:
            rule_able += 'Filter(7) '

            char = '>='
        elif ' between ' in current_query_str:
            rule_able += 'Filter(8) '

            char = 'between'
        elif ' like ' in current_query_str:
            rule_able += 'Filter(9) '

            char = 'like'
        elif ' not_like ' in current_query_str:
            rule_able += 'Filter(10) '

            char = 'not_like'

        else:
            print(current_query_str)

        item = current_query_str[: current_query_str.index(char)].strip()
        item = Main.delete_space(item=item)
        return rule_able, item

    @staticmethod
    def generate_act(item, col_set, table_names, current_query_list, query_list):
        # A: Aggregator
        if 'max(' in item:
            aggregator = 'A(1) '
            item = item[item.index('(') + 1:item.index(')')]
        elif 'min(' in item:
            aggregator = 'A(2) '
            item = item[item.index('(') + 1:item.index(')')]
        elif 'count(' in item:
            aggregator = 'A(3) '
            item = item[item.index('(') + 1:item.index(')')]
        elif 'sum(' in item:
            aggregator = 'A(4) '
            item = item[item.index('(') + 1:item.index(')')]
        elif 'avg(' in item:
            aggregator = 'A(5) '
            item = item[item.index('(') + 1:item.index(')')]
        elif 'distinct(' in item:
            aggregator = 'A(6) '
            item = item[item.index('(') + 1:item.index(')')]
        else:
            aggregator = 'A(0) '

        # C: column
        item = item.replace('.', ' ')
        while True:
            if item in col_set:
                break

            item_0 = item.split(' ')[0]
            item_1 = ' '.join(item.split(' ')[1:])
            if item_0 in col_set:
                item = item_0
                break

            item = item_1
        column = 'C(' + str(col_set.index(item)) + ') '

        # T: table
        table_index = current_query_list.index('from') + 1
        table_value = current_query_list[table_index]
        if table_value.strip('(').strip(')') in table_names:
            table_value = table_value.strip('(').strip(')')
        else:
            while table_value not in table_names:
                table_index = query_list.index(table_value)
                current_text = query_list[table_index:]
                table_index = current_text.index('from') + 1
                table_value = current_text[table_index].strip('(').strip(')')
                if table_value in table_names:
                    break
        table = 'T(' + str(table_names.index(table_value)) + ') '

        return aggregator + column + table

    @staticmethod
    def generate_rule_label(query, query_list, table_names, col_set):
        rule_able = ''
        # Root1
        if ' intersect ' in query:
            rule_able += 'Root1(0) '
        elif ' union ' in query:
            rule_able += 'Root1(1) '
        elif ' except ' in query:
            rule_able += 'Root1(2) '
        else:
            rule_able += 'Root1(3) '

        select_start_index = query_list.index('select')
        select_end_indexs = [i for i, x in enumerate(query_list) if x == 'select']
        select_end_indexs.append(len(query_list))
        for select_end_index in select_end_indexs[1:]:
            current_query_list = query_list[select_start_index:select_end_index]

            current_query_str = ' '.join(current_query_list)
            select_start_index = select_end_index
            # Root
            """
                0: 'Root Sel Sup Filter',    关键字[(desc limit 或 asc limit) 和 where]
                2: 'Root Sel Sup',           关键字[(desc limit 或 asc limit)]
                1: 'Root Sel Filter Order',  关键字[where 和 (asc 或 desc) 注asc和desc通常位于尾部]
                3: 'Root Sel Filter',        关键字[where]
                4: 'Root Sel Order',         关键字[(asc 或 desc)]
                5: 'Root Sel'                关键字[select]
            """
            root = ''
            if (
                    ' desc limit ' in current_query_str or ' asc limit ' in current_query_str) and ' where ' in current_query_str:
                root = 'Root(0) '
            elif ' desc limit ' in current_query_str or ' asc limit ' in current_query_str:
                root = 'Root(2) '
            elif ' where' in current_query_str and (
                    current_query_str.endswith(' asc') or current_query_str.endswith(' desc')):
                root = 'Root(1) '
            elif ' where ' in current_query_str:
                root = 'Root(3) '
            elif current_query_str.endswith(' asc') or current_query_str.endswith(' desc'):
                root = 'Root(4) '
            else:
                root = 'Root(5) '
            rule_able += root

            # Sel
            rule_able += 'Sel(0) '

            # 从select到from关键字之间的字符 解决['avg(hs) ', ' max(hs) ', ' min(hs)']和['max ( distinct length )']问题
            select_text_from_list = ' '.join(current_query_list[1:current_query_list.index('from')]).split(',')
            select_text_from_list_norm = []
            for item in select_text_from_list:
                item = item.strip()
                item = Main.delete_space(item=item)

                select_text_from_list_norm.append(item)

            # N
            rule_able += 'N(' + str(len(select_text_from_list_norm) - 1) + ') '

            for item in select_text_from_list_norm:
                item = item.strip()

                rule_able += Main.generate_act(item=item, col_set=col_set, table_names=table_names,
                                               current_query_list=current_query_list, query_list=query_list)

            # 'Root Sel Sup Filter' eg: Sup(0) A(3) C(0) T(2) Filter(4) A(0) C(3) T(0)
            if root == 'Root(0) ':
                pass

                # # Sup
                # if ' desc limit ' in current_query_str:
                #     rule_able += 'Sup(0) '
                # else:
                #     rule_able += 'Sup(1) '
                #
                # rule_able += Main.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)
                #
                # # Filter
                # rule_able += Main.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)

            # 'Root Sel Sup' eg: Sup(0) A(3) C(0) T(2)
            elif root == 'Root(2) ':
                # Sup
                rule_able, item = Main.deal_sup_action(rule_able=rule_able, current_query_str=current_query_str)
                rule_able += Main.generate_act(item=item, col_set=col_set, table_names=table_names,
                                               current_query_list=current_query_list, query_list=query_list)

            # 'Root Sel Filter Order' eg: Filter(5) A(0) C(3) T(0) Order(1) A(0) C(1) T(0)
            elif root == 'Root(1) ':
                pass

                # rule_able += Main.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)
                #
                # # Order
                # if ' desc' in current_query_str:
                #     rule_able += 'Order(0) '
                # else:
                #     rule_able += 'Order(1) '
                #
                # rule_able += Main.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)

            # 'Root Sel Filter' eg: Filter(18) A(0) C(14) T(2)
            elif root == 'Root(3) ':
                pass
                # # Filter
                # item_str = current_query_str[current_query_str.index('where') + len('where'):].strip()
                #
                # while len(item_str) != 0:
                #     if item_str.count(' and ') > 1:
                #         print('\n' + item_str)
                #         break
                #
                #     if ' and ' in item_str:
                #         print(item_str)
                #         rule_able += 'Filter(0) '
                #         current_item_str = item_str[:item_str.index(' and ')]
                #         current_item_str = Main.delete_space(item=current_item_str)
                #
                #         rule_able, item = Main.deal_filter_action(rule_able=rule_able,
                #                                                   current_query_str=current_item_str)
                #         rule_able += Main.generate_act(item=item, col_set=col_set,
                #                                        table_names=table_names,
                #                                        current_query_list=current_query_list, query_list=query_list)
                #         item_str = item_str[item_str.index(' and ') + len(' and '):]
                #
                #     else:
                #         break
            # 'Root Sel Order' eg: Order(1) A(0) C(2) T(0)
            elif root == 'Root(4) ':
                rule_able, item = Main.deal_order_action(rule_able=rule_able, current_query_str=current_query_str)

                rule_able += Main.generate_act(item=item, col_set=col_set, table_names=table_names,
                                               current_query_list=current_query_list, query_list=query_list)


            # 'Root Sel' eg: 无需处理
            else:
                pass

            # print('\n')
            # print(query)
            # print(rule_able)
            # print('\n')

        return rule_able

    @staticmethod
    def generate_sample(input_data, input_tables):
        result = []
        count = 0
        for index in range(input_data.shape[0]):
            # print('\nindex: {}'.format(index))
            sample = dict()
            # eg.'program_share'
            db_id = input_data.iloc[index, 0]
            # eg.'what is the number of different channel owners?'
            question = input_data.iloc[index, 1]
            # eg.'select count(distinct owner) from channel'
            query = input_data.iloc[index, 2]
            # table
            table = input_tables[db_id]

            # ################## db_id, query, question, question_toks, question_arg, tabel_names
            sample['db_id'] = db_id
            sample['question'] = question
            sample['question_toks'] = re.findall(r"[\w']+|[.,!?;']", question.lower())
            sample['question_arg'] = [[i.lower()] for i in sample['question_toks']]
            sample['table_names'] = table['table_names_original']

            # ################## query 待优化 存在 "count ( t4.paperid )" 的情况
            query_list = query.strip(';').split(' ')
            while '' in query_list:
                query_list.remove('')
            sample['query'] = query

            # ################## col_set
            col_set = []
            for i in table['column_names_original']:
                if i[1] not in col_set:
                    col_set.append(i[1])
            sample['col_set'] = col_set

            # ################## question_arg_type 待优化
            tmp = []
            for char in sample['question_arg']:
                char = char[0]
                if char == query_list[query_list.index('from') + 1]:
                    tmp.append(['table'])
                else:
                    tmp.append(['NONE'])
            sample['question_arg_type'] = tmp
            assert len(sample['question_arg']) == len(sample['question_arg_type'])

            # ################## rule_table
            with open('../data/query_rule.json', mode='r', encoding='utf-8') as file:
                query_rule_able = json.load(file)

            if sample['query'] in query_rule_able:
                # print('query->rule_table: {}'.format(query))
                sample['rule_label'] = query_rule_able[sample['query']].strip()
            else:
                count += 1
                sample['rule_label'] = Main.generate_rule_label(query=query, query_list=query_list, col_set=col_set,
                                                                table_names=sample['table_names'])
            result.append(sample)

        print(count)
        return result

    @staticmethod
    def generate_sample_1(input_data, input_tables):
        result = []

        with open(file='data/contrast.json', encoding='utf-8', mode='r') as file:
            contrast = json.load(fp=file)

        count = 0
        for index in range(input_data.shape[0]):
            # print('\nindex: {}'.format(index))
            sample = dict()
            # eg.'program_share'
            db_id = input_data.iloc[index, 0]
            # eg.'what is the number of different channel owners?'
            question = input_data.iloc[index, 1]
            # eg.'select count(distinct owner) from channel'
            query = input_data.iloc[index, 2]
            # # table
            # table = input_tables[db_id]

            if query in contrast:
                sample['db_id'], sample['question'], sample['question_toks'], sample['question_arg'], sample[
                    'question_arg_type'], sample['query'], sample['table_names'], sample['col_set'], sample[
                    'rule_label'] = \
                    contrast[query]

                sample['db_id'] = db_id
                result.append(sample)
            else:
                count += 1

        print(count)
        return result

    def deal_with_data(self):
        """
        处理数据，没有可不写。
        :return:
        """
        # 表数据
        self.tables = json.load(fp=open('data/table.json'))
        self.tables = dict(zip([i['db_id'] for i in self.tables], self.tables))
        # 加载数据
        self.data = pd.read_csv(os.path.join('data/train.csv'))
        # 划分训练集、测试集
        train_data, valid_data = train_test_split(self.data, test_size=0.2, random_state=6, shuffle=True)

        self.train_data = Main.generate_sample_1(input_data=train_data, input_tables=self.tables)

        self.vaild_data = Main.generate_sample_1(input_data=valid_data, input_tables=self.tables)

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

        model.word_emb = utils.load_word_emb(args.glove_embed_path)
        # begin train

        model_save_path = utils.init_log_checkpoint_path(args)
        utils.save_args(args, os.path.join(model_save_path, 'config.json'))
        best_dev_acc = .0

        try:
            with open(os.path.join(model_save_path, 'epoch.log'), 'w') as epoch_fd:
                for epoch in tqdm.tqdm(range(args.epoch)):
                    if args.lr_scheduler:
                        scheduler.step()
                    epoch_begin = time.time()
                    loss = utils.epoch_train(model, optimizer, args.batch_size, self.train_data, self.tables, args,
                                             loss_epoch_threshold=args.loss_epoch_threshold,
                                             sketch_loss_coefficient=args.sketch_loss_coefficient)
                    epoch_end = time.time()
                    json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, self.vaild_data, self.tables,
                                                                  beam_size=args.beam_size)
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
            json_datas, sketch_acc, acc = utils.epoch_acc(model, args.batch_size, self.vaild_data, self.tables,
                                                          beam_size=args.beam_size)
            acc = utils.eval_acc(json_datas, self.vaild_data)

            print("Sketch Acc: %f, Acc: %f, Beam Acc: %f" % (sketch_acc, acc, acc,))


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

    main = Main()
    main.deal_with_data()
    main.train()

    exit(0)
