# -*- coding:utf-8 -*-
from __future__ import absolute_import, division, print_function
import os
import sys

os.chdir(sys.path[0])

import re
import json


class Sample(object):
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
        item = Sample.delete_space(item=item)
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
        item = Sample.delete_space(item=item)
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
        item = Sample.delete_space(item=item)
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
                item = Sample.delete_space(item=item)

                select_text_from_list_norm.append(item)

            # N
            rule_able += 'N(' + str(len(select_text_from_list_norm) - 1) + ') '

            for item in select_text_from_list_norm:
                item = item.strip()

                rule_able += Sample.generate_act(item=item, col_set=col_set, table_names=table_names,
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
                # rule_able += Sample.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)
                #
                # # Filter
                # rule_able += Sample.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)

            # 'Root Sel Sup' eg: Sup(0) A(3) C(0) T(2)
            elif root == 'Root(2) ':
                # Sup
                rule_able, item = Sample.deal_sup_action(rule_able=rule_able, current_query_str=current_query_str)
                rule_able += Sample.generate_act(item=item, col_set=col_set, table_names=table_names,
                                                 current_query_list=current_query_list, query_list=query_list)

            # 'Root Sel Filter Order' eg: Filter(5) A(0) C(3) T(0) Order(1) A(0) C(1) T(0)
            elif root == 'Root(1) ':
                pass

                # rule_able += Sample.generate_act(item='', col_set=col_set, table_names=table_names,
                #                                current_query_list=current_query_list, query_list=query_list)
                #
                # # Order
                # if ' desc' in current_query_str:
                #     rule_able += 'Order(0) '
                # else:
                #     rule_able += 'Order(1) '
                #
                # rule_able += Sample.generate_act(item='', col_set=col_set, table_names=table_names,
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
                #         current_item_str = Sample.delete_space(item=current_item_str)
                #
                #         rule_able, item = Sample.deal_filter_action(rule_able=rule_able,
                #                                                   current_query_str=current_item_str)
                #         rule_able += Sample.generate_act(item=item, col_set=col_set,
                #                                        table_names=table_names,
                #                                        current_query_list=current_query_list, query_list=query_list)
                #         item_str = item_str[item_str.index(' and ') + len(' and '):]
                #
                #     else:
                #         break
            # 'Root Sel Order' eg: Order(1) A(0) C(2) T(0)
            elif root == 'Root(4) ':
                rule_able, item = Sample.deal_order_action(rule_able=rule_able, current_query_str=current_query_str)

                rule_able += Sample.generate_act(item=item, col_set=col_set, table_names=table_names,
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
        for index in range(input_data.shape[0]):
            # print('\nindex: {}'.format(index))
            sample = dict()
            # eg.'program_share'
            db_id = input_data.iloc[index, input_data.columns.tolist().index('table_id')]
            # eg.'what is the number of different channel owners?'
            question = input_data.iloc[index, input_data.columns.tolist().index('question')]
            # eg.'select count(distinct owner) from channel'
            query = input_data.iloc[index, input_data.columns.tolist().index('query')]
            # table
            table = dict(zip(input_tables[input_tables['id'] == db_id].columns,
                             input_tables[input_tables['id'] == db_id].values[0]))

            # ################## db_id, query, question, question_toks, question_arg, tabel_names
            sample['db_id'] = db_id
            sample['question'] = question
            sample['question_toks'] = re.findall(r"[\w']+|[.,!?;']", question.lower())
            sample['question_arg'] = [[i.lower()] for i in sample['question_toks']]
            sample['table_names'] = table['name']

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
            sample['rule_label'] = Sample.generate_rule_label(query=query, query_list=query_list, col_set=col_set,
                                                              table_names=sample['table_names'])
            result.append(sample)

        return result

    @staticmethod
    def generate_sample_std(input_data, input_contrast_question):
        result = []

        with open(file=input_contrast_question, encoding='utf-8', mode='r') as file:
            contrast = json.load(fp=file)

        count = 0
        for index in range(input_data.shape[0]):
            # print('\nindex: {}'.format(index))
            sample = dict()
            # eg.'program_share'
            db_id = input_data.iloc[index, 0]
            # eg.'what is the number of different channel owners?'
            question = input_data.iloc[index, 1]

            if question in contrast:
                sample['db_id'], sample['question'], sample['question_toks'], sample['question_arg'], sample[
                    'question_arg_type'], sample['query'], sample['table_names'], sample['col_set'], sample[
                    'rule_label'] = \
                    contrast[question]

                # 注意是否要转成小写的
                sample['question'] = sample['question'].lower()

                sample['db_id'] = db_id

                result.append(sample)
            else:
                count += 1

        # ########## step 1 ##########
        # # 训练集
        # 忽略了: 1608 条样本

        # # 验证集
        # 忽略了: 388 条样本

        # # 测试集
        # 忽略了: 497 条样本
        # ########## step 1 ##########
        print("忽略了: {} 条样本".format(count))
        return result
