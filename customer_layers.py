# -*- coding: utf-8 -*-
import os
import re
import records
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable
from babel.numbers import parse_decimal, NumberFormatError


class DBEngine:
    def __init__(self, fdb):
        # fdb = 'data/test.db'
        self.db = records.Database('sqlite:///{}'.format(fdb))
        self.schema_re = re.compile(r'\((.+)\)')
        self.num_re = re.compile(r'[-+]?\d*\.\d+|\d+')

        self.agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        self.cond_ops = ['=', '>', '<', 'OP']

    def execute_query(self, table_id, query, *args, **kwargs):
        return self.execute(table_id, query.sel_index, query.agg_index, query.conditions, *args, **kwargs)

    def execute(self, table_id, select_index, aggregation_index, conditions, lower=True):
        if not table_id.startswith('table'):
            table_id = 'table_{}'.format(table_id.replace('-', '_'))
        table_info = self.db.query('SELECT sql from sqlite_master WHERE tbl_name = :name',
                                   name=table_id).all()[0].sql.replace('\n', '')
        schema_str = self.schema_re.findall(table_info)[0]
        schema = {}
        for tup in schema_str.split(', '):
            c, t = tup.split()
            schema[c] = t
        select = 'col{}'.format(select_index)
        agg = self.agg_ops[aggregation_index]
        if agg:
            select = '{}({})'.format(agg, select)
        where_clause = []
        where_map = {}
        for col_index, op, val in conditions:
            if lower and isinstance(val, str):
                val = val.lower()
            if schema['col{}'.format(col_index)] == 'real' and not isinstance(val, (int, float)):
                try:
                    val = float(parse_decimal(val))
                except NumberFormatError as e:
                    val = float(self.num_re.findall(val)[0])
            where_clause.append('col{} {} :col{}'.format(col_index, self.cond_ops[op], col_index))
            where_map['col{}'.format(col_index)] = val
        where_str = ''
        if where_clause:
            where_str = 'WHERE ' + ' AND '.join(where_clause)
        query = 'SELECT {} AS result FROM {} {}'.format(select, table_id, where_str)
        # print(query)
        out = self.db.query(query, **where_map)
        return [o.result for o in out]


def load_word_emb(file_name):
    print('Loading word embedding from %s' % file_name)
    ret = {}
    with open(file_name, 'r', encoding='utf-8') as inf:
        for idx, line in enumerate(inf):
            info = line.strip().split(' ')
            if info[0].lower() not in ret:
                ret[info[0]] = np.array(list(map(lambda x: float(x), info[1:])))
    return ret


def best_model_name(model_path):
    new_data = 'best'
    mode = 'sqlnet'

    if os.path.exists(model_path) is False:
        os.makedirs(model_path)
    agg_model_name = os.path.join(model_path, '%s_%s.agg_model' % (new_data, mode))
    sel_model_name = os.path.join(model_path, '%s_%s.sel_model' % (new_data, mode))
    cond_model_name = os.path.join(model_path, '%s_%s.cond_model' % (new_data, mode))

    return agg_model_name, sel_model_name, cond_model_name


def step_train(model, optimizer, sql_data, table_data, sql_l, pred_entry):
    model.train()

    # 构造数据
    q_seq = [eval(str(x))['question_tok'] for x in sql_data]
    query_seq = [eval(str(x))['query_tok'] for x in sql_data]
    # question = [eval(str(x))['question'] for x in sql_train]

    ans_seq = [(eval(str(sql))['agg'], eval(str(sql))['sel'], len(eval(str(sql))['conds']),
                tuple(x[0] for x in eval(str(sql))['conds']), tuple(x[1] for x in eval(str(sql))['conds']))
               for sql in sql_l]
    gt_cond_seq = [eval(str(sql))['conds'] for sql in sql_l]

    col_seq = [eval(str(x))['header_tok'] for x in table_data]
    col_num = [len(eval(str(x))['header']) for x in table_data]
    # header = [eval(str(x))['header'] for x in table_data_t]

    gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
    gt_sel_seq = [x[1] for x in ans_seq]
    score = model.forward(q_seq, col_seq, col_num, pred_entry,
                          gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
    loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
    loss.data.cpu().numpy()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


def compute_acc(model, sql_data, table_data, sql_l, pred_entry):
    model.eval()

    # 构造数据
    # print('--', sql_data[0])
    q_seq = [eval(str(x))['question_tok'] for x in sql_data]
    # query_seq = [eval(str(x))['query_tok'] for x in sql_train]
    question = [eval(str(x))['question'] for x in sql_data]

    ans_seq = [(eval(str(sql))['agg'], eval(str(sql))['sel'], len(eval(str(sql))['conds']),
                tuple(x[0] for x in eval(str(sql))['conds']), tuple(x[1] for x in eval(str(sql))['conds']))
               for sql in sql_l]

    # gt_cond_seq = [eval(str(sql))['conds'] for sql in sql_t]

    col_seq = [eval(str(x))['header_tok'] for x in table_data]
    col_num = [len(eval(str(x))['header']) for x in table_data]
    header = [eval(str(x))['header'] for x in table_data]

    raw_data = [(question[i], header[i], eval(str(sql_data[i]))['query']) for i in range(len(sql_data))]
    sql_l = [eval(str(sql)) for sql in sql_l]

    gt_sel_seq = [x[1] for x in ans_seq]
    score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel=gt_sel_seq)
    pred_queries = model.gen_query(score, q_seq, col_seq, question, header, pred_entry)
    one_err, tot_err = model.check_acc(raw_data, pred_queries, sql_l, pred_entry)

    one_acc_num = (len(sql_data) - one_err)
    tot_acc_num = (len(sql_data) - tot_err)

    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)


def get_val_batch(data, batch_size):
    sql_data, table_data, sql = list(), list(), list()
    for ind, row in data.iterrows():
        sql_data.append(np.array(eval(row['sql_data'])))
        table_data.append(np.array(eval(row['table_data'])))
        sql.append(np.array(eval(row['sql'])))

    sqld_batch = sql_data[0: batch_size]
    tabled_batch = table_data[0: batch_size]
    sql_batch = sql[0: batch_size]
    return [np.array(sqld_batch), np.array(tabled_batch)], sql_batch


def get_train_batches(data, batch_size):
    sql_data, table_data, sql = list(), list(), list()
    for ind, row in data.iterrows():
        sql_data.append(np.array(eval(row['sql_data'])))
        table_data.append(np.array(eval(row['table_data'])))
        sql.append(np.array(eval(row['sql'])))

    for batch_i in range(0, len(sql_data)//batch_size):
        start_i = batch_i * batch_size
        sqld_batch = sql_data[start_i: start_i + batch_size]
        tabled_batch = table_data[start_i: start_i + batch_size]
        sql_batch = sql[start_i: start_i + batch_size]
        yield [np.array(sqld_batch), np.array(tabled_batch)], sql_batch


def predict_test(model, sql_data, table_data, pred_entry):
    model.eval()

    # 构造数据
    q_seq = [eval(str(x))['question_tok'] for x in sql_data]
    question = [eval(str(x))['question'] for x in sql_data]

    col_seq = [eval(str(x))['header_tok'] for x in table_data]
    col_num = [len(eval(str(x))['header']) for x in table_data]

    # q_seq = [eval(str(sql_data[0]))['question_tok']]
    # question = [eval(str(sql_data[0]))['question']]
    #
    # col_seq = [eval(str(table_data[0]))['header_tok']]
    # col_num = [len(eval(str(table_data[0]))['header'])]

    gt_sel = [0 for _ in sql_data]
    score = model.forward(q_seq, col_seq, col_num, pred_entry, gt_sel=gt_sel)
    sql_pred = model.gen_query(score, q_seq, None, question, None, pred_entry)

    return sql_pred


class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK, our_model):
        super(WordEmbedding, self).__init__()
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        self.word_emb = word_emb
        print("Using fixed embedding")

    def gen_x_batch(self, q, col):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (one_q, one_col) in enumerate(zip(q, col)):
            q_val = list(map(lambda x: self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), one_q))
            if self.our_model:
                val_embs.append([np.zeros(self.N_word, dtype=np.float32)] +
                                q_val + [np.zeros(self.N_word, dtype=np.float32)])  # <BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                one_col_all = [x for toks in one_col for x in toks+[',']]
                col_val = list(map(lambda x: self.word_emb.get(x, np.zeros(self.N_word,
                                                                           dtype=np.float32)), one_col_all))
                val_embs.append([np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] +
                                col_val + [np.zeros(self.N_word, dtype=np.float32)] + q_val
                                + [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, one_cols in enumerate(cols):
            names = names + one_cols
            col_len[b] = len(one_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, one_str in enumerate(str_list):
            val = [self.word_emb.get(x, np.zeros(
                self.N_word, dtype=np.float32)) for x in one_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
        for i in range(B):
            for t in range(len(val_embs[i])):
                val_emb_array[i, t, :] = val_embs[i][t]
        val_inp = torch.from_numpy(val_emb_array)
        if self.gpu:
            val_inp = val_inp.cuda()
        val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


def run_lstm(lstm, inp, inp_len, hidden=None):
    # Run the LSTM using packed sequence.
    # This requires to first sort the input according to its length.
    sort_perm = np.array(sorted(range(len(inp_len)), key=lambda k: inp_len[k], reverse=True))
    sort_inp_len = inp_len[sort_perm]
    sort_perm_inv = np.argsort(sort_perm)
    if inp.is_cuda:
        sort_perm = torch.LongTensor(sort_perm).cuda()
        sort_perm_inv = torch.LongTensor(sort_perm_inv).cuda()

    lstm_inp = nn.utils.rnn.pack_padded_sequence(inp[sort_perm], sort_inp_len, batch_first=True)
    if hidden is None:
        lstm_hidden = None
    else:
        lstm_hidden = (hidden[0][:, sort_perm], hidden[1][:, sort_perm])

    sort_ret_s, sort_ret_h = lstm(lstm_inp, lstm_hidden)
    ret_s = nn.utils.rnn.pad_packed_sequence(
            sort_ret_s, batch_first=True)[0][sort_perm_inv]
    ret_h = (sort_ret_h[0][:, sort_perm_inv], sort_ret_h[1][:, sort_perm_inv])
    return ret_s, ret_h


def col_name_encode(name_inp_var, name_len, col_len, enc_lstm):
    # Encode the columns.
    # The embedding of a column name is the last state of its LSTM output.
    name_hidden, _ = run_lstm(enc_lstm, name_inp_var, name_len)
    name_out = name_hidden[tuple(range(len(name_len))), name_len-1]
    ret = torch.FloatTensor(len(col_len), max(col_len), name_out.size()[1]).zero_()
    if name_out.is_cuda:
        ret = ret.cuda()

    st = 0
    for idx, cur_len in enumerate(col_len):
        ret[idx, :cur_len] = name_out.data[st:st+cur_len]
        st += cur_len
    ret_var = Variable(ret)

    return ret_var, col_len


class AggPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, use_ca):
        super(AggPredictor, self).__init__()
        self.use_ca = use_ca

        self.agg_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)
        if use_ca:
            print("Using column attention on aggregator predicting")
            self.agg_col_name_enc = nn.LSTM(input_size=N_word,
                                            hidden_size=int(N_h/2), num_layers=N_depth,
                                            batch_first=True, dropout=0.3, bidirectional=True)
            self.agg_att = nn.Linear(N_h, N_h)
        else:
            print("Not using column attention on aggregator predicting")
            self.agg_att = nn.Linear(N_h, 1)
        self.agg_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 6))
        self.softmax = nn.Softmax()

    def forward(self, x_emb_var, x_len, col_inp_var=None, col_name_len=None,
                col_len=None, col_num=None, gt_sel=None):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        h_enc, _ = run_lstm(self.agg_lstm, x_emb_var, x_len)
        if self.use_ca:
            e_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.agg_col_name_enc)
            chosen_sel_idx = torch.LongTensor(gt_sel)
            aux_range = torch.LongTensor(range(len(gt_sel)))
            if x_emb_var.is_cuda:
                chosen_sel_idx = chosen_sel_idx.cuda()
                aux_range = aux_range.cuda()
            chosen_e_col = e_col[aux_range, chosen_sel_idx]
            att_val = torch.bmm(self.agg_att(h_enc),
                    chosen_e_col.unsqueeze(2)).squeeze()
        else:
            att_val = self.agg_att(h_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                att_val[idx, num:] = -100
        att = self.softmax(att_val)

        K_agg = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
        agg_score = self.agg_out(K_agg)
        return agg_score


class SelPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_tok_num, use_ca):
        super(SelPredictor, self).__init__()
        self.use_ca = use_ca
        self.max_tok_num = max_tok_num
        self.sel_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                num_layers=N_depth, batch_first=True,
                                dropout=0.3, bidirectional=True)
        if use_ca:
            print("Using column attention on selection predicting")
            self.sel_att = nn.Linear(N_h, N_h)
        else:
            print("Not using column attention on selection predicting")
            self.sel_att = nn.Linear(N_h, 1)
        self.sel_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                        num_layers=N_depth, batch_first=True,
                                        dropout=0.3, bidirectional=True)
        self.sel_out_K = nn.Linear(N_h, N_h)
        self.sel_out_col = nn.Linear(N_h, N_h)
        self.sel_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))
        self.softmax = nn.Softmax()

    def forward(self, x_emb_var, x_len, col_inp_var,
            col_name_len, col_len, col_num):
        B = len(x_emb_var)
        max_x_len = max(x_len)

        e_col, _ = col_name_encode(col_inp_var, col_name_len,
                col_len, self.sel_col_name_enc)

        if self.use_ca:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = torch.bmm(e_col, self.sel_att(h_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, :, num:] = -100
            att = self.softmax(att_val.view((-1, max_x_len))).view(
                    B, -1, max_x_len)
            K_sel_expand = (h_enc.unsqueeze(1) * att.unsqueeze(3)).sum(2)
        else:
            h_enc, _ = run_lstm(self.sel_lstm, x_emb_var, x_len)
            att_val = self.sel_att(h_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    att_val[idx, num:] = -100
            att = self.softmax(att_val)
            K_sel = (h_enc * att.unsqueeze(2).expand_as(h_enc)).sum(1)
            K_sel_expand=K_sel.unsqueeze(1)

        sel_score = self.sel_out( self.sel_out_K(K_sel_expand) + \
                self.sel_out_col(e_col) ).squeeze()
        max_col_num = max(col_num)
        for idx, num in enumerate(col_num):
            if num < max_col_num:
                sel_score[idx, num:] = -100

        return sel_score


class SQLNetCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num, use_ca, gpu):
        super(SQLNetCondPredictor, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num
        self.max_col_num = max_col_num
        self.gpu = gpu
        self.use_ca = use_ca

        self.cond_num_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)
        self.cond_num_att = nn.Linear(N_h, 1)
        self.cond_num_out = nn.Sequential(nn.Linear(N_h, N_h),
                nn.Tanh(), nn.Linear(N_h, 5))
        self.cond_num_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                         num_layers=N_depth, batch_first=True,
                                         dropout=0.3, bidirectional=True)
        self.cond_num_col_att = nn.Linear(N_h, 1)
        self.cond_num_col2hid1 = nn.Linear(N_h, 2*N_h)
        self.cond_num_col2hid2 = nn.Linear(N_h, 2*N_h)

        self.cond_col_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)
        if use_ca:
            print("Using column attention on where predicting")
            self.cond_col_att = nn.Linear(N_h, N_h)
        else:
            print("Not using column attention on where predicting")
            self.cond_col_att = nn.Linear(N_h, 1)
        self.cond_col_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                         num_layers=N_depth, batch_first=True,
                                         dropout=0.3, bidirectional=True)
        self.cond_col_out_K = nn.Linear(N_h, N_h)
        self.cond_col_out_col = nn.Linear(N_h, N_h)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.cond_op_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                    num_layers=N_depth, batch_first=True,
                                    dropout=0.3, bidirectional=True)
        if use_ca:
            self.cond_op_att = nn.Linear(N_h, N_h)
        else:
            self.cond_op_att = nn.Linear(N_h, 1)
        self.cond_op_out_K = nn.Linear(N_h, N_h)
        self.cond_op_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                        num_layers=N_depth, batch_first=True,
                                        dropout=0.3, bidirectional=True)
        self.cond_op_out_col = nn.Linear(N_h, N_h)
        self.cond_op_out = nn.Sequential(nn.Linear(N_h, N_h), nn.Tanh(),
                nn.Linear(N_h, 3))

        self.cond_str_lstm = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                     num_layers=N_depth, batch_first=True,
                                     dropout=0.3, bidirectional=True)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_tok_num,
                                        hidden_size=N_h, num_layers=N_depth,
                                        batch_first=True, dropout=0.3)
        self.cond_str_name_enc = nn.LSTM(input_size=N_word, hidden_size=int(N_h/2),
                                         num_layers=N_depth, batch_first=True,
                                         dropout=0.3, bidirectional=True)
        self.cond_str_out_g = nn.Linear(N_h, N_h)
        self.cond_str_out_h = nn.Linear(N_h, N_h)
        self.cond_str_out_col = nn.Linear(N_h, N_h)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax()

    def gen_gt_batch(self, split_tok_seq):
        B = len(split_tok_seq)
        max_len = max([max([len(tok) for tok in tok_seq]+[0]) for
            tok_seq in split_tok_seq]) - 1  # The max seq len in the batch.
        if max_len < 1:
            max_len = 1
        ret_array = np.zeros((
            B, 4, max_len, self.max_tok_num), dtype=np.float32)
        ret_len = np.zeros((B, 4))
        for b, tok_seq in enumerate(split_tok_seq):
            idx = 0
            for idx, one_tok_seq in enumerate(tok_seq):
                out_one_tok_seq = one_tok_seq[:-1]
                ret_len[b, idx] = len(out_one_tok_seq)
                for t, tok_id in enumerate(out_one_tok_seq):
                    ret_array[b, idx, t, tok_id] = 1
            if idx < 3:
                ret_array[b, idx+1:, 0, 1] = 1
                ret_len[b, idx+1:] = 1

        ret_inp = torch.from_numpy(ret_array)
        if self.gpu:
            ret_inp = ret_inp.cuda()
        ret_inp_var = Variable(ret_inp)

        return ret_inp_var, ret_len  # [B, IDX, max_len, max_tok_num]

    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len,
            col_len, col_num, gt_where, gt_cond, reinforce):
        max_x_len = max(x_len)
        B = len(x_len)
        if reinforce:
            raise NotImplementedError('Our model doesn\'t have RL')

        # Predict the number of conditions
        # First use column embeddings to calculate the initial hidden unit
        # Then run the LSTM and predict condition number.
        e_num_col, col_num = col_name_encode(col_inp_var, col_name_len, col_len, self.cond_num_name_enc)
        num_col_att_val = self.cond_num_col_att(e_num_col).squeeze()
        for idx, num in enumerate(col_num):
            if num < max(col_num):
                num_col_att_val[idx, num:] = -100
        num_col_att = self.softmax(num_col_att_val)
        K_num_col = (e_num_col * num_col_att.unsqueeze(2)).sum(1)
        cond_num_h1 = self.cond_num_col2hid1(K_num_col).view(B, 4, int(self.N_h/2)).transpose(0, 1).contiguous()
        cond_num_h2 = self.cond_num_col2hid2(K_num_col).view(B, 4, int(self.N_h/2)).transpose(0, 1).contiguous()

        h_num_enc, _ = run_lstm(self.cond_num_lstm, x_emb_var, x_len, hidden=(cond_num_h1, cond_num_h2))

        num_att_val = self.cond_num_att(h_num_enc).squeeze()

        for idx, num in enumerate(x_len):
            if num < max_x_len:
                num_att_val[idx, num:] = -100
        num_att = self.softmax(num_att_val)

        K_cond_num = (h_num_enc * num_att.unsqueeze(2).expand_as(
            h_num_enc)).sum(1)
        cond_num_score = self.cond_num_out(K_cond_num)

        # Predict the columns of conditions
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.cond_col_name_enc)

        h_col_enc, _ = run_lstm(self.cond_col_lstm, x_emb_var, x_len)
        if self.use_ca:
            col_att_val = torch.bmm(e_cond_col, self.cond_col_att(h_col_enc).transpose(1, 2))
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, :, num:] = -100
            col_att = self.softmax(col_att_val.view(
                (-1, max_x_len))).view(B, -1, max_x_len)
            K_cond_col = (h_col_enc.unsqueeze(1) * col_att.unsqueeze(3)).sum(2)
        else:
            col_att_val = self.cond_col_att(h_col_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    col_att_val[idx, num:] = -100
            col_att = self.softmax(col_att_val)
            K_cond_col = (h_col_enc * col_att_val.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(K_cond_col) +
                self.cond_col_out_col(e_cond_col)).squeeze()
        max_col_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_col_num:
                cond_col_score[b, num:] = -100

        # Predict the operator of conditions
        chosen_col_gt = []
        if gt_cond is None:
            cond_nums = np.argmax(cond_num_score.data.cpu().numpy(), axis=1)
            col_scores = cond_col_score.data.cpu().numpy()
            chosen_col_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]])
                             for b in range(len(cond_nums))]
        else:
            chosen_col_gt = [ [x[0] for x in one_gt_cond] for one_gt_cond in gt_cond]

        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.cond_op_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x]
                for x in chosen_col_gt[b]] + [e_cond_col[b, 0]] *
                (4 - len(chosen_col_gt[b])))  # Pad the columns to maximum (4)
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        h_op_enc, _ = run_lstm(self.cond_op_lstm, x_emb_var, x_len)
        if self.use_ca:
            op_att_val = torch.matmul(self.cond_op_att(h_op_enc).unsqueeze(1), col_emb.unsqueeze(3)).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, :, num:] = -100
            op_att = self.softmax(op_att_val.view(B*4, -1)).view(B, 4, -1)
            K_cond_op = (h_op_enc.unsqueeze(1) * op_att.unsqueeze(3)).sum(2)
        else:
            op_att_val = self.cond_op_att(h_op_enc).squeeze()
            for idx, num in enumerate(x_len):
                if num < max_x_len:
                    op_att_val[idx, num:] = -100
            op_att = self.softmax(op_att_val)
            K_cond_op = (h_op_enc * op_att.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_op_score = self.cond_op_out(self.cond_op_out_K(K_cond_op) + self.cond_op_out_col(col_emb)).squeeze()

        # Predict the string of conditions
        h_str_enc, _ = run_lstm(self.cond_str_lstm, x_emb_var, x_len)
        e_cond_col, _ = col_name_encode(col_inp_var, col_name_len, col_len, self.cond_str_name_enc)
        col_emb = []
        for b in range(B):
            cur_col_emb = torch.stack([e_cond_col[b, x]
                for x in chosen_col_gt[b]] +
                [e_cond_col[b, 0]] * (4 - len(chosen_col_gt[b])))
            col_emb.append(cur_col_emb)
        col_emb = torch.stack(col_emb)

        if gt_where is not None:
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where)
            g_str_s_flat, _ = self.cond_str_decoder(gt_tok_seq.view(B*4, -1, self.max_tok_num))
            g_str_s = g_str_s_flat.contiguous().view(B, 4, -1, self.N_h)

            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            g_ext = g_str_s.unsqueeze(3)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)

            cond_str_score = self.cond_str_out(
                    self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext) +
                    self.cond_str_out_col(col_ext)).squeeze()
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100
        else:
            h_ext = h_str_enc.unsqueeze(1).unsqueeze(1)
            col_ext = col_emb.unsqueeze(2).unsqueeze(2)
            scores = []

            t = 0
            init_inp = np.zeros((B*4, 1, self.max_tok_num), dtype=np.float32)
            init_inp[:, 0, 0] = 1  # Set the <BEG> token
            if self.gpu:
                cur_inp = Variable(torch.from_numpy(init_inp).cuda())
            else:
                cur_inp = Variable(torch.from_numpy(init_inp))
            cur_h = None
            while t < 50:
                if cur_h:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp, cur_h)
                else:
                    g_str_s_flat, cur_h = self.cond_str_decoder(cur_inp)
                g_str_s = g_str_s_flat.view(B, 4, 1, self.N_h)
                g_ext = g_str_s.unsqueeze(3)

                cur_cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(h_ext) + self.cond_str_out_g(g_ext)
                        + self.cond_str_out_col(col_ext)).squeeze()
                for b, num in enumerate(x_len):
                    if num < max_x_len:
                        cur_cond_str_score[b, :, num:] = -100
                scores.append(cur_cond_str_score)

                _, ans_tok_var = cur_cond_str_score.view(B*4, max_x_len).max(1)
                ans_tok = ans_tok_var.data.cpu()
                data = torch.zeros(B*4, self.max_tok_num).scatter_(
                        1, ans_tok.unsqueeze(1), 1)
                if self.gpu:  # To one-hot
                    cur_inp = Variable(data.cuda())
                else:
                    cur_inp = Variable(data)
                cur_inp = cur_inp.unsqueeze(1)

                t += 1

            cond_str_score = torch.stack(scores, 2)
            for b, num in enumerate(x_len):
                if num < max_x_len:
                    cond_str_score[b, :, :, num:] = -100  # [B, IDX, T, TOK_NUM]

        cond_score = (cond_num_score, cond_col_score, cond_op_score, cond_str_score)

        return cond_score


class SQLNet(nn.Module):
    def __init__(self, word_emb, N_word, N_h=100, N_depth=2, gpu=False, use_ca=True):
        super(SQLNet, self).__init__()
        self.use_ca = use_ca
        self.gpu = gpu
        self.N_h = N_h
        self.N_depth = N_depth
        self.max_col_num = 45
        self.max_tok_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        # Word embedding
        self.embed_layer = WordEmbedding(word_emb, N_word, gpu, self.SQL_TOK, our_model=True)

        # Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth, use_ca=use_ca)

        # Predict selected column
        self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num, use_ca=use_ca)

        # Predict number of cond
        self.cond_pred = SQLNetCondPredictor(N_word, N_h, N_depth, self.max_col_num, self.max_tok_num, use_ca, gpu)

        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax()
        self.bce_logit = nn.BCEWithLogitsLoss()
        if gpu:
            self.cuda()

    def generate_gt_where_seq(self, q, col, query):
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            cur_values = []
            st = cur_query.index(u'WHERE') + 1 if u'WHERE' in cur_query else len(cur_query)
            all_toks = ['<BEG>'] + cur_q + ['<END>']
            while st < len(cur_query):
                ed = len(cur_query) if 'AND' not in cur_query[st:] else cur_query[st:].index('AND') + st
                if 'EQL' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('EQL') + st
                elif 'GT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('GT') + st
                elif 'LT' in cur_query[st:ed]:
                    op = cur_query[st:ed].index('LT') + st
                else:
                    raise RuntimeError("No operator in it!")
                this_str = ['<BEG>'] + cur_query[op + 1:ed] + ['<END>']
                cur_seq = [all_toks.index(s) if s in all_toks else 0 for s in this_str]
                cur_values.append(cur_seq)
                st = ed + 1
            ret_seq.append(cur_values)
        return ret_seq

    def forward(self, q, col, col_num, pred_entry, gt_where=None, gt_cond=None, reinforce=False, gt_sel=None):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None

        # Predict aggregator
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)
        col_inp_var, col_name_len, col_len = self.embed_layer.gen_col_batch(col)
        max_x_len = max(x_len)
        if pred_agg:
            agg_score = self.agg_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num, gt_sel=gt_sel)

        if pred_sel:
            sel_score = self.sel_pred(x_emb_var, x_len, col_inp_var, col_name_len, col_len, col_num)

        if pred_cond:
            cond_score = self.cond_pred(x_emb_var, x_len, col_inp_var, col_name_len,
                                        col_len, col_num, gt_where, gt_cond, reinforce=reinforce)

        return (agg_score, sel_score, cond_score)

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        loss = 0
        torch.LongTensor()
        if pred_agg:
            agg_truth = list(map(lambda x: x[0], truth_num))
            data = torch.from_numpy(np.array(agg_truth, dtype=np.int64))
            if self.gpu:
                agg_truth_var = Variable(data.cuda())
            else:
                agg_truth_var = Variable(data)

            loss += self.CE(agg_score, agg_truth_var)

        if pred_sel:
            sel_truth = list(map(lambda x: x[1], truth_num))
            data = torch.from_numpy(np.array(sel_truth, dtype=np.int64))
            if self.gpu:
                sel_truth_var = Variable(data.cuda())
            else:
                sel_truth_var = Variable(data)

            loss += self.CE(sel_score, sel_truth_var)

        if pred_cond:
            B = len(truth_num)
            cond_num_score, cond_col_score, cond_op_score, cond_str_score = cond_score
            # Evaluate the number of conditions
            cond_num_truth = list(map(lambda x: x[2], truth_num))
            data = torch.from_numpy(np.array(cond_num_truth, dtype=np.int64))
            if self.gpu:
                cond_num_truth_var = Variable(data.cuda())
            else:
                cond_num_truth_var = Variable(data)
            loss += self.CE(cond_num_score, cond_num_truth_var)

            # Evaluate the columns of conditions
            T = len(cond_col_score[0])
            truth_prob = np.zeros((B, T), dtype=np.float32)
            for b in range(B):
                if len(truth_num[b][3]) > 0:
                    truth_prob[b][list(truth_num[b][3])] = 1
            data = torch.from_numpy(truth_prob)
            if self.gpu:
                cond_col_truth_var = Variable(data.cuda())
            else:
                cond_col_truth_var = Variable(data)

            sigm = nn.Sigmoid()
            cond_col_prob = sigm(cond_col_score)
            bce_loss = -torch.mean(3 * (cond_col_truth_var * torch.log(cond_col_prob + 1e-10))
                                   + (1 - cond_col_truth_var) * torch.log(1 - cond_col_prob + 1e-10))
            loss += bce_loss

            # Evaluate the operator of conditions
            for b in range(len(truth_num)):
                if len(truth_num[b][4]) == 0:
                    continue
                data = torch.from_numpy(np.array(truth_num[b][4], dtype=np.int64))
                if self.gpu:
                    cond_op_truth_var = Variable(data.cuda())
                else:
                    cond_op_truth_var = Variable(data)
                cond_op_pred = cond_op_score[b, :len(truth_num[b][4])]
                loss += (self.CE(cond_op_pred, cond_op_truth_var) / len(truth_num))

            # Evaluate the strings of conditions
            for b in range(len(gt_where)):
                for idx in range(len(gt_where[b])):
                    cond_str_truth = gt_where[b][idx]
                    if len(cond_str_truth) == 1:
                        continue
                    data = torch.from_numpy(np.array(cond_str_truth[1:], dtype=np.int64))
                    if self.gpu:
                        cond_str_truth_var = Variable(data.cuda())
                    else:
                        cond_str_truth_var = Variable(data)
                    str_end = len(cond_str_truth) - 1
                    cond_str_pred = cond_str_score[b, idx, :str_end]
                    loss += (self.CE(cond_str_pred, cond_str_truth_var) / (len(gt_where) * len(gt_where[b])))

        return loss

    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        def pretty_print(vis_data):
            print('question:', vis_data[0])
            print('headers: (%s)' % (' || '.join(vis_data[1])))
            print('query:', vis_data[2])

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(header[cond[0]] + ' ' +
                                self.COND_OPS[cond[1]] + ' ' + cond[2].lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = 0.0
        cond_num_err = cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(x[0] for x in cond_pred) != \
                        set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(
                        x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and str(cond_gt[gt_idx][2]).lower() != str(cond_pred[idx][2]).lower():
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err

    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry, reinforce=False, verbose=False):
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-': '(',
                       '-RRB-': ')',
                       '-LSB-': '[',
                       '-RSB-': ']',
                       '``': '"',
                       '\'\'': '"',
                       '--': u'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) \
                        and (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        ret_queries = []
        if pred_agg:
            B = len(agg_score)
        elif pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0])
        for b in range(B):
            cur_query = {}
            if pred_agg:
                cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            if pred_cond:
                cur_query['conds'] = []
                cond_num_score, cond_col_score, cond_op_score, cond_str_score = \
                    [x.data.cpu().numpy() for x in cond_score]
                cond_num = np.argmax(cond_num_score[b])
                all_toks = ['<BEG>'] + q[b] + ['<END>']
                max_idxes = np.argsort(-cond_col_score[b])[:cond_num]
                for idx in range(cond_num):
                    cur_cond = []
                    cur_cond.append(max_idxes[idx])
                    cur_cond.append(np.argmax(cond_op_score[b][idx]))
                    cur_cond_str_toks = []
                    for str_score in cond_str_score[b][idx]:
                        str_tok = np.argmax(str_score[:len(all_toks)])
                        str_val = all_toks[str_tok]
                        if str_val == '<END>':
                            break
                        cur_cond_str_toks.append(str_val)
                    cur_cond.append(merge_tokens(cur_cond_str_toks, raw_q[b]))
                    cur_query['conds'].append(cur_cond)
            ret_queries.append(cur_query)

        return ret_queries
