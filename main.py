# -*- coding: utf-8 -*-
import argparse
from sklearn.model_selection import train_test_split
from flyai.framework import FlyAI
from flyai.data_helper import DataHelper
from flyai.utils import remote_helper
from customer_layers import *
from path import MODEL_PATH, DATA_PATH

'''
此项目为FlyAI2.0新版本框架，数据读取，评估方式与之前不同
2.0框架不再限制数据如何读取
样例代码仅供参考学习，可以自己修改实现逻辑。
模版项目下载支持 PyTorch、Tensorflow、Keras、MXNET、scikit-learn等机器学习框架
第一次使用请看项目中的：FlyAI2.0竞赛框架使用说明.html
使用FlyAI提供的预训练模型可查看：https://www.flyai.com/models
学习资料可查看文档中心：https://doc.flyai.com/
常见问题：https://doc.flyai.com/question.html
遇到问题不要着急，添加小姐姐微信，扫描项目里面的：FlyAI小助手二维码-小姐姐在线解答您的问题.png
'''

# 项目的超参，不使用可以删除
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--EPOCHS", default=100, type=int, help="train epochs")
parser.add_argument("-b", "--BATCH", default=32, type=int, help="batch size")
args = parser.parse_args()


class Main(FlyAI):
    '''
    项目中必须继承FlyAI类，否则线上运行会报错。
    '''
    def download_data(self):
        # 下载数据
        data_helper = DataHelper()
        data_helper.download_from_ids("TextSQL")
        # 二选一或者根据app.json的配置下载文件
        # data_helper.download_from_json()
        print('=*=数据下载完成=*=')

    def deal_with_data(self):
        '''
        处理数据，没有可不写。
        :return:
        '''
        # 加载数据
        self.data = pd.read_csv(os.path.join(DATA_PATH, 'TextSQL/train.csv'))
        # 划分训练集、测试集
        self.train_data, self.valid_data = train_test_split(self.data, test_size=0.2, random_state=6, shuffle=True)
        print('=*=数据处理完成=*=')

    def train(self):
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        # 超参数
        BATCH_SIZE = args.BATCH
        TRAIN_ENTRY = (True, True, True)  # (AGG, SEL, COND)
        TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY
        learning_rate = 1e-4
        if torch.cuda.is_available():
            GPU = True
        else:
            GPU = False
        # './data/input/model/glove.twitter.27B.zip'
        path = remote_helper.get_remote_date('https://www.flyai.com/m/glove.twitter.27B.zip')
        glove_path = os.path.split(path)[0]  # './data/input/model'
        word_emb = load_word_emb(os.path.join(glove_path, 'glove.twitter.27B.25d.txt'))

        sql_model = SQLNet(word_emb, N_word=25, use_ca=True, gpu=GPU)
        optimizer = torch.optim.Adam(sql_model.parameters(), lr=learning_rate, weight_decay=0)

        agg_m, sel_m, cond_m = best_model_name(MODEL_PATH)
        if TRAIN_AGG:
            torch.save(sql_model.agg_pred.state_dict(), agg_m)

        if TRAIN_SEL:
            torch.save(sql_model.sel_pred.state_dict(), sel_m)

        if TRAIN_COND:
            torch.save(sql_model.cond_pred.state_dict(), cond_m)

        # 按批次加载数据
        best_agg_acc, best_sel_acc, best_cond_acc = 0.0, 0.0, 0.0
        best_agg_idx, best_sel_idx, best_cond_idx = 0, 0, 0
        sql_val, table_val = get_val_batch(self.valid_data, batch_size=100)
        sql_data_v, table_data_v = sql_val
        sql_v = table_val
        batch_nums = int(self.train_data.shape[0]/BATCH_SIZE)
        for epoch in range(args.EPOCHS):
            for batch_i, (sql_train, table_train) in \
                    enumerate(get_train_batches(self.train_data, batch_size=BATCH_SIZE)):
                sql_data_t, table_data_t = sql_train
                sql_t = table_train

                step_train(sql_model, optimizer, sql_data_t, table_data_t, sql_t, TRAIN_ENTRY)
                tol_acc_t, one_acc_t = compute_acc(sql_model, sql_data_t, table_data_t, sql_t, TRAIN_ENTRY)

                current_steps = 'Epoch: {} | steps: {}/{}'.format(epoch+1, batch_i+1, batch_nums)
                print('{} | The Total Acc: {} | The One acc: {}'.format(current_steps, tol_acc_t, one_acc_t))

            tol_acc_v, one_acc_v = compute_acc(sql_model, sql_data_v, table_data_v, sql_v, TRAIN_ENTRY)
            if TRAIN_AGG:
                if one_acc_v[0] > best_agg_acc:
                    best_agg_acc = one_acc_v[0]
                    best_agg_idx += 1
                    torch.save(sql_model.agg_pred.state_dict(), agg_m)

            if TRAIN_SEL:
                if one_acc_v[1] > best_sel_acc:
                    best_sel_acc = one_acc_v[1]
                    best_sel_idx += 1
                    torch.save(sql_model.sel_pred.state_dict(), sel_m)

            if TRAIN_COND:
                if one_acc_v[2] > best_cond_acc:
                    best_cond_acc = one_acc_v[2]
                    best_cond_idx += 1
                    torch.save(sql_model.cond_pred.state_dict(), cond_m)

            print('Best val acc: %s\nOn epoch individually %s' % (
                (best_agg_acc, best_sel_acc, best_cond_acc),
                (best_agg_idx, best_sel_idx, best_cond_idx)))

            print('Save Model Done for epoch:[{}]!'.format(epoch + 1))


if __name__ == '__main__':
    main = Main()
    main.download_data()
    main.deal_with_data()
    main.train()

    exit(0)
