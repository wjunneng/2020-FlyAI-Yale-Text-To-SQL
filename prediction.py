# -*- coding: utf-8 -*
from flyai.framework import FlyAI
from customer_layers import *
from path import MODEL_PATH
from flyai.utils import remote_helper

N_word = 25
TEST_ENTRY = (True, True, True)  # (AGG, SEL, COND)
TENSORFLOW_MODEL_DIR = 'best'
if torch.cuda.is_available():
    GPU = True
else:
    GPU = False


class Prediction(FlyAI):
    def load_model(self):
        '''
        模型初始化，必须在构造方法中加载模型
        '''
        torch.cuda.empty_cache()
        path = remote_helper.get_remote_date('https://www.flyai.com/m/glove.twitter.27B.zip')
        glove_path = os.path.split(path)[0]
        self.word_emb = load_word_emb(os.path.join(glove_path, 'glove.twitter.27B.25d.txt'))
        self.model = SQLNet(self.word_emb, N_word=N_word, gpu=GPU)

    def predict(self, datas):
        '''
        模型预测返回结果
        :param datas: datas是app.json中 model-》input设置的输入，项目设定不可以修改，输入一条数据是字典类型
        datas: [{'sql_data': "{'question': 'Evening ...', ...'question_tok': ['evening', ...]}",
                 'table_data': "{'header_tok': [['country'], ...], ...'types': ['text', ...]}"},
                {'sql_data': "{'question': 'Evening ...', ...'question_tok': ['evening', ...]}",
                 'table_data': "{'header_tok': [['country'], ...], ...'types': ['text', ...]}"},
                {'sql_data': "{'question': 'Evening ...', ...'question_tok': ['evening', ...]}",
                 'table_data': "{'header_tok': [['country'], ...], ...'types': ['text', ...]}"},
                {'sql_data': "{'question': 'Evening ...', ...'question_tok': ['evening', ...]}",
                 'table_data': "{'header_tok': [['country'], ...], ...'types': ['text', ...]}"}
                ]
        :return: 返回预测结果，是list形式，详情请参考样例sql_pred的输出值
        sql_pred:  [{'agg': 0, 'sel': 2, 'conds': [[0, 0, ''], [5, 0, ''], [1, 0, ''], [6, 0, '']]},
                    {'agg': 0, 'sel': 3, 'conds': [[3, 0, ''], [1, 0, ''], [4, 0, ''], [2, 0, '']]},
                    {'agg': 0, 'sel': 1, 'conds': [[3, 0, ''], [0, 0, ''], [2, 0, ''], [4, 0, '']]},
                    {'agg': 0, 'sel': 0, 'conds': [[3, 0, ''], [4, 0, ''], [0, 0, ''], [2, 0, '']]}]
        '''
        self.load_model()
        agg_m, sel_m, cond_m = best_model_name(MODEL_PATH)
        self.model.agg_pred.load_state_dict(torch.load(agg_m, map_location='cuda:0' if GPU else 'cpu'))
        self.model.sel_pred.load_state_dict(torch.load(sel_m, map_location='cuda:0' if GPU else 'cpu'))
        # self.model.cond_pred.load_state_dict(torch.load(cond_m, map_location='cuda:0' if GPU else 'cpu'))
        self.model.cond_pred.load_state_dict(torch.load(cond_m, map_location='cpu'))

        sql_list, table_list = list(), list()
        for data in datas:
            sql_data = eval(data['sql_data'])
            table_data = eval(data['table_data'])
            sql_list.append(sql_data)
            table_list.append(table_data)
        torch.cuda.empty_cache()
        sql_pred = predict_test(self.model, sql_list, table_list, TEST_ENTRY)

        return sql_pred


if __name__ == '__main__':
    Prediction().predict(datas=[
        {
            'sql_data': "{'question': 'For what year is the SAR no. 874?', 'query_tok': ['SELECT', 'year', 'WHERE', 'sar', 'no', '.', 'EQL', '874'], 'query_tok_space': [' ', ' ', ' ', ' ', '', ' ', ' ', ''], 'table_id': '1-29753553-1', 'question_tok_space': [' ', ' ', ' ', ' ', ' ', ' ', ' ', '', ''], 'phase': 1, 'query': 'SELECT year WHERE sar no. EQL 874', 'question_tok': ['for', 'what', 'year', 'is', 'the', 'sar', 'no.', '874', '?']}",
            'table_data': "{'header_tok': [['sar', 'no', '.'], ['builder'], ['year'], ['works', 'no', '.'], ['firebox'], ['driver', 'diameter']], 'rows': [[843, 'Baldwin', 1929, 60820, 'Narrow', '63""/1600mm'], [844, 'Baldwin', 1929, 60821, 'Narrow', '60""/1520mm'], [845, 'Baldwin', 1929, 60822, 'Narrow', '60""/1520mm'], [846, 'Baldwin', 1929, 60823, 'Narrow', '63""/1600mm'], [847, 'Baldwin', 1929, 60824, 'Narrow', '60""/1520mm'], [848, 'Baldwin', 1929, 60825, 'Narrow', '63""/1600mm'], [849, 'Baldwin', 1929, 60826, 'Narrow', '60""/1520mm'], [850, 'Baldwin', 1929, 60827, 'Narrow', '60""/1520mm'], [868, 'Hohenzollern', 1928, 4653, 'Narrow', '63""/1600mm'], [869, 'Hohenzollern', 1928, 4654, 'Narrow', '63""/1600mm'], [870, 'Hohenzollern', 1928, 4655, 'Narrow', '60""/1520mm'], [871, 'Hohenzollern', 1928, 4656, 'Narrow', '60""/1520mm'], [872, 'Hohenzollern', 1928, 4657, 'Narrow', '60""/1520mm'], [873, 'Hohenzollern', 1928, 4658, 'Narrow', '63""/1600mm'], [874, 'Henschel', 1930, 21749, 'Wide', '63""/1600mm'], [875, 'Henschel', 1930, 21750, 'Wide', '63""/1600mm'], [876, 'Henschel', 1930, 21751, 'Wide', '60""/1520mm'], [877, 'Henschel', 1930, 21752, 'Wide', '60""/1520mm'], [878, 'Henschel', 1930, 21753, 'Wide', '63""/1600mm']], 'page_title': 'South African Class 16DA 4-6-2', 'name': 'table_29753553_1', 'caption': 'Class 16DA 4-6-2 Builders, Works Numbers & Variations', 'section_title': 'Modification', 'header': ['SAR No.', 'Builder', 'Year', 'Works No.', 'Firebox', 'Driver Diameter'], 'header_tok_space': [[' ', '', ''], [''], [''], [' ', '', ''], [''], [' ', '']], 'id': '1-29753553-1', 'types': ['real', 'text', 'real', 'real', 'text', 'text']}"
        },
        {
            'sql_data': "{'question': 'who is the reader of the audiobook authored by cole, stephen stephen cole and released on 2008-11-13 13 november 2008', 'query_tok': ['SELECT', 'reader', 'WHERE', 'author', 'EQL', 'cole', ',', 'stephen', 'stephen', 'cole', 'AND', 'release', 'date', 'EQL', '2008-11-13', '13', 'november', '2008'], 'query_tok_space': [' ', ' ', ' ', ' ', ' ', '', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ''], 'table_id': '1-20174050-24', 'question_tok_space': [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ''], 'phase': 1, 'query': 'SELECT reader WHERE author EQL cole, stephen stephen cole AND release date EQL 2008-11-13 13 november 2008', 'question_tok': ['who', 'is', 'the', 'reader', 'of', 'the', 'audiobook', 'authored', 'by', 'cole', ',', 'stephen', 'stephen', 'cole', 'and', 'released', 'on', '2008-11-13', '13', 'november', '2008']}",
            'table_data': "{'header_tok': [['title'], ['author'], ['reader'], ['format'], ['company'], ['release', 'date'], ['notes']], 'rows': [['The Glittering Storm', 'Cole, Stephen Stephen Cole', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2007-11-05 5 November 2007', 'An original audiobook, not published in book form.'], ['The Thirteenth Stone', 'Richards, Justin Justin Richards', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2007-11-05 5 November 2007', 'An original audiobook, not published in book form.'], ['The Time Capsule', 'Anghelides, Peter Peter Anghelides', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2008-11-13 13 November 2008', 'An original audiobook, not published in book form.'], ['The Ghost House', 'Cole, Stephen Stephen Cole', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2008-11-13 13 November 2008', 'An original audiobook, not published in book form.'], ['The White Wolf', 'Russell, Gary Gary Russell', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2009-09-03 3 September 2009', 'An original audiobook, not published in book form.'], ['The Shadow People', 'Handcock, Scott Scott Handcock', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2009-09-03 3 September 2009', 'An original audiobook, not published in book form.'], ['Deadly Download', 'Arnopp, Jason Jason Arnopp', 'Sladen, Elisabeth Elisabeth Sladen', 'CD', 'BBC Audio', '2010-11-04 4 November 2010', 'An original audiobook, not published in book form.'], ['Children of Steel', 'Day, Martin Martin Day', 'Anthony, Daniel Daniel Anthony', 'CD', 'BBC Audio', '2011-10-06 6 October 2011', 'An original audiobook, not published in book form.']], 'page_title': 'List of Doctor Who audiobooks', 'name': 'table_20174050_24', 'caption': 'The Sarah Jane Adventures', 'section_title': 'The Sarah Jane Adventures', 'header': ['Title', 'Author', 'Reader', 'Format', 'Company', 'Release Date', 'Notes'], 'header_tok_space': [[''], [''], [''], [''], [''], [' ', ''], ['']], 'id': '1-20174050-24', 'types': ['text', 'text', 'text', 'text', 'text', 'text', 'text']}"}
    ])
