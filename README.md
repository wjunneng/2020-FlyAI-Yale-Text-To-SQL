# 2020-FlyAI-Yale-Text-To-SQL
2020 FlyAI 耶鲁文本转SQL


# aggregator:agg | selection:sel | where:conds

# AI研习社

    原始：
    # 忽略db_id不匹配的样本 if sample['db_id'] != db_id:
    
    # 训练集 忽略了: 1608 条样本
    # 验证集 忽略了: 388 条样本
    # 测试集 忽略了: 497 条样本
    score: 74.3682
    
    
    # 忽略db_id不存在的样本 if sample['db_id'] not in table_data:
    # 训练集 忽略了: 1590 条样本
    # 验证集 忽略了: 383 条样本
    # 测试集 忽略了: 491 条样本
    score: 74.5229
    
    # 将sample['db_id'] 替换为 db_id
    # 训练集 忽略了: 1323 条样本
    # 验证集 忽略了: 313 条样本
    # 测试集 忽略了: 409 条样本
    score: 78.6488
    
    # 使用Spider辅助数据
    # 训练集 忽略了: 319 条样本
    # 验证集 忽略了: 81 条样本
    # 测试集 忽略了: 94 条样本
    score: 94.688
    
    
## 数据说明

    {
        "column_names": [
            [
                -1,
                "*"
            ],
            ...
            [
                1,
                "home town"
            ]
        ],
        "column_names_original": [
            [
                -1,
                "*"
            ],
            ...
            [
                1,
                "home town"
            ]
        ],
        "column_types": [
            "text",
            "number",
            ...
            "number",
            "text"
        ],
        "db_id": "perpetrator",
        "foreign_keys": [
            [
                2,
                9
            ]
        ],
        "primary_keys": [
            1,
            9
        ],
        "table_names": [
            "perpetrator",
            "people"
        ],
        "table_names_original": [
            "perpetrator",
            "people"
        ]
    },
    
    

# Flyai

    sql_data:
    
    1. question            'For what year is the SAR no. 874?'
    2. question_tok        ['for', 'what', 'year', 'is', 'the', 'sar', 'no.', '874', '?']
    3. question_tok_space  [' ', ' ', ' ', ' ', ' ', ' ', ' ', '', '']
    4. query               'SELECT year WHERE sar no. EQL 874'
    6. query_tok           ['SELECT', 'year', 'WHERE', 'sar', 'no', '.', 'EQL', '874']
    7. query_tok_space     [' ', ' ', ' ', ' ', '', ' ', ' ', '']
    8. table_id            '1-29753553-1'
    9. phase               1
    10.sql                 {'agg': 0, 'sel': 2, 'conds': [[0, 0, 874]]}
    
    tables:
    1.header_tok           [['sar', 'no', '.'], ['builder'], ['year'], ['works', 'no', '.'], ['firebox'], ['driver', 'diameter']]
    2.rows                 [[843, 'Baldwin', 1929, 60820, 'Narrow', '63"/1600mm'], [844, 'Baldwin', 1929, 60821, 'Narrow', '60"/1520mm'], [845, 'Baldwin', 1929, 60822, 'Narrow', '60"/1520mm'], [846, 'Baldwin', 1929, 60823, 'Narrow', '63"/1600mm'], [847, 'Baldwin', 1929, 60824, 'Narrow', '60"/1520mm'], [848, 'Baldwin', 1929, 60825, 'Narrow', '63"/1600mm'], [849, 'Baldwin', 1929, 60826, 'Narrow', '60"/1520mm'], [850, 'Baldwin', 1929, 60827, 'Narrow', '60"/1520mm'], [868, 'Hohenzollern', 1928, 4653, 'Narrow', '63"/1600mm'], [869, 'Hohenzollern', 1928, 4654, 'Narrow', '63"/1600mm'], [870, 'Hohenzollern', 1928, 4655, 'Narrow', '60"/1520mm'], [871, 'Hohenzollern', 1928, 4656, 'Narrow', '60"/1520mm'], [872, 'Hohenzollern', 1928, 4657, 'Narrow', '60"/1520mm'], [873, 'Hohenzollern', 1928, 4658, 'Narrow', '63"/1600mm'], [874, 'Henschel', 1930, 21749, 'Wide', '63"/1600mm'], [875, 'Henschel', 1930, 21750, 'Wide', '63"/1600mm'], [876, 'Henschel', 1930, 21751, 'Wide', '60"/1520mm'], [877, 'Henschel', 1930, 21752, 'Wide', '60"/1520mm'], [878, 'Henschel', 1930, 21753, 'Wide', '63"/1600mm']]
    3.page_title           'South African Class 16DA 4-6-2'
    4.name                 'table_29753553_1'
    5.caption              'Class 16DA 4-6-2 Builders, Works Numbers & Variations'
    6.section_title        'Modification'
    7.header               ['SAR No.', 'Builder', 'Year', 'Works No.', 'Firebox', 'Driver Diameter']
    8.header_tok_space     [[' ', '', ''], [''], [''], [' ', '', ''], [''], [' ', '']]
    9.id                   '1-29753553-1'
    10.types               ['real', 'text', 'real', 'real', 'text', 'text']
    
## 说明：
    
    {
       "phase":1,
       "question":"who is the manufacturer for the order year 1998?",
       "sql":{
          "conds":[
             [
                0,
                0,
                "1998"
             ]
          ],
          "sel":1,
          "agg":0
       },
       "table_id":"1-10007452-3"
    }
    
    question 是自然语言问题
    table_id 是与这个这个问题相关的表格编号
    sql      字段是标签数据。这个数据集进一步把 SQL 语句结构化（简化），分成了conds，sel，agg三个部分。
    sel      是查询目标列，其值是表格中对应列的序号
    agg      的值是聚合操作的编号，可能出现的聚合操作有['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']共 6 种。
    conds    是筛选条件，可以有多个。每个条件用一个三元组(column_index, operator_index, condition)表示，可能的operator_index共有['=', '>', '<', 'OP']四种，condition是操作的目标值，这是不能用分类解决的目标。
    