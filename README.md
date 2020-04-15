# 2020-FlyAI-Yale-Text-To-SQL
2020 FlyAI 耶鲁文本转SQL


# aggregator:agg | selection:sel | where:conds

## AI研习社

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
    