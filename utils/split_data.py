import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def split_and_format_fgvc8(csv_path='data/raw/train.csv', output_dir='data/raw'):
    print("⏳ 正在读取原始 CSV 并展开多标签...")
    df = pd.read_csv(csv_path)
    
    # 1. 解析标签为列表
    df['labels_list'] = df['labels'].apply(lambda x: x.split(' '))
    
    # 2. 转换为 0/1 矩阵，并获取病害类别名称
    mlb = MultiLabelBinarizer()
    labels_matrix = mlb.fit_transform(df['labels_list'])
    classes = mlb.classes_
    
    print(f"✅ 成功提取 6 大类病害: {list(classes)}")
    
    # 🎯 核心修正：把 0/1 矩阵变成 DataFrame，并和图片名拼接到一起！
    df_labels = pd.DataFrame(labels_matrix, columns=classes)
    # 最终的表只保留 image 列和 6个病害列 (彻底丢掉原始的字符串 labels)
    df_final = pd.concat([df[['image']], df_labels], axis=1)
    
    X = df_final['image'].values
    y = labels_matrix
    
    # ==========================================
    # 第一次切分 (10% Test)
    # ==========================================
    msss1 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_val_idx, test_idx in msss1.split(X, y):
        X_train_val, y_train_val = X[train_val_idx], y[train_val_idx]
        df_train_val = df_final.iloc[train_val_idx].copy()
        df_test = df_final.iloc[test_idx].copy()

    # ==========================================
    # 第二次切分 (70% Train, 20% Val)
    # ==========================================
    msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.2222, random_state=42)
    for train_idx, val_idx in msss2.split(X_train_val, y_train_val):
        df_train = df_train_val.iloc[train_idx].copy()
        df_val = df_train_val.iloc[val_idx].copy()

    # ==========================================
    # 🎯 核心修正：安全地保存文件 (使用 os.path.join)
    # ==========================================
    os.makedirs(output_dir, exist_ok=True) # 确保目录存在
    
    train_path = os.path.join(output_dir, 'train_split.csv')
    val_path = os.path.join(output_dir, 'val_split.csv')
    test_path = os.path.join(output_dir, 'test_split.csv')
    
    df_train.to_csv(train_path, index=False)
    df_val.to_csv(val_path, index=False)
    df_test.to_csv(test_path, index=False)
    
    print("\n🎉 切分与 0/1 格式化圆满完成！")
    print(f"📁 训练集: {len(df_train)} 张 -> 保存至 {train_path}")
    print(f"📁 验证集: {len(df_val)} 张 -> 保存至 {val_path}")
    print(f"📁 测试集: {len(df_test)} 张 -> 保存至 {test_path}")
    print("CSV 预览 (前2行):")
    print(df_train.head(2))

if __name__ == '__main__':
    # 注意：请在项目根目录 (FGVC8_Project) 下运行此脚本：
    # python utils/split_data.py
    split_and_format_fgvc8()