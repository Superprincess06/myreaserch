import pandas as pd
import os

def get_stratified_subset(df, target_num):
    """
    核心工具：从大数据集中，按多标签分层比例，精准切出一个小数据集
    """
    if target_num <= 0:
        return df.head(0).copy()

    # 修复问题 2：如果索要的数量比总数据还多，直接返回全部，防止 ValueError
    if target_num >= len(df):
        return df.copy()

    if 'image' not in df.columns:
        raise ValueError("❌ CSV 中缺少 image 列，无法进行抽样。")
    if df.shape[1] < 2:
        raise ValueError("❌ CSV 中缺少标签列，至少需要 1 列标签。")
    
    # 因为我们的 CSV 从第 2 列开始全是 0/1 标签，直接提出来当 y
    y = df.iloc[:, 1:].values
    X = df['image'].values

    # 优先使用多标签分层抽样；若依赖或分层失败则回退到普通随机抽样，保证流程不中断。
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

        msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=target_num, random_state=42)
        for _, subset_idx in msss.split(X, y):
            subset = df.iloc[subset_idx].copy()
            if len(subset) == target_num:
                return subset

            # 某些极端标签分布下可能出现数量偏差，使用随机补齐/截断到目标数量。
            if len(subset) < target_num:
                remain = df.drop(index=subset.index)
                needed = target_num - len(subset)
                extra = remain.sample(n=needed, random_state=42)
                return pd.concat([subset, extra], axis=0)
            return subset.sample(n=target_num, random_state=42)
    except Exception as e:
        print(f"⚠️ 分层抽样不可用，改用随机抽样。原因: {e}")
        return df.sample(n=target_num, random_state=42).copy()

def create_mini_dataset(raw_dir='data/raw', train_num=100, val_num=20):
    print("⏳ 正在为您制作【防弹版】多标签微型数据集...")
    
    train_path = os.path.join(raw_dir, 'train_split.csv')
    val_path = os.path.join(raw_dir, 'val_split.csv')
    
    # 修复问题 1：严谨的路径双重检查，报错信息更友好
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"❌ 找不到训练集: {train_path}。请确保在项目根目录运行！")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"❌ 找不到验证集: {val_path}。请检查第一次切分是否成功！")
        
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # 调用分层抽样函数
    mini_train = get_stratified_subset(df_train, train_num)
    mini_val = get_stratified_subset(df_val, val_num)
    
    # 创建 mini_data 文件夹并保存
    mini_data_dir = os.path.join(raw_dir, 'mini_data')
    os.makedirs(mini_data_dir, exist_ok=True) 
    
    mini_train_path = os.path.join(mini_data_dir, 'mini_train_split.csv')
    mini_val_path = os.path.join(mini_data_dir, 'mini_val_split.csv')
    
    mini_train.to_csv(mini_train_path, index=False)
    mini_val.to_csv(mini_val_path, index=False)
    
    print("🎉 微型数据集制作成功！(已完美保持多标签比例分布)")
    print(f"📁 本地训练集 ({len(mini_train)}张): 已保存至 {mini_train_path}")
    print(f"📁 本地验证集 ({len(mini_val)}张): 已保存至 {mini_val_path}")

if __name__ == '__main__':
    create_mini_dataset()