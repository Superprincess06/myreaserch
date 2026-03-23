import os
import cv2
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class FGVC8Dataset(Dataset):
    """
    FGVC8 苹果叶片多标签数据集的读取器
    只负责两件事：1. 把硬盘里的图片读进内存；2. 把 CSV 里的 0/1 提出来变成张量。
    """
    def __init__(self, csv_file, img_dir, transform=None):
        """
        初始化函数 (场务拿着名单在仓库门口就位)
        :param csv_file: 包含 0/1 多标签的切分文件路径 (如 data/raw/train_split.csv)
        :param img_dir: 图片所在的真实文件夹路径 (如 data/raw/train_images)
        :param transform: 从 augment.py 传过来的化妆师 (图像增强策略)
        """
        # 读取我们之前切分好的包含 0和1 的完美 CSV
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        
        # 自动推断标签列的名字 (跳过第一列的 'image'，剩下的全都是病害名字)
        self.label_cols = self.df.columns[1:].tolist()

    def __len__(self):
        """告诉 PyTorch 我们总共有多少张图"""
        return len(self.df)

    def __getitem__(self, idx):
        """
        最核心的提货机制：当模型喊“给我第 idx 张图”时，执行以下操作
        """
        # 1. 对暗号：拿图片名字，拼接出电脑上的绝对/相对路径
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)
        
        # 2. 提货：用 OpenCV 读取图片
        image = cv2.imread(img_path)
        if image is None:
            # 极其重要的排雷机制：如果图片漏下了或者名字不对，立刻报错，别让模型干等
            raise FileNotFoundError(f"❌ 严重错误: 找不到图片 {img_path}！请检查文件夹！")
        
        # ⚠️ 避坑：cv2 默认读出来是 BGR 色彩顺序，深度学习通常用 RGB，必须翻转一下
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 3. 拿体检报告：提取这行数据从第 2 列开始的 0/1 标签，变成 float32 格式
        # 注意：多标签分类的 BCE Loss 极其严格，标签必须是浮点数 (float32) 而不能是整数 (int)
        labels = self.df.iloc[idx, 1:].values.astype(np.float32)
        labels_tensor = torch.tensor(labels)
        
        # 4. 去化妆间：如果外面派了化妆师 (transform) 过来，就对图片进行处理
        # 先把 numpy (H,W,C) 的 RGB 转换成 PIL Image（torchvision.transforms 需要 PIL Image 作为输入）
        image_pil = Image.fromarray(image.astype(np.uint8))
        
        if self.transform:
            # torchvision 的标准调用格式 (不需要关键字参数)
            image_tensor = self.transform(image_pil)
        else:
            # 如果没化妆师，自己粗糙地处理一下：转成张量，并把 (高度, 宽度, 通道) 变成 PyTorch 喜欢的 (通道, 高度, 宽度)
            image_tensor = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            
        # 5. 装盘上菜：返回处理好的图片张量和对应的标签张量
        return image_tensor, labels_tensor

# ==========================================
# 🚀 本地排雷测试区 (Sanity Check)
# ==========================================
if __name__ == '__main__':
    # 假设我们在项目根目录下运行 (python utils/dataset.py)
    # 请确保以下路径在您的电脑上是真实存在的！
    TEST_CSV = 'data/raw/train_split.csv'
    TEST_IMG_DIR = 'data/raw/train_images' 
    
    print("⏳ 正在测试 dataset.py 是否能正常工作...")
    
    try:
        # 实例化 Dataset (先不加 transform 试试底盘稳不稳)
        my_dataset = FGVC8Dataset(csv_file=TEST_CSV, img_dir=TEST_IMG_DIR, transform=None)
        
        print(f"✅ 成功加载 CSV！总计包含 {len(my_dataset)} 张图片。")
        print(f"🏷️ 侦测到 {len(my_dataset.label_cols)} 种病害标签: {my_dataset.label_cols}")
        
        # 尝试抽出第 0 张图
        img, lbl = my_dataset[0]
        
        print("\n🍽️ 成功抽出第 1 份数据！")
        print(f"🖼️ 图片 Tensor 形状: {img.shape} (预期为 [3, 高度, 宽度])")
        print(f"🎯 标签 Tensor 形状: {lbl.shape} (预期为 [6])")
        print(f"📊 标签数值 (前五个): {lbl}")
        print("🎉 完美通关！您的场务专家已经就位！")
        
    except Exception as e:
        print(f"\n❌ 测试失败，排雷报错信息如下: \n{e}")