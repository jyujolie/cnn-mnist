# -*- coding: utf-8 -*-
"""
基于 PyTorch 的 CNN 手写数字识别（MNIST）

本文件是《深度学习》实验一「手写数字识别」的 PyTorch 实现。
模型结构沿用之前 TensorFlow 版本的思路（两层卷积 + 全连接），
用 PyTorch 重新实现，满足实验指导书「掌握 PyTorch 基本用法」的要求。

依赖：torch、torchvision、numpy、matplotlib
运行：conda activate dl_lab && python CNN手写数字识别_pytorch.py

说明：
  - 首次运行会自动下载 MNIST 数据集（约 12MB）；
  - 训练结束后会打印测试集准确率，并保存模型为 mnist_cnn_pytorch.pth；
  - 可视化图片会保存为 PNG 文件（predict_examples.png），方便写报告时引用。
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ==================== 全局配置 ====================
EPOCHS = 6                 # 训练轮数
BATCH_SIZE = 128           # 每批样本数
LEARNING_RATE = 0.001      # 学习率
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "mnist_cnn_pytorch.pth"

# 设置随机种子，保证结果可复现
torch.manual_seed(42)


# ==================== 数据加载与预处理 ====================
def load_data():
    """
    加载 MNIST 数据集并构造 DataLoader。

    transforms 作用：
      - ToTensor()：把 PIL 图像转成张量，并把像素从 0~255 缩放到 0~1；
      - Normalize((0.1307,), (0.3081,))：按 MNIST 的均值和标准差做标准化，
        帮助模型更快收敛。

    数据增强（只在训练集上做，测试集保持原样）：
      - RandomRotation(±10°)：随机旋转，让模型适应数字的倾斜；
      - RandomAffine(平移/缩放)：随机平移和缩放，模拟手写位置和大小的差异。
      通过「制造多种变体」扩大训练样本的多样性，抑制过拟合、提升泛化能力。
    """
    # 训练集：加上数据增强，让模型看到更多样的样本
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),          # 随机旋转 ±10°
        transforms.RandomAffine(
            degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机平移/缩放
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # 测试集：不做增强，保持原始数据，保证评估公平
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # download=True 表示数据集不存在时自动下载
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=test_transform)

    print(f"训练集样本数：{len(train_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    print(f"单张图片形状：{train_dataset[0][0].shape}  (通道, 高, 宽)")

    # shuffle=True：训练时打乱顺序，防止模型记住样本顺序
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, test_loader


# ==================== 定义 CNN 模型 ====================
class CNN(nn.Module):
    """
    卷积神经网络模型。

    网络结构（每层之后接 ReLU 激活）：
      输入 1×28×28
        -> Conv2d(1, 32, 3)   输出 32×26×26  提取局部笔画特征
        -> MaxPool2d(2)       输出 32×13×13  缩小特征图
        -> Conv2d(32, 64, 3)  输出 64×11×11  提取更复杂的特征
        -> MaxPool2d(2)       输出 64×5×5
        -> Flatten            拉平成 64*5*5 = 1600
        -> Linear(1600, 64)   全连接，整合特征
        -> Linear(64, 10)     输出 10 个类别的得分（logits）
    """

    def __init__(self):
        super(CNN, self).__init__()
        # 卷积层 + 池化层（特征提取部分）
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)      # 1 通道 -> 32 通道
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) # 2×2 最大池化
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)     # 32 通道 -> 64 通道

        # 全连接层（分类部分）
        # 经过两次池化后特征图尺寸为 5×5，共 64 个通道
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        """定义前向传播：数据如何从输入流到输出。"""
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv1(x)))
        # 卷积 -> ReLU -> 池化
        x = self.pool(F.relu(self.conv2(x)))

        # 展平：把 (batch, 64, 5, 5) 拉成 (batch, 64*5*5)
        x = x.view(x.size(0), -1)

        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.fc2(x)   # 注意：这里不接 softmax，
                          # 因为 CrossEntropyLoss 内部已经包含 softmax
        return x


# ==================== 训练函数 ====================
def train(model, train_loader, criterion, optimizer):
    """训练一个 epoch，返回本轮的平均损失和准确率。"""
    model.train()                      # 切换到训练模式（启用 Dropout/BatchNorm）
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        # 数据搬到 GPU
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        # 1. 前向传播：计算预测
        outputs = model(images)
        # 2. 计算损失
        loss = criterion(outputs, labels)

        # 3. 反向传播与参数更新（三步曲，必须按顺序）
        optimizer.zero_grad()   # 清空上一步的梯度
        loss.backward()         # 反向传播，自动计算梯度
        optimizer.step()        # 用梯度更新参数

        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)   # 取得分最大的类别作为预测
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / len(train_loader), correct / total


# ==================== 评估函数 ====================
def evaluate(model, test_loader, criterion):
    """在测试集上评估模型，返回平均损失和准确率。"""
    model.eval()                       # 切换到评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():              # 关闭梯度计算，节省显存、加快速度
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / len(test_loader), correct / total


# ==================== 可视化：预测正确与错误的例子 ====================
def show_prediction_examples(model, test_dataset, count=4):
    """
    从测试集中分别选取预测正确和预测错误的各 count 张图片，
    绘制成两排并保存为 PNG 文件（上排正确、下排错误）。
    """
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    with torch.no_grad():
        images, labels = next(iter(test_loader))
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)

    # 转回 CPU 并转成 numpy，便于 matplotlib 绘图
    images = images.cpu().numpy()
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    correct_idx = np.flatnonzero(predictions == labels)[:count]
    wrong_idx = np.flatnonzero(predictions != labels)[:count]

    fig, axes = plt.subplots(2, count, figsize=(12, 6))
    for row, indices in enumerate([correct_idx, wrong_idx]):
        for col, ax in enumerate(axes[row]):
            if col >= len(indices):
                ax.axis("off")
                continue
            i = indices[col]
            # 图片是 (1, 28, 28)，取第一个通道并转置成 (28, 28)
            ax.imshow(images[i][0], cmap="gray")
            ax.set_title(f"Pred: {predictions[i]} | True: {labels[i]}")
            ax.axis("off")

    fig.suptitle("Top: correct predictions | Bottom: incorrect predictions")
    plt.tight_layout()
    plt.savefig("predict_examples.png", dpi=150)
    print(f"已保存预测示例图：predict_examples.png "
          f"（正确 {len(correct_idx)} 张，错误 {len(wrong_idx)} 张）")


# ==================== 主流程 ====================
def main():
    print(f"使用设备：{DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU 型号：{torch.cuda.get_device_name(0)}")

    # 1. 加载数据
    train_loader, test_loader = load_data()

    # 2. 构建模型
    model = CNN().to(DEVICE)
    print(model)
    print(f"模型可训练参数总数："
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. 定义损失函数与优化器
    criterion = nn.CrossEntropyLoss()          # 交叉熵损失（内含 softmax）
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. 训练与评估
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}  "
              f"训练损失 {train_loss:.4f}  训练准确率 {train_acc:.2%}  |  "
              f"测试损失 {test_loss:.4f}  测试准确率 {test_acc:.2%}")

    # 5. 最终测试集评估
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    print(f"\n===== 最终结果 =====")
    print(f"测试集损失：{test_loss:.4f}")
    print(f"测试集准确率：{test_acc:.2%}")
    assert test_acc >= 0.98, "准确率未达到 98%，请调整超参数或模型结构"

    # 6. 可视化预测示例
    test_dataset = datasets.MNIST(
        root="./data", train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]))
    show_prediction_examples(model, test_dataset)

    # 7. 保存模型
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"模型已保存到：{MODEL_PATH}")


if __name__ == "__main__":
    main()
