# CNN 手写数字识别（PyTorch）

使用卷积神经网络（CNN）识别 MNIST 数据集中的手写数字 0～9。基于 PyTorch 实现，用于《深度学习》实验一「手写数字识别」。

## 数据集

MNIST 包含 60,000 张训练图片和 10,000 张测试图片。每张图片为 28×28 像素的灰度图，对应一个 0～9 的数字标签。数据通过 `torchvision.datasets.MNIST` 自动下载，图片经 `ToTensor` 缩放到 0～1，并按 MNIST 的均值和标准差（0.1307, 0.3081）做标准化。

训练集额外加入**数据增强**：随机旋转（±10°）和随机仿射变换（随机平移/缩放），用于提升模型的泛化能力、抑制过拟合。测试集保持原始数据，不做增强，保证评估公平。

## CNN 结构

| 顺序 | 层 | 作用 |
| --- | --- | --- |
| 1 | Conv2d(1→32, 3×3) + ReLU | 提取局部笔画特征 |
| 2 | MaxPool2d(2×2) | 缩小特征图 |
| 3 | Conv2d(32→64, 3×3) + ReLU | 提取更复杂的图像特征 |
| 4 | MaxPool2d(2×2) | 再次缩小特征图 |
| 5 | Flatten + Linear(1600→64) + ReLU | 整合特征 |
| 6 | Linear(64→10) | 输出 10 个数字类别的得分 |

模型共有 **121,930** 个可训练参数。使用 `CrossEntropyLoss` 损失函数和 `Adam` 优化器（lr=0.001），支持 GPU（`cuda`）加速。

## 运行方式

### 1. 创建并激活环境

```bash
conda create -n dl_lab python=3.10 -y
conda activate dl_lab
```

### 2. 安装依赖（GPU 版，CUDA 12.8）

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install matplotlib numpy
```

> 无 NVIDIA GPU 时，可改为 `pip install torch torchvision` 安装 CPU 版，代码会自动回退到 CPU 运行。

### 3. 运行

```bash
python CNN手写数字识别_pytorch.py
```

首次运行会自动下载 MNIST 数据集（约 12MB）。训练结束后会打印测试集准确率，保存模型 `mnist_cnn_pytorch.pth`，并生成预测示例图 `predict_examples.png`。

## 训练与结果

- 训练轮数：6（`epochs=6`），批量大小 128；
- 加入数据增强后，测试集准确率：**99.16%**（要求 ≥ 98%）。

训练过程中每个 epoch 都会输出训练/测试的损失和准确率，便于观察收敛情况。数据增强带来的「训练准确率降低、测试准确率提升」现象，正是抑制过拟合、提升泛化能力的体现。

## 预测示例

`predict_examples.png` 展示了模型从测试集中选取的预测正确（上排）与预测错误（下排）的样本。每张图上方 `Pred` 为模型预测，`True` 为正确答案。错误的样本多为「长得像」的数字混淆（如 9↔5、6↔0、2↔7 等），这是 MNIST 中典型的难例。

## 可调参数

代码顶部提供以下超参数，可按需修改：

- `EPOCHS`：训练轮数；
- `BATCH_SIZE`：每批样本数；
- `LEARNING_RATE`：学习率。
