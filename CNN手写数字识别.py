"""基于 CNN 的 MNIST 手写数字识别（Python 脚本版）

本文件是 CNN手写数字识别.ipynb 的等价 Python 脚本版本，模型结构、
训练设置与笔记本完全一致，适合在本地命令行或服务器环境中直接运行，
不依赖 Jupyter / Colab。

依赖：tensorflow、numpy、matplotlib
运行：python CNN手写数字识别.py

说明：脚本中的图片通过 plt.show() 弹出显示；如果在无图形界面的环境中
运行，可把 plt.show() 换成 plt.savefig("结果.png") 保存为文件。
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 训练超参数
EPOCHS = 5
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.1
MODEL_PATH = "mnist_cnn.keras"


def load_data():
    """加载 MNIST 数据集并打印基本信息。"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    print("训练图片：", x_train.shape)
    print("测试图片：", x_test.shape)
    print("第一张图片的正确答案：", y_train[0])
    return (x_train, y_train), (x_test, y_test)


def show_sample(x_train, y_train, index=0):
    """显示训练集中的一张图片及其正确答案。"""
    plt.imshow(x_train[index], cmap="gray")
    plt.title(f"正确数字：{y_train[index]}")
    plt.axis("off")
    plt.show()


def preprocess(x_train, x_test):
    """把像素缩放到 0~1，并增加一个灰度通道维度。"""
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    print("预处理后的训练数据：", x_train.shape)
    return x_train, x_test


def build_model():
    """构建并编译 CNN 模型（共 121,930 个可训练参数）。"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),   # 提取局部笔画特征
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),   # 提取更复杂的特征
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(10, activation="softmax"),         # 输出 10 个类别的概率
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train(model, x_train, y_train):
    """训练模型，从训练数据中划出 10% 作为验证集。"""
    return model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
    )


def evaluate(model, x_test, y_test):
    """在测试集上评估模型，输出损失与准确率。"""
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失：{test_loss:.4f}")
    print(f"测试集准确率：{test_accuracy:.2%}")
    return test_loss, test_accuracy


def show_single_prediction(model, x_test, y_test, index=0):
    """显示测试集中某一张图片的预测结果与正确答案。"""
    predictions = model.predict(x_test[: index + 1], verbose=0)
    predicted_digit = predictions[index].argmax()

    plt.imshow(x_test[index, :, :, 0], cmap="gray")
    plt.title(f"预测：{predicted_digit}；正确答案：{y_test[index]}")
    plt.axis("off")
    plt.show()


def show_prediction_examples(model, x_test, y_test, count=4):
    """上排展示预测正确的样本，下排展示预测错误的样本。"""
    all_probabilities = model.predict(x_test, verbose=0)
    all_predictions = np.argmax(all_probabilities, axis=1)
    correct_indices = np.flatnonzero(all_predictions == y_test)[:count]
    wrong_indices = np.flatnonzero(all_predictions != y_test)[:count]

    fig, axes = plt.subplots(2, count, figsize=(12, 6))
    for row, indices in enumerate([correct_indices, wrong_indices]):
        for column, ax in enumerate(axes[row]):
            if column >= len(indices):
                ax.axis("off")
                continue
            i = indices[column]
            ax.imshow(x_test[i, :, :, 0], cmap="gray")
            ax.set_title(f"Pred: {all_predictions[i]} | True: {y_test[i]}")
            ax.axis("off")

    fig.suptitle("Top: correct predictions | Bottom: incorrect predictions")
    plt.tight_layout()
    plt.show()
    print(f"正确示例 {len(correct_indices)} 张；错误示例 {len(wrong_indices)} 张")


def main():
    (x_train, y_train), (x_test, y_test) = load_data()
    show_sample(x_train, y_train)
    x_train, x_test = preprocess(x_train, x_test)

    model = build_model()
    train(model, x_train, y_train)

    evaluate(model, x_test, y_test)
    show_single_prediction(model, x_test, y_test)
    show_prediction_examples(model, x_test, y_test)

    model.save(MODEL_PATH)
    print(f"模型已保存到：{MODEL_PATH}")


if __name__ == "__main__":
    main()
