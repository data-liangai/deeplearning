import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
import pickle

# 设置随机种子
np.random.seed(42)

# 设置图像大小
image_size = 160

# 加载训练好的模型
model = load_model('best_model.h5')
model.summary()

# 加载训练历史数据
with open('history.pkl', 'rb') as f:
    history = pickle.load(f)

# 加载测试集标签文件
# 假设 test.csv 文件只有一列 'id'，没有 'label' 列
test_labels = pd.read_csv('dataset/test.csv')
test_labels['id'] = test_labels['id'].astype(str)  # 确保 id 为字符串

# 定义函数：加载测试集图像并进行预处理
def load_images(df, img_dir):
    """
    加载测试集图像并将其转换为模型可用的格式。
    参数:
        df (DataFrame): 包含图像 id 的 DataFrame。
        img_dir (str): 图像文件所在的目录路径。
    返回:
        images (np.array): 图像数据数组，形状为 (num_samples, height, width, channels)。
        ids (np.array): 对应的图像 id 数组。
    """
    images = []
    ids = []
    for index, row in df.iterrows():
        # 构建图像文件的完整路径
        img_path = os.path.join(img_dir, row['id'] + '.png')
        # 检查图像文件是否存在
        if os.path.exists(img_path):
            # 加载图像并调整大小为 (image_size, image_size)
            img = load_img(img_path, target_size=(image_size, image_size))
            # 将图像转换为数组并归一化到 [0, 1] 范围
            img_array = img_to_array(img) / 255.0
            # 将图像和 id 添加到列表中
            images.append(img_array)
            ids.append(row['id'])
        else:
            print(f"File not found: {img_path}")
    # 将列表转换为 NumPy 数组并返回
    return np.array(images), np.array(ids)

# 加载测试集图像数据
x_test, test_ids = load_images(test_labels, 'dataset/test')

# 获取模型预测结果
predictions = model.predict(x_test)

# 将完整预测结果保存到 submission.csv
results = np.argmax(predictions, axis=1)  # 获取预测类别（0 或 1）
submissions = pd.DataFrame({'id': test_ids, 'label': results})  # 创建包含 id 和 label 的 DataFrame
submissions.to_csv('submission.csv', index=False)  # 保存为 CSV 文件

print("预测结果已保存到 submission.csv")

# 保存训练过程中模型的accuracy和loss变化
acc = history['accuracy']
val_acc = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.plot(epochs, loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'y', label='Validation loss')
plt.title('Training and validation accuracy, Training and validation loss')
plt.legend() # 绘制图例，默认在右上角
# 保存训练过程图像
plt.savefig('training.png')

# 随机抽取64张图片
num_samples = min(64, len(x_test))  # 确保不超过测试集图像数量
random_indices = np.random.choice(len(x_test), num_samples, replace=False)
x_test_random = x_test[random_indices]
predictions_random = predictions[random_indices]

# 显示 64 张测试集图像的预测结果
plt.figure(figsize=(20, 20))
for i in range(num_samples):
    plt.subplot(8, 8, i + 1)
    plt.imshow(x_test_random[i])
    # 根据预测结果生成标签
    pred_label = 'nofire' if np.argmax(predictions_random[i]) == 0 else 'fire'
    # 显示图像和预测标签
    plt.title(f'Pred: {pred_label}')
    plt.axis('off')

# 自适应调整子图布局
plt.tight_layout()
# 保存预测结果图像
plt.savefig('predict.png')
