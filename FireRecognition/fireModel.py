import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import gc
import pickle

image_size = 160
# 加载标签文件
train_labels = pd.read_csv('dataset/train.csv')

# 将 id 和 label 列转换为字符串类型，确保数据格式一致
train_labels['id'] = train_labels['id'].astype(str)
train_labels['label'] = train_labels['label'].astype(str)

# 将数据集分割为训练集和验证集
train_df, validation_df = train_test_split(train_labels, test_size=0.15, stratify=train_labels['label'], random_state=33)

# 定义函数：加载图像并进行预处理
def load_images(df, img_dir):
    images = []
    labels = []
    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, row['id'] + '.png')
        if os.path.exists(img_path):
            img = load_img(img_path, target_size=(image_size, image_size))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            labels.append(row['label'])
        else:
            print(f"File not found: {img_path}")
    return np.array(images), np.array(labels)

# 加载训练集和验证集的图像数据
x_train, y_train = load_images(train_df, 'dataset/train')
x_val, y_val = load_images(validation_df, 'dataset/train')
print("数据加载完成")

# 删除不再需要的变量以释放内存
del train_labels, train_df, validation_df
gc.collect()
print("内存清理完成")

# 将标签转换为独热编码格式
# 例如，如果标签是 0 和 1，则转换为 [1, 0] 和 [0, 1]
y_train = tf.keras.utils.to_categorical(y_train)
y_val = tf.keras.utils.to_categorical(y_val)

# 构建卷积神经网络（CNN）模型
model = keras.Sequential([
    # 第一个卷积块
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(image_size, image_size, 3),
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),

    # 第二个卷积块
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),

    # 第三个卷积块
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu',
                        kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.4),

    # 展平层
    keras.layers.Flatten(),
    # 全连接层
    keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.5),
    # 输出层
    keras.layers.Dense(2, activation='softmax')
])

# 编译模型
# 使用 Adam 优化器，交叉熵损失函数，并监控准确率
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 定义模型检查点回调
# 在训练过程中，如果验证集准确率提高，则保存模型
checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5',
                                             monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
# 创建ReduceLROnPlateau回调，用于在验证损失不再降低时减少学习率
reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.01, min_lr=1e-5, verbose=1)

# 训练模型
# batch_size=64：每次训练使用 64 个样本
# validation_data=(x_val, y_val)：使用验证集评估模型性能
# callbacks=[checkpoint, reduce]：在训练过程中使用检查点回调
hist = model.fit(
    x_train, y_train, validation_data=(x_val, y_val),
    epochs=50, batch_size=64,
    callbacks=[checkpoint, reduce],
    use_multiprocessing=True, workers=16
)

# 保存训练历史数据到 history.pkl 文件
with open('history.pkl', 'wb') as f:
    pickle.dump(hist.history, f)

# 保存最终训练好的模型
model.save('final_model.h5')
