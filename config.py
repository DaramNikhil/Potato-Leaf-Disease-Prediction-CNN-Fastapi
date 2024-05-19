import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
import os
import warnings
import matplotlib.pyplot as plt
import cv2
import random
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")


def Image_Dir_Func2(data_dir):
    data_image_dir = []
    image_size = 256
    categories = ["Potato___Early_blight", "Potato___healthy", "Potato___Late_blight"]
    try:
        for i in os.listdir(data_dir):
            train_test_val_dirs = os.path.join(data_dir, i)
            for j in os.listdir(train_test_val_dirs):
                cat_index = categories.index(j)
                image_path = os.path.join(train_test_val_dirs, j)
                for k in os.listdir(image_path):
                    image_path_name = os.path.join(image_path, k)
                    data_image = cv2.imread(image_path_name)
                    data_image = cv2.resize(data_image, (image_size, image_size))
                    data_image_dir.append([data_image, cat_index])
        return data_image_dir
    except Exception as e:
        print("Error Generated image directory")
        raise e


data_dir = r"D:\my_projects\Potato-Disease-Prediction-CNN\data\Potato"
Data = Image_Dir_Func2(data_dir)

random.shuffle(Data)

traning_data, testing_data = train_test_split(Data)

X1 = []
y1 = []

for features, label in traning_data:
    X1.append(features)
    y1.append(label)

X2 = []
y2 = []

for features, label in testing_data:
    X2.append(features)
    y2.append(label)


train_data_array_input = np.array(X1)
train_data_array_target = np.array(y1)
test_data_array_input = np.array(X2)
test_data_array_target = np.array(y2)

arr_train = train_data_array_input / 255.0
arr_test = test_data_array_input / 255.0

IMAGE_SIZE = 256
CHANNELS = 3

input_shape = (None, 256, 256, 3)
n_classes = 3
model = models.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.Conv2D(
            32, kernel_size=(3, 3), activation="relu", input_shape=input_shape
        ),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ]
)

model.build(input_shape=input_shape)

model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

history = model.fit(
    arr_train,
    train_data_array_target,
    verbose=1,
    epochs=15,
)
