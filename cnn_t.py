from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score


img = image.load_img("/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/att_faces/Testing/s1/1.jpg")

print(img)

#chuyển đổi PIL image into numpy array
x = image.img_to_array(img)
print(x.shape)


#scale to [0,1]
train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

# flow from dicrectory
train_dataset = train.flow_from_directory('/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/att_faces/Training',
                                          target_size=(90,90),batch_size = 1,
                                          class_mode='categorical')


test_dataset = test.flow_from_directory('/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/att_faces/Testing',
                                          target_size=(90,90), batch_size = 1, 
                                          class_mode='categorical')

#print(train_dataset.image_shape)

# label of test data
test_labels = test_dataset.labels
print("test_labels:",test_labels)

print(test_dataset.class_indices)

# build model with vgg16
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Lambda, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

vgg = VGG16(input_shape=[90, 90, 3], weights='imagenet', include_top=False)
# include_top = Flase vì ta trong VGG dùng convolution layers để trích xuất các đặc trưng
# và thêm Dense Layer để phân loại hình ảnh của tập AT&T (vì không phải là làm cho tập imagenet)


# không train với các trọng số sẵn có (đã có được khi đào tạo tập imagenet)
for layer in vgg.layers:
  layer.trainable = False

# Thêm layer Flatten sau 
x = Flatten()(vgg.output)

# tầng Dense: output model
    # 40 lớp
prediction = Dense(40, activation='softmax')(x)
model = Model(inputs=vgg.input, outputs=prediction)

model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
model.summary()

# history = model.fit(train_dataset, epochs = 5, validation_data = test_dataset)


# model.save('model_vgg16_1.h5')

# model.save_weights('weight_vgg16_1.h5')
