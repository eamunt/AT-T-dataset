
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import tensorflow as tf
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

print(tf.__version__)

# cv2.imread("/content/drive/MyDrive/data/Training/s1/1.png").shape
# #doc hinh anh bang opencv (chieu cao,chieu rong, so chieu)

# img = image.load_img("/content/drive/MyDrive/data/Training/s1/1.png")
# plt.imshow(img)

train = ImageDataGenerator(rescale=1/255)
#doc hinh anh


test = ImageDataGenerator(rescale=1/255)

train_dataset = train.flow_from_directory('D:/CNN/data/Training',
                                          target_size=(90,90),batch_size =1,
                                          class_mode='binary')


test_dataset = test.flow_from_directory('D:/CNN/data/Testing/',
                                          target_size=(90,90), batch_size=1, 
                                          class_mode='binary')

test_labels = test_dataset.labels
print (test_labels )

test_dataset.class_indices

# Xây dựng mô hình

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(90,90,3)),
#     tf.keras.layers.Dense(32, activation='relu'),

#     tf.keras.layers.Dense(64, activation='relu'),

#     tf.keras.layers.Dense(128, activation='relu'),
  
#     tf.keras.layers.Dense(256, activation='relu'),
  
  

#     tf.keras.layers.Dense(40)
# ])

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Flatten,Dense,Activation
 
model=Sequential()
# Thêm Convolutional layer với 16 kernel, kích thước kernel 3*3
# dùng hàm relu làm activation và chỉ rõ input_shape cho layer đầu tiên
model.add(Conv2D(16,(3,3),activation='relu',input_shape=(90,90,3)))
model.add(MaxPooling2D(pool_size=(2,2)))

# Thêm Convolutional layer
model.add(Conv2D(16,(3,3),activation='relu'))
# Thêm Max pooling layer
model.add(MaxPooling2D(pool_size=(2,2)))

# Flatten layer chuyển từ tensor sang vector
model.add(Flatten())
# Thêm Fully Connected layer với 64 nodes và dùng hàm relu
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
# Output layer với 40 node và dùng softmax function để chuyển sang xác xuất.
model.add(Dense(40,activation='softmax'))

#categorical_crossentropy : dùng classifier nhiều class,metrics:Để đánh giá accuracy của models,optimizer:Dùng để chọn thuật toán training
# Biên dịch mô hình (compile)
# – Gắn kết mô hình với bộ tối ưu, hàm loss, và tiêu
# chí đánh giá trong quá trình huấn luyện
# • Optimizer: adam (một dạng của Gradient descent)
# • Loss: SparseCategoricalCrossentropy
# • Metrics: accuracy
model.compile(optimizer ='adam',
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ['accuracy'])
model.summary()
#total params:tong tham so
#Trainable params:Các yếu tố có thể huấn luyện

history = model.fit(train_dataset, epochs = 50, validation_data = test_dataset)



model.save('model.h5')

#Vẽ đồ thị loss, accuracy của traning set và validation set
history_dict = history.history
import matplotlib.pyplot as plt
loss = history_dict['loss']
val_loss = history_dict['val_loss']
acc = history_dict['accuracy']
val_acc =history_dict['val_accuracy']
epochs = range(1,len(acc)+1)
plt.plot(epochs,loss,'r',label = 'Loss')
plt.plot(epochs,acc,'b',label='Accuracy')
plt.plot(epochs,val_loss,'g',label = 'val_Loss')
plt.plot(epochs,val_acc,'c',label='val_Accuracy')
plt.title('Loss and Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# tạo mô hình dự báo
probability_model = tf.keras.Sequential([model,tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_dataset)
print (predictions)
predictions[0]

from PIL import Image
dir_path = "D:/CNN/data/Test/"

results={
 0  : 'Đối tượng 1' ,                                             
 1 :'Đối tượng 10',
 2 :'Đối tượng 11' ,
 3  :'Đối tượng 12',
 4  :'Đối tượng 13',
 5  :'Đối tượng 14',
 6  :'Đối tượng 15',
 7  :'Đối tượng 16',
 8  :'Đối tượng 17',
 9  :'Đối tượng 18',
 10 :'Đối tượng 19',
 11 :'Đối tượng 2',
 12 :'Đối tượng 20',
 13 :'Đối tượng 21',
 14 :'Đối tượng 22',
 15 :'Đối tượng 23',
 16 :'Đối tượng 24',
 17 :'Đối tượng 25',
 18 :'Đối tượng 26',
 19 :'Đối tượng 27',
 20 :'Đối tượng 28',
 21 :'Đối tượng 29',
 22  :'Đối tượng 3',
 23 :'Đối tượng 30',
 24 :'Đối tượng 31',
 25 :  'Đối tượng 32',
 26 :  'Đối tượng 33',
 27 :  'Đối tượng 34',
 28 :  'Đối tượng 35',
 29 :  'Đối tượng 36',
 30 :  'Đối tượng 37',
 31 :  'Đối tượng 38',
 32 :  'Đối tượng 39',
 33 :    'Đối tượng 4',
 34 :  'Đối tượng 40',
 35 :  'Đối tượng 5',
 36 :  'Đối tượng 6',
  37:    'Đối tượng 7',
38  :  'Đối tượng 8',
 39 :     'Đối tượng 9',


}
for i in os.listdir(dir_path):
   print (i)
   img = image.load_img(dir_path + i, target_size=(90, 90))
   # print (img)

   x = image.img_to_array(img)
   
  #  Thêm 1 chiều nữa cho giống với dữ liệu đầu vào
   x = np.expand_dims(x, axis=0)
  
   images = np.vstack([x])
  
  #  Dự báo nhãn
   x=images/255 
   pred = model.predict_classes([x])[0]
   print(pred,results[pred])
   plt.imshow(img)
   plt.show()
