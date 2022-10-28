

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
from tensorflow.keras.models import load_model
from PIL import Image
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
                                          target_size=(90,90),batch_size =1,
                                          class_mode='categorical')


test_dataset = test.flow_from_directory('/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/att_faces/Testing',
                                          target_size=(90,90), batch_size=1, 
                                          class_mode='categorical')


# label of test data
test_labels = test_dataset.labels
print("test_labels:",test_labels)

dir_path = "/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/att_faces/Testing/"


model = load_model('/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/model_vgg16_1.h5')
model.load_weights('/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/weight_vgg16_1.h5')

#  vì class indices 
#print(test_dataset.class_indices)

#{'s1': 0, 's10': 1, 's11': 2, 's12': 3, 's13': 4, 's14': 5, 's15': 6, 's16': 7, 's17': 8, 's18': 9, 's19': 10, 's2': 11, 's20': 12, 's21': 13, 's22': 14, 's23': 15, 's24': 16, 's25': 17, 's26': 18, 's27': 19, 's28': 20, 's29': 21, 's3': 22, 's30': 23, 's31': 24, 's32': 25, 's33': 26, 's34': 27, 's35': 28, 's36': 29, 's37': 30, 's38': 31, 's39': 32, 's4': 33, 's40': 34, 's5': 35, 's6': 36, 's7': 37, 's8': 38, 's9': 39}
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
dir = os.listdir(dir_path)
pred_list = []
for i in dir:
   if i != '.DS_Store':
        print(i)
        t = "/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/att_faces/Testing/" + i
        for j in os.listdir(t):
            img = image.load_img(dir_path + i + "/" + j, target_size=(90, 90))


        x = image.img_to_array(img)
        
        #Thêm 1 chiều nữa cho giống với dữ liệu đầu vào
        x = np.expand_dims(x, axis=0)
        
        images = np.vstack([x])
        
        #Dự báo nhãn
        x=images/255 
        pred = model.predict([x])[0]
        #print(pred.shape)
        pred = np.argmax(pred)
        #print(pred)
        print(pred,results[pred])
        #print(pred)

        #plt.imshow(img)
        #plt.show()

print(pred_list)
#print(accuracy_score(test_labels, pred_list))