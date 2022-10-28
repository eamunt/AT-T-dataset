import os
import uuid
import flask
import urllib
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)
#
model = load_model("/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/model_vgg16_1.h5")
model.load_weights("/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/weight_vgg16_1.h5")

#class_names = ['s1', 's10', 's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's2', 's20', 's21', 's22', 's23', 's24', 's25', 's26', 's27', 's28', 's29', 's3', 's30', 's31', 's32', 's33', 's34', 's35', 's36', 's37', 's38', 's39', 's4', 's40', 's5', 's6', 's7', 's8', 's9']
class_names = ['Nhân vật 1', ' Nhân vật 10', 'Nhân vật 11', 'Nhân vậtg 12', 'Nhân vật 13', 'Nhân vật 14', 'Nhân vật 15', 'Nhân vật 16', 'Nhân vật 17', 
    'Nhân vật 18', 'Nhân vật 19', 'Nhân vật 2', 'Nhân vật 20', 'Nhân vật 21', 'Nhân vật 22', 'Nhân vật 23', 'Nhân vật 24', 'Nhân vật 25', 'Nhân vật 26', 'Nhân vật 27', 'Nhân vật 28', 'Nhân vật 29', 'Nhân vật 3', 'Nhân vật 30', 
    'Nhân vật 31', 'Nhân vật 32', 'Nhân vật 33', 'Nhân vật 34', 'Nhân vật 35', 'Nhân vật 36', 'Nhân vật 37', 'Nhân vật 38', 'Nhân vật 39', 'Nhân vật 4', 'Nhân vật 40', 'Nhân vật 5', 'Nhân vật 6', 'Nhân vật 7', 'Nhân vật 8', 'Nhân vật 9']
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif', 'gif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
def predict(filename , model):


    from keras.preprocessing.image import load_img
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing.image import array_to_img
    # load the image
    img = load_img(filename, target_size=(90,90))
    #print(type(img))
    # convert to numpy array
    x = img_to_array(img)
        
    #Thêm 1 chiều nữa cho giống với dữ liệu đầu vào
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    
    #Dự báo nhãn
    x=images/255 
    predictions = model.predict([x])[0]
    # print(predictions)
    #print(pred.shape)
    prediction = np.argmax(predictions)
    #print(pred)
    # print(prediction,class_names[prediction])
    # print(prediction)
    return class_names[prediction]
@app.route('/')
def home():
        return render_template("index.html")
@app.route('/success' , methods = ['GET' , 'POST'])
def success():
    error = ''
    target_img = os.path.join(os.getcwd() , '/Users/haokahnguyen/Desktop/course/May_hoc_nang_cao/CNN/static/images')
    if request.method == 'POST':
     
            
        if (request.files):
            file = request.files['file']
            if file and allowed_file(file.filename):
                file.save(os.path.join(target_img , file.filename))
                img_path = os.path.join(target_img , file.filename)
                img = file.filename
                class_result= predict(img_path , model)
                predictions = {
                      "class1":class_result
                }
            else:
                error = "Vui lòng chỉ tải lên hình ảnh của phần mở rộng jpg, jpeg và png"
            if(len(error) == 0):
                return  render_template('success.html' , img  = img , predictions = predictions)
            else:
                return render_template('index.html' , error = error)
    else:
        return render_template('index.html')
if __name__ == "__main__":
    app.run(debug = True)

