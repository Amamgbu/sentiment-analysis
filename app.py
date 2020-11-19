from flask import Flask,render_template,jsonify,request
import keras
from keras.models import load_model
from preprocessing import detect_and_resize
import cv2 as cv
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
    return render_template('home.html')

@app.route('/predict',methods = ['POST'])
def predict():
    with open('model/lb.pickle','rb') as f:
        lb = pickle.load(f)
    emotions = {0:'neutral',1:'anger',2:'disgust',3:'fear',4:'sad',5:'happy',6:'surprise',7:'none_of_the_above'}
    model= load_model('model/emotions_classifier.model')
    img = request.files['file']
    image = plt.imread(img)
    #Change RGB to BGR
    image = image[..., ::-1]
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    resize_img = detect_and_resize(gray_image)
    final_im = resize_img.astype("float") / 255.0
    final_im = img_to_array(final_im)
    final_im = np.expand_dims(final_im,axis=0)
    prediction = model.predict(final_im)
    idx = np.argmax(prediction)
    l = lb.classes_[idx]
    final_pred = emotions[l]
    return jsonify(final_pred)
    

if __name__ == '__main__':
    app.run(port=5000, debug=True)