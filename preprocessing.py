import cv2 as cv
import numpy as np
 
def detect_and_resize(img):
    #Get height and width of img
    height,width = img.shape[:2]

    #Identify faces using Haar Cascades
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, 1.3, 1)
  
    if isinstance(faces, tuple):
        resized_img = cv.resize(img, (48,48))
        
    elif isinstance(faces, np.ndarray):
        for (x,y,w,h) in faces:
            if w * h < (height * width) / 3:
                resized_img = cv.resize(img, (48,48))
            else:
                roi_gray = img[y:y+h, x:x+w]
                resized_img = cv.resize(roi_gray, (48,48))

    return resized_img    