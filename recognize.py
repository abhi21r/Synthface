import cv2
import numpy as np
from keras.models import load_model

classifier = cv2.CascadeClassifier(r"C:\Users\sonut\OneDrive\Desktop\Face Detection\Face Detection\haarcascade_frontalface_default.xml")

model = load_model(r"C:\Users\sonut\OneDrive\Desktop\Face Detection\Face Detection\final_model.h5", compile=False)

def get_pred_label(pred):
    labels = ['Abhishek', 'Aniket', 'Pratisha', 'Samikhya']
    return labels[pred]

def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (100, 100))
    img = cv2.equalizeHist(img)
    img = img.reshape(1, 100, 100, 1)
    img = img / 255
    return img

cap = cv2.VideoCapture(0)  # Use default system webcam (index 0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    faces = classifier.detectMultiScale(frame, 1.5, 5)
      
    for x, y, w, h in faces:
        face = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255, 0), 3)
        cv2.putText(frame, get_pred_label(np.argmax(model.predict(preprocess(face)))),
                    (200, 200), cv2.FONT_HERSHEY_COMPLEX, 1,
                    (255, 0, 0), 1)
        
    cv2.imshow("capture", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()