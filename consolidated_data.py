import cv2
import os
import pickle #file format h pickle ka mtlv ,#yha folder ko compress kr dense 
import numpy as np

data_dir = os.path.join(os.getcwd(),'clean_data') #folder ko conncet krega"path.join, aur getcwd bttat h ki kis directlt pe kaam kr rhe h
img_dir = os.path.join(os.getcwd(),'images')


image_data = []
labels = []

for i in os.listdir(img_dir): #us particular folder me file ya images joavailable j wo dikhayega
    image = cv2.imread(os.path.join(img_dir,i))
    image = cv2.resize(image,(100,100))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image_data.append(image)
    labels.append(str(i).split("_")[0]) #ye name file ka split kr rha h aur 0 index no lena h bs
    
image_data = np.array(image_data)    
labels = np.array(labels) 

import matplotlib.pyplot as plt
plt.imshow(image_data[78],cmap="gray")
plt.show()


with open(os.path.join(data_dir,"images.p"),'wb') as f:   #.p picke file ka etension
    pickle.dump(image_data,f)
    
with open(os.path.join(data_dir,"labels.p"),'wb') as f:
    pickle.dump(labels,f)
    

