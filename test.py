# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import scipy
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
from keras.models import load_model
model=load_model('/kaggle/input/saved_model.h5')


dir1 = "/kaggle/input/gnr638_test/"
temp1 = os.listdir(dir1)
print(temp1)

# %% [code]
vec1 = []
temp3 = os.listdir(dir1+"test")

l1 = ["/kaggle/input/gnr638_test/test/"]*len(temp3)
temp3 = list(map(lambda x,y:x+y,l1,temp3))
print(temp3)
img = sorted(temp3)
print(img)

# %% [code]
img_ar = []
for img_path in img:
    #print(type(img_path))
    #print(img_path)
    image = scipy.misc.imread(img_path)
    image = scipy.misc.imresize(image,(200,200,3),interp="cubic")
    img_ar.append(image)
    
img_ar = np.array(img_ar)  
print(type(img_ar))

# %% [code]
temp=model.predict(img_ar)
predicted = []
for x in temp:
    x=list(x)
    #print(x)
    idx=x.index(max(x))
    #print(idx)
    predicted.append(idx)
print(predicted)    


# %% [code]
predicted=np.array(predicted)
#print(predicted)
predicted=predicted+1
#print(predicted)
predicted=list(predicted)
print(predicted)

import csv
# %% [code]
with open("170070038.csv","w") as f:
    writer1 = csv.writer(f)
    writer1.writerow(["imageId"] + ["label"])

for idx,val in enumerate(img):
    v1 = val.split("/")[-1].split(".")[0]
    print(v1)
    print(predicted[idx])
    with open("170070038.csv","a") as f:
        writer1 = csv.writer(f)
        writer1.writerow([v1] + [predicted[idx]])