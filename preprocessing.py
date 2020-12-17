# Standard library imports
import cv2
import os

# Third party imports
import numpy as np
from keras.utils import np_utils

# Compiling category names and potential labels
data_path = 'dataset'
categories = os.listdir(data_path)
labels = [i for i in range(len(categories))]

label_dict = dict(zip(categories, labels))

data = []
target = []

for category in categories:

    print(f'Processing category: {category}')

    # Getting list of items in category
    folder_path = os.path.join(data_path,category)
    img_names = os.listdir(folder_path)

    # Iterating through images
    for img_name in img_names:

        print(f'Processing {category}: {img_name}')

        # Getting path to image
        img_path = os.path.join(folder_path,img_name)
        # Reading image to variable
        img = cv2.imread(img_path)

        try:
            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Standardize size
            resized = cv2.resize(gray, (100,100)) #dataset?
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception: ', e)

    
data = np.array(data)/255.0
data = np.reshape(data,(data.shape[0], 100, 100, 1))
target = np.array(target)

new_target = np_utils.to_categorical(target)

np.save('data', data)
np.save('target', new_target)