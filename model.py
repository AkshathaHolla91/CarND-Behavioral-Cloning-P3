import csv
import cv2
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation, Cropping2D,ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import sklearn
 
lines=[]
with open("./data/driving_log.csv") as csvfile:
    reader=csv.reader(csvfile)
    for line in reader:
        lines.append(line)
#os.listdir(".")
#print(len(lines))   
lines=lines[1:]
train_samples, validation_samples = train_test_split(lines, test_size=0.2)
# images=[]
# measurements=[]

def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
#                 correction=1
                name_center = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_image_flipped = np.fliplr(center_image)

#                 name_left = './data/IMG/'+batch_sample[1].split('/')[-1]
#                 left_image = cv2.imread(name_left)
#                 name_right = './data/IMG/'+batch_sample[2].split('/')[-1]
#                 right_image = cv2.imread(name_right)
                
                center_angle = float(batch_sample[3])
                center_angle_flipped = -center_angle
#                 left_angle = float(batch_sample[3])+correction
#                 right_angle = float(batch_sample[3])-correction
                images.append(center_image)
                images.append(center_image_flipped)
#                 images.append(left_image)
#                 images.append(right_image)
                angles.append(center_angle)
                angles.append(center_angle_flipped)
#                 angles.append(left_angle)
#                 angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)
# print(next(train_generator)[0].shape)
ch, row, col = 3, 80, 320  # Trimmed image format
# for line in lines:
#     source_path=line[0]
#     filename=source_path.split('/')[-1]
#     current_path='./data/IMG/'+filename
#     image=cv2.imread(current_path)
#     images.append(image)
#     #print(line[3])
#     measurement=float(line[3])
#     measurements.append(measurement)
          
# X_train=np.array(images)
# y_train=np.array(measurements)
#print(np.shape(X_train))
#print(y_train)
model=Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: x/255.0 - 0.5))
model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(ELU())
model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
model.add(Flatten())
model.add(Dropout(.2))
model.add(ELU())
model.add(Dense(512))
model.add(Dropout(.5))
model.add(ELU())
model.add(Dense(1))

#Lenet model with 15 epochs

# model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Dropout(0.5))
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(Flatten())
# model.add(Dense(120))
# model.add(Dense(84))
# model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
# #model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
model.fit_generator(train_generator, samples_per_epoch= len(train_samples)*2, validation_data=validation_generator, nb_val_samples=len(validation_samples)*2, nb_epoch=15)
model.save("model.h5")