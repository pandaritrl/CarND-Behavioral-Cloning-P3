import csv
import time
import numpy as np

import cv2
import h5py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Conv2D, MaxPooling2D, Cropping2D
from keras import optimizers
from keras.models import load_model


#%% Function definitions
def load_keras_model(model_file):
    # Load the previous model
    f = h5py.File(model_file, mode='r')
    model_version = f.attrs.get('keras_version')
    model = load_model(model_file)
    return model

def read_csv_log(log_file_path, correction):
    global lines, line
    ## Read the csv file
    lines = []
    with open(log_file_path + 'driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    lines.pop(0)

    ## Read the image file path and measurements
    file_list = []
    measurements = []

    for j in range(len(lines)):
        for i in range(3):
            line = lines[j]
            source_path = line[i]
            tokens = source_path.split('/')
            tokens = tokens[-1].split('\\')
            filename = tokens[-1]
            local_path = log_file_path + "/IMG/" + filename
            # image = mpimg.imread(local_path)
            measurement = float(line[3])
            file_list.append(local_path)
        # center
        measurements.append(measurement)
        # left
        measurements.append(measurement + correction)
        # right
        measurements.append(measurement - correction)
    return file_list, measurements

def get_random_id(length, test_ratio):
    id_max = length *2
    id_list = np.arange(id_max)
    np.random.shuffle(id_list)
    test_length = int(id_max*test_ratio)
    train_id_list = id_list[:test_length]
    valid_id_list = id_list[test_length+1:]
    return train_id_list, valid_id_list

def train_image_generator(id_list, file_list, measurements, batch_size=2):
    data_length = len(file_list)
    id_max = len(id_list)
    i = 0
    while True:
        # Select files (paths/indices) for the batch
        batch_input = []
        batch_output = []

        if i >= id_max:
            i = 0

        # Read in each input, perform preprocessing and get labels
        for j in range(batch_size):
            if id_list[i]>=data_length:
                output = measurements[id_list[i]-data_length]*-1.0
                input = mpimg.imread(file_list[id_list[i]-data_length])
                input = cv2.flip(input, 1)
            else:
                input = mpimg.imread(file_list[id_list[i]])
                output = measurements[id_list[i]]


            batch_input.append(input)
            batch_output.append(output)
            i=i+1
            if i >= id_max:
                break
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        print()
        print("Training Data Ingestion:{}".format(i))
        assert (batch_x.shape[0] == batch_y.shape[0])
        yield (batch_x, batch_y)

def valid_image_generator(id_list, file_list, measurements, batch_size=2):
    data_length = len(file_list)
    id_max = len(id_list)
    i = 0
    while True:
        # Select files (paths/indices) for the batch
        batch_input = []
        batch_output = []

        if i >= id_max:
            i = 0

        # Read in each input, perform preprocessing and get labels
        for j in range(batch_size):
            if id_list[i] >= data_length:
                output = measurements[id_list[i] - data_length] * -1.0
                input = mpimg.imread(file_list[id_list[i] - data_length])
                input = cv2.flip(input, 1)
            else:
                input = mpimg.imread(file_list[id_list[i]])
                output = measurements[id_list[i]]

            batch_input.append(input)
            batch_output.append(output)
            i = i + 1
            if i >= id_max:
                break
        # Return a tuple of (input,output) to feed the network
        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)
        print()
        print("Validation Data Ingestion:{}".format(i))
        assert (batch_x.shape[0] == batch_y.shape[0])
        yield (batch_x, batch_y)

def create_keras_model(image_size_x, image_size_y, image_size_z):
    # layer parameters
    k1, k2, k3, k4, k5 = 5, 5, 5, 3, 3
    f1, f2, f3, f4, f5 = 24, 36, 48, 64, 64
    s1, s2, s3, s4, s5 = 2, 2, 2, 1, 1
    fcn1, fcn2, fcn3, fcn4 = 1164, 100, 50, 10

    model = Sequential()
    model.add(Lambda(lambda x: (x - 128) / 128, input_shape=(image_size_x, image_size_y, image_size_z)))
    # model.add(Cropping2D(cropping=((70, 25), (0, 0)))) # Used for simpler track 1 without elevations
    model.add(Cropping2D(cropping=((30, 15), (0, 0))))
    model.add(Conv2D(filters=f1, kernel_size=(k1, k1), strides = (s1, s1), activation='elu'))
    model.add(Conv2D(filters=f2, kernel_size=(k2, k2), strides = (s2, s2), activation='elu'))
    model.add(Conv2D(filters=f3, kernel_size=(k3, k3), strides = (s3, s3), activation='elu'))
    model.add(Conv2D(filters=f4, kernel_size=(k4, k4), strides = (s4, s4), activation='elu'))
    model.add(Conv2D( filters=f5,kernel_size=(k5, k5), strides = (s5, s5), activation='elu'))
    model.add(Dropout(rate=0.5))
    model.add(Flatten())
    model.add(Dense(fcn1, activation='elu'))
    model.add(Dense(fcn2, activation='elu'))
    model.add(Dense(fcn3, activation='elu'))
    model.add(Dense(fcn4, activation='elu'))
    model.add(Dense(1))

    # compile and fit the model
    adamOpt = optimizers.adam(lr=0.0007)
    model.compile(optimizer=adamOpt, loss='mse', metrics=['accuracy'])
    return model

#%%
correction = 0.8
#%%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
tf.Session(config = tf.ConfigProto(gpu_options=gpu_options))
#%% Path of data folders to use
csv_paths=[]

# Track 1
csv_paths.append('./data/udacityDataM1/')
csv_paths.append('./data/fullLap1M1/')
csv_paths.append('./data/redstripeCornerM1/')
csv_paths.append('./data/bridgeDirtM1/')
csv_paths.append('./data/bridgeDirtM1/')
csv_paths.append('./data/bridgeDirtM1/')
csv_paths.append('./data/udacityDataM1/')

# Track 2
csv_paths.append('./data/jungleLap1/')
csv_paths.append('./data/jungleLap2/')
csv_paths.append('./data/jungleLap3/')
csv_paths.append('./data/jungleHairPin1/')
csv_paths.append('./data/jungleHairPin2/')
csv_paths.append('./data/JungleUp1/')
csv_paths.append('./data/JungleUp1/')
csv_paths.append('./data/JungleUp1/')
csv_paths.append('./data/jungleRedSignCurve1/')
csv_paths.append('./data/jungleRedSignCurve1/')
csv_paths.append('./data/jungleRedSignCurve2/')
csv_paths.append('./data/jungleRedSignCurve3/')
csv_paths.append('./data/jungleHairPinNoGuard1/')
csv_paths.append('./data/jungleHairPinNoGuard1/')

#%%
measurements = []
file_lists = []
for csv_path in csv_paths:
    [file_list, measurement] = read_csv_log(csv_path, correction)
    measurements.extend(measurement)
    file_lists.extend(file_list)

train_batch_size = 1200
valid_batch_size = 1000

[train_id_list, valid_id_list] = get_random_id(len(file_lists), 0.9)
print("Training data size : {}".format(len(train_id_list)))
print("Validation data size : {}".format(len(valid_id_list)))

# Data generators for keras
train_generator = train_image_generator(train_id_list, file_lists, measurements, batch_size=train_batch_size)
valid_generator = valid_image_generator(valid_id_list, file_lists, measurements, batch_size=valid_batch_size)

#%%
# [X_train, y_train] = (next(train_generator))
# offset = 0
# plot_no=2
# fig, axs = plt.subplots(plot_no,1, figsize=(12, 14))
# fig.subplots_adjust(hspace = .4, wspace=.2)
# axs = axs.ravel()
#
# for i in range(plot_no):
#     axs[i].axis('off')
#     axs[i].imshow(X_train[offset+i]) # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     axs[i].set_title('Meas:{}'.format(y_train[offset+i]))
# plt.show()
#%% setup tensorboard
NAME = "cloning-{}".format\
    (correction,int(time.time()))
tensorboard = TensorBoard(log_dir = 'logs\{}'.format(NAME), histogram_freq=0, batch_size=train_batch_size, write_graph=True,
                             write_images=True)


#%% Building keras model
image_size_x,image_size_y,image_size_z = [160,320,3]
model = create_keras_model(image_size_x, image_size_y, image_size_z)
#%% Loading keras model
# model = load_keras_model(model_file = "model-m1-m2-c-0_4.h5")
#%% Model summary
model.summary()
#%% Train keras model
history = model.fit_generator(train_generator, validation_data=valid_generator,
                              steps_per_epoch=len(train_id_list) // train_batch_size, validation_steps=1,
                              epochs=3, callbacks=[tensorboard])
model.save('model-m1-m2-c-0_4.h5')
#%% serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)

