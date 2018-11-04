import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
import matplotlib.pyplot as plt

import preprocessor

LR = 0.001
N_EPOCH = 100
IMG_SIZE = 50
PERFORM_TRAINING = 0
TRAIN_DIR = 'train'
TEST_DIR = 'test'
ALL_EMOTIONS = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
MODEL_NAME = '{}-recognition-using-cnn-{}.model'.format('mixed', LR)

# --------------------------------------------------------------------------------------------------------
train = []
test = []
div = 0.2

def create_train_data():
    print('---Creating training and testing data---')
    training_data = []
    for emotions_folder in os.listdir(TRAIN_DIR):
        emotions_dir = os.path.join(TRAIN_DIR, emotions_folder)

        label = [0, 0, 0, 0, 0, 0, 0]
        for idx, val in enumerate(ALL_EMOTIONS):
            if(val == emotions_folder):
                label[idx] = 1

        emotion_face = []
        for img_name in tqdm(os.listdir(emotions_dir)):
            img_path = os.path.join(emotions_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            emotion_face.append([np.array(img), np.array(label)])
        shuffle(emotion_face)
        size=len(emotion_face)
        print("training size of", emotions_folder, size, div*size)
        for i, data in enumerate(emotion_face):
            if(i<div*size): test.append(data)
            else: train.append(data)    
        shuffle(train)
        shuffle(test)
        np.save('data/train.npy', train)
        np.save('data/test.npy', test)

# --------------------------------------------------------------------------------------------------------
def process_test_data():
    print('---Creating testing data in test_data.npy---')
    testing_data = []
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR,img)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
        testing_data.append(np.array(img))
        
    shuffle(testing_data)
    np.save('data/test_data.npy', testing_data)
    return testing_data

# --------------------------------------------------------------------------------------------------------
print('---Loading existing data---')
# create_train_data()
train = np.load('data/train.npy')
test = np.load('data/test.npy')

test_data = process_test_data()

# --------------------------------------------------------------------------------------------------------
print('---Creating network model---')
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 2, activation='relu')
convnet = max_pool_2d(convnet, 2)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 7, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='logs')

# --------------------------------------------------------------------------------------------------------
print('---Training the model---')

class EarlyStoppingCallback(tflearn.callbacks.Callback):
    def __init__(self, val_acc_thresh):
        self.val_acc_thresh = val_acc_thresh
    
    def on_epoch_end(self, training_state):
        if training_state.val_acc is None: return
        if training_state.val_acc > self.val_acc_thresh:
            self.val_acc_thresh = training_state.val_acc
            model.save('model/'+MODEL_NAME)
            print('------------------model saved-------------------', training_state.val_acc)

if(PERFORM_TRAINING):
    # Initializae our callback.
    early_stopping_cb = EarlyStoppingCallback(val_acc_thresh=0.5)

    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]
    try:
        model.fit({'input': X}, {'targets': Y}, n_epoch=N_EPOCH, validation_set=({'input': test_x}, {'targets': test_y}), 
            snapshot_step=1000, show_metric=True, batch_size=101, run_id=MODEL_NAME, callbacks=early_stopping_cb)
    except StopIteration:
        print("Caught callback exception. Returning control to user program.")

# --------------------------------------------------------------------------------------------------------
if os.path.exists('model/{}.meta'.format(MODEL_NAME)):
    print('Loading trained model')
    model.load('model/'+MODEL_NAME)
else:
    print('Model not found')

# --------------------------------------------------------------------------------------------------------
# print('---Checking testing output---')
# fig = plt.figure()
# error_count = 0
# for num,data in enumerate(test):
#     img_data = data[0]
#     img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
#     model_out = model.predict([img_data])[0]
#     str_label = ALL_EMOTIONS[np.argmax(model_out)]
#     actual = ALL_EMOTIONS[np.argmax(data[1])]
#     if(actual == str_label):
#         found = 1
#     else:
#         found = 0
#         error_count += 1
#     show = '{}-{}'.format(str_label, found)

#     y = fig.add_subplot(8, 11,num+1)
#     y.imshow(data[0], cmap = 'gray', interpolation = 'bicubic')
#     plt.title(show)
#     y.axes.get_xaxis().set_visible(False)
#     y.axes.get_yaxis().set_visible(False)
# print('Found', error_count, 'errors out of', len(test), 'images')
# print('Accuracy on test data: %.2f %%' % ((1-(error_count/len(test)))*100))
# plt.show()

# --------------------------------------------------------------------------------------------------------
print('---Unseen test---')
fig = plt.figure()

for num,data in enumerate(test_data):
    img_data = data
    img_data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
    model_out = model.predict([img_data])[0]

    str_label = ALL_EMOTIONS[np.argmax(model_out)]
    show = '{}'.format(str_label)

    y = fig.add_subplot(5,10,num+1)
    y.imshow(data, cmap = 'gray', interpolation = 'bicubic')
    plt.title(show)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()