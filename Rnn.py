import glob
import numpy as np
import os
import shutil
import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn import metrics
from keras import metrics
from keras.applications import vgg16
from keras.models import Model,load_model
import keras
import tkinter as tk
from tkinter import filedialog
import winsound
def rnn():

    model = load_model('rnn11Lstm.h5')
    IMG_DIM = (150, 150)
    input_shape = (150, 150, 3)
    filename = filedialog.askopenfilename()
    cap = cv2.VideoCapture(filename)
    c = 0
    x = 0
    test_frames = []
    test_framesz = []
    cz = 0
    z = 0
    while (cap.isOpened()):

        z = z + 1
        x = x + 1
        ret, frame = cap.read()

        if (ret != True and c < 15):
            cv2.rectangle(frame, (5, 2), (55, 13), (255, 255, 255), -1)
            train_imgs = [img_to_array(img) for img in test_framesz[0:cz]]
            print("l", len(test_framesz))
            train_imgs = np.array(train_imgs)
            train_imgs_scaled = train_imgs.astype('float32')
            train_imgs_scaled /= 255
            vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                              input_shape=input_shape)

            output = vgg.layers[-1].output
            output = keras.layers.Flatten()(output)
            vgg_model = Model(vgg.input, output)

            vgg_model.trainable = False
            for layer in vgg_model.layers:
                layer.trainable = False

            # input_shape = vgg_model.output_shape[1]
            features = vgg_model.predict(train_imgs_scaled, verbose=0)
            print("leeeee", len(features))
            frames_num = 15
            count = 0
            joint_transfer = []
            for i in range(int(len(features) / frames_num)):
                inc = count + frames_num
                joint_transfer.append([features[count:inc]])
                count = inc

            data = []
            for i in joint_transfer:
                data.append(i[0])
            data = np.array([data])
            # target=np.array([target])
            print("data", data.shape)
            data = data.reshape(data.shape[0] * data.shape[2], data.shape[1], data.shape[3])
            print("data", data.shape)
            data = np.reshape(data, (data.shape[1], data.shape[0], data.shape[2]))
            features = model.predict(data, verbose=0)

            if features[0][0] < features[0][1]:
                winsound.PlaySound("F:/CV-Competation/alarm.wav", winsound.SND_ASYNC)
                break
            print(features)
            break
        if (ret != True):
            break
        frame = cv2.resize(frame, (150, 150))
        # frame1 = cv2.cvtColor(frame1_, cv2.COLOR_BGR2GRAY)
        if (x % 5 == 0 and x != 0):
            print("c=====", c)
            c = c + 1
            test_frames.append(frame)
        if (z % 3 == 0 and z != 0):
            print("z=====", z)
            cz = cz + 1
            test_framesz.append(frame)

        if (x % 75 == 0 and x != 0):
            print(1)
            train_imgs = [img_to_array(frame) for img in test_frames[c - 15:c]]
            print("l", len(test_frames))
            train_imgs = np.array(train_imgs)
            train_imgs_scaled = train_imgs.astype('float32')
            train_imgs_scaled /= 255
            vgg = vgg16.VGG16(include_top=False, weights='imagenet',
                              input_shape=input_shape)

            output = vgg.layers[-1].output
            output = keras.layers.Flatten()(output)
            vgg_model = Model(vgg.input, output)

            vgg_model.trainable = False
            for layer in vgg_model.layers:
                layer.trainable = False

            # input_shape = vgg_model.output_shape[1]
            features = vgg_model.predict(train_imgs_scaled, verbose=0)
            print("leeeee", len(features))
            frames_num = 15
            count = 0
            joint_transfer = []
            for i in range(int(len(features) / frames_num)):
                inc = count + frames_num
                joint_transfer.append([features[count:inc]])
                count = inc

            data = []
            for i in joint_transfer:
                data.append(i[0])
            data = np.array([data])
            # target=np.array([target])
            print("data", data.shape)
            data = data.reshape(data.shape[0] * data.shape[2], data.shape[1], data.shape[3])
            print("data", data.shape)
            data = np.reshape(data, (data.shape[1], data.shape[0], data.shape[2]))
            features = model.predict(data, verbose=0)

            if features[0][0] < features[0][1]:
                winsound.PlaySound("F:/CV-Competation/alarm.wav", winsound.SND_ASYNC)
                print('Suspicious Activity Detected')
                plt.show()

            print(features)

        cv2.rectangle(frame, (5, 2), (65, 13), (255, 255, 255), -1)
        cv2.putText(frame, str("sus_Activity"), (10, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
        frame = cv2.resize(frame, (300, 300))
        cv2.imshow('frame', frame)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

root = tk.Tk()
button = tk.Button(root, text='Choose the video', command=rnn)
button.pack()

root.mainloop()
