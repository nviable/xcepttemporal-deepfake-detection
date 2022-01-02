from keras.applications.xception import Xception
from keras.layers import Input, TimeDistributed, LSTM, Dense, Dropout, Bidirectional
from keras.utils import Sequence, to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.utils import to_categorical, multi_gpu_model
from matplotlib import pyplot as plt
from math import floor, ceil
from os import listdir, path
import cv2
import numpy as np
import random

def model(learning_rate = 0.0001, decay = 0.00001, frames = 20, classes = 2):
    optimizer = Adam(lr = learning_rate, decay = decay)

    x = Input(shape = (299, 299, 3))
    cnn = Xception(weights='imagenet', include_top=False, pooling='max')

    x_ = TimeDistributed(cnn)(x)

    encoded_video = Bidirectional(LSTM(256, return_sequences = True))(x_)
    encoded_video_2 = Bidirectional(LSTM(256))(encoded_video)
    fc = Dense(1024)(encoded_video_2)
    fc = Dropout(0.5)(fc)
    out = Dense(classes, activation='softmax')(fc)

    model = Model(inputs = [x], outputs = out)
    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # print(model.summary())

    return model


class Generator(Sequence):
    def __init__(self, batch_size=16, stride=8, length=8, path=[]):
        self.batch_size = batch_size
        self.stride = stride
        self.length = length
        self.samples = len(path)
        self.path = path

    def __len__(self):
        return floor(self.samples/self.batch_size)

    def normalize(self, img):
        img = (1/255)*img
        img[:,:,0] -= np.mean(img[:,:,0])
        img[:,:,1] -= np.mean(img[:,:,1])
        img[:,:,2] -= np.mean(img[:,:,2])
        return img

    def __getitem__(self, idx):
        batch_x = []
        batch_y = []
        for j in range(self.batch_size):
            try:
                temp = []
                for k in range(self.length):
                    img = cv2.resize(cv2.imread(self.path[self.batch_size*idx + j][0][k]), (224, 224))
                    choice = random.randint(1,4)
                    if choice == 4:
                        gauss = np.random.normal(0, random.uniform(1,5), (224,224,3))
                        gauss = gauss.reshape(224,224,3)
                        img = img + gauss
                    elif choice == 3:
                        k_size = random.choice([1,3,5,7,9])
                        img = cv2.GaussianBlur(img, (k_size, k_size), cv2.BORDER_DEFAULT)
                    elif choice == 2:
                        _, enc_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, random.randint(15, 85)])
                        img = cv2.imdecode(enc_img, 1)
                    else:
                        pass
                    temp.append(self.normalize(img))
                    
                # batch_x.append([self.normalize(cv2.resize(cv2.imread(self.path[self.batch_size*idx + j][0][k]), (224, 224))) for k in range(self.length)])
                batch_x.append(temp)
                batch_y.append(self.path[self.batch_size*idx + j][1])
            except:
                pass
                # print('FAIL', choice, np.shape(img))
                # print(self.path[self.batch_size*idx + j][0])
        return np.array(batch_x), np.array(batch_y)



def get_path(start, end, dir, y, length=8, stride=8, total=256):
    lst = []
    for i in sorted(listdir(dir))[start:end]:
        path_list = sorted(listdir(dir + i + '/'))
        if (len(path_list) < 256): 
            total = len(path_list)
        else:
            total = 256
        for j in range(int(total/stride)):
            if(dir.find('VP9') == -1):
                temp = ([dir + i + '/' + str(j*length + k) + '.jpg' for k in range(length)], y)
            else:
                temp = ([dir + i + '/' + str(j*length + k).zfill(4) + '.jpg' for k in range(length)], y)
            lst.append(temp)
    return lst

def get_path_list(tuples, shuffle=False):
    """Generate a list of image paths for the training

    Args:
        tuples (tuple): (first file no., last file no., path to preprocessed directory, category) 
        shuffle (bool, optional): [shuffle the list of images]. Defaults to False.

    Returns:
        [list]: [list of image paths]
    """
    lst = []
    for i in tuples:
        lst.extend(get_path(*i))
    
    if(shuffle):
        random.shuffle(lst)
    return lst

if __name__ == "__main__":
    tr_path = []
    val_path = []
    test_path = []
    
    temporal = model(frames=8, classes = 3)
    random.seed(55)

    # training dataset
    tr_path = get_path_list(
        [
            (0, 720, '/data/datasets/FaceForensics++/Faces/NeuralTextures/train/original/', to_categorical(0, 3)),
            (0, 100, '/data/datasets/Celeb-DF/Faces/real/', to_categorical(0, 3)),
            (0, 720, '/data/datasets/FaceForensics++/Faces/DeepFakes/train/altered/', to_categorical(1, 3)),
            (0, 720, '/data/datasets/FaceForensics++/Faces/FaceSwap/train/altered/', to_categorical(1, 3)),
            (0, 720, '/data/datasets/Celeb-DF/Faces/fake/', to_categorical(1, 3)),
            (0, 720, '/data/datasets/FaceForensics++/Faces/Face2Face/train/altered/', to_categorical(2, 3)),
            (0, 720, '/data/datasets/FaceForensics++/Faces/NeuralTextures/train/altered/', to_categorical(2, 3)),
            (0, 720, '/data/datasets/FaceForensics++/FOM_faces/train/', to_categorical(2, 3))
    ], shuffle=True)
    
    # validation dataset
    val_path = get_path_list(
        [
            (0, 140, '/data/datasets/FaceForensics++/Faces/NeuralTextures/val/original/', to_categorical(0, 3)),
            (100, 158, '/data/datasets/Celeb-DF/Faces/real/', to_categorical(0, 3)),
            (0, 140, '/data/datasets/FaceForensics++/Faces/DeepFakes/val/altered/', to_categorical(1, 3)),
            (0, 140, '/data/datasets/FaceForensics++/Faces/FaceSwap/val/altered/', to_categorical(1, 3)),
            (720, 750, '/data/datasets/Celeb-DF/Faces/fake/', to_categorical(1, 3)),
            (0, 140, '/data/datasets/FaceForensics++/Faces/Face2Face/val/altered/', to_categorical(2, 3)),
            (0, 140, '/data/datasets/FaceForensics++/Faces/NeuralTextures/val/altered/', to_categorical(2, 3)),
            (0, 140, '/data/datasets/FaceForensics++/FOM_faces/val/', to_categorical(2, 3))
        ]
    )

    gen_train = Generator(batch_size=2, path=tr_path)
    gen_val = Generator(batch_size=2, path=val_path)
    # gen_test = Generator(batch_size=2, path=test_path)

    # temporal.load_weights('weights/enb4_ce3_50.04-0.95.h5')

    checkpoint_callback = ModelCheckpoint('weights/enb4_ce3_aug.{epoch:02d}-{val_accuracy:.2f}.h5', monitor='val_accuracy', verbose=1, mode='max', period = 1)
    history = temporal.fit_generator(gen_train, epochs=10, validation_data= gen_val, verbose=1, use_multiprocessing=False, callbacks=[checkpoint_callback])
    
    # test_path = get_path(720, 795, '/data/datasets/Celeb-DF/Faces25/fake/', to_categorical(5, 6))
    # gen_test = Generator(batch_size=16, path=test_path)
    # print(temporal.evaluate_generator(gen_test, verbose=1, workers=0))
