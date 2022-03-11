import numpy as np
import tensorflow as tf
from keras import Model, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from extractNamesFromPics import extract_csv
from generateEncodings import import_encodings

csv_file = extract_csv('data-cleaned.csv')
appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
appearances['Id'] = csv_file[0][4:]
model_save_path = 'model.h5'


import keras.backend as K

out_items = ['SkinColour', 'UpperEyelidType', 'BottomEyelidType', 'FacialHairType']

def custom_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype='float32')
    loss = K.abs(y_pred - y_true)  # (batch_size, 2)
    return K.sum(loss, axis=1)


def prepare_data(encodings):
    in_train = []
    out_train = []
    out_idxs = [appearances['Id'].index(item) for item in out_items]
    for id, encoding in encodings.items():
        if id in appearances:
            in_train.append(encoding)
            out = []
            out = [appearances[id][i] for i in out_idxs]
            # out = appearances[id]
            out_train.append(out)
    in_train = np.array(in_train)
    out_train = np.array(out_train)
    return in_train, out_train


def build_new_model():
    from keras.layers import Input, Dense

    Input_1 = Input(shape=(128,))

    x = Dense(512, activation='relu')(Input_1)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(512, activation='softmax')(x)

    outs = [Dense(1, activation='linear', name=f'{i[:13]}{out_items.index(i)}')(x) for i in out_items]
    # out1 = Dense(1, activation='linear')(x)
    # out2 = Dense(1, activation='linear')(x)
    # out3 = Dense(1, activation='linear')(x)

    model = Model(inputs=Input_1, outputs=outs)
    # model = Model(inputs=Input_1, outputs=[out1, out2, out3])
    model.compile(optimizer='rmsprop', loss=custom_loss, metrics=['mse'])
    return model

if __name__ == '__main__':

    encodings = import_encodings('encodings.json')

    in_train, out_train = prepare_data(encodings)
    in_valid, out_valid = [], []
    # print(
    #     f'Training Data: {len(out_train)} - Validation Data: {len(out_valid)} - All Data: {len(out_valid) + len(out_train)}')
    # train_model(in_train, out_train, in_valid, out_valid, 150)

    model = build_new_model()

    history = model.fit(in_train, out_train, epochs=200)
    from matplotlib import pyplot as plt
    print(history.history)
    plt.plot(history.history['loss'])
    plt.show()
    # for name, hist in history.history.items():
      # plt.plot(hist)
    model.save(model_save_path)