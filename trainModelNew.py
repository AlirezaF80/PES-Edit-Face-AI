import re

import numpy as np
from keras import Model

from extractNamesFromPics import extract_csv
from generateEncodings import import_encodings

csv_file = extract_csv('data-cleaned.csv')
appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
appearances['Id'] = csv_file[0][4:]
model_save_path = 'modelNew.h5'

# out_items = ['SkinColour', 'UpperEyelidType', 'BottomEyelidType', 'EyeHeight']
out_items = appearances['Id']


# TODO: Try this post to limit output range:
#  https://stackoverflow.com/questions/54435998/keras-limiting-output-min-max-value
def prepare_data(encodings):
    in_train = []
    out_train = []
    for i in range(len(out_items)):
        out_train.append([])
    out_idxs = [appearances['Id'].index(item) for item in out_items]

    for id, encoding in encodings.items():
        if id in appearances:
            in_train.append(encoding)
            # out = []
            # out = [appearances[id][i] for i in out_idxs]
            # out = appearances[id]
            [out_train[i].append(appearances[id][out_idxs[i]]) for i in range(len(out_idxs))]
    for i in range(len(out_items)):
        out_train[i] = np.array(out_train[i])
    in_train = np.array(in_train)
    out_train = out_train
    return in_train, out_train


def fix_layer_name(name):
    name = re.sub(r'[()]', '_', name)
    return re.sub('[^a-zA-Z0-9.]', '', name)


def build_model():
    from keras.layers import Input, Dense

    Input_1 = Input(shape=(128,))
    outs = []
    for i in range(len(out_items)):
        l = Dense(256, activation='relu')(Input_1)
        # l = Dense(256, activation='relu')(l)
        # l = Dense(256, activation='relu')(l)
        out = Dense(1, activation='linear', name=fix_layer_name(out_items[i]))(l)
        outs.append(out)

    model = Model(inputs=Input_1, outputs=outs)
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def plot_history(history):
    from matplotlib import pyplot as plt

    print(history.keys())
    plt.plot(history['loss'])
    plt.show()


if __name__ == '__main__':
    encodings = import_encodings('encodings.json')

    in_train, out_train = prepare_data(encodings)
    in_valid, out_valid = [], []
    # print(
    #     f'Training Data: {len(out_train)} - Validation Data: {len(out_valid)} - All Data: {len(out_valid) + len(out_train)}')
    # train_model(in_train, out_train, in_valid, out_valid, 150)

    model = build_model()

    history = model.fit(in_train, out_train, epochs=150)
    model.save(model_save_path)
    # model.evaluate(in_train, out_train)
    plot_history(history.history)
