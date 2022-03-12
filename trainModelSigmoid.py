import re
from typing import Dict, List

import numpy as np
from keras import Model

from extractNamesFromPics import extract_csv


def load_json(filename):
    import json
    with open(filename) as f:
        json_data = json.load(f)
        return json_data


csv_file = extract_csv('data-cleaned.csv')
appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
all_items_names: list[str] = [i.strip() for i in csv_file[0][4:]]
model_save_path = 'modelSigmoid.h5'
select_items: List[str] = load_json('items_names.json')
items_val_range: Dict[str, List[int]] = load_json('items_val_range.json')
items_types: Dict[str, Dict[str, str]] = load_json('items_types.json')

select_items_idxs: List[int] = [all_items_names.index(item) for item in select_items]


def prepare_output(val, item_name):
    val_range = items_val_range[item_name]
    item_type = items_types[item_name]['loss']
    if item_type == 'categorical_crossentropy':
        return val_to_categories_arr(val, val_range)
    elif item_type == 'mse':
        return scale_to_sigmoid(val, val_range)


def scale_to_sigmoid(val, val_range):
    min_val, max_val = val_range
    return (val - min_val) / (max_val - min_val)


def val_to_categories_arr(val, val_range):
    min_val, max_val = val_range
    arr = np.zeros(max_val - min_val + 1)
    arr[val - min_val] = 1
    return arr


# TODO: Try this post to limit output range:
#  https://stackoverflow.com/questions/54435998/keras-limiting-output-min-max-value
def prepare_data(encodings):
    in_train = []
    out_train = []
    for i in range(len(select_items)):
        out_train.append([])

    for id, encoding in encodings.items():
        if id not in appearances:
            continue
        appearance_data = appearances[id]
        in_train.append(encoding)
        for i in range(len(select_items)):
            item_name = select_items[i]  # Item name
            item_idx = select_items_idxs[i]  # Item index in all appearance data
            item_data = appearance_data[item_idx]  # Appearance data value of item
            out_train[i].append(prepare_output(item_data, item_name))

    for i in range(len(select_items)):
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
    for i in range(len(select_items)):
        item_name = select_items[i]  # Item name
        activation_func = items_types[item_name]['activation']
        loss_type = items_types[item_name]['loss']
        item_val_range = items_val_range[item_name]
        output_dim = 1
        if loss_type == 'categorical_crossentropy':
            output_dim = abs(item_val_range[0] - item_val_range[1]) + 1
        elif loss_type == 'mse':
            output_dim = 1
        l = Dense(128, activation='relu')(Input_1)
        l = Dense(128, activation='relu')(l)
        # l = Dense(256, activation='relu')(l)
        out = Dense(output_dim, activation=activation_func, name=fix_layer_name(item_name))(l)
        outs.append(out)

    loss_funcs = {fix_layer_name(item_name): items_types[item_name]['loss'] for item_name in select_items}
    model = Model(inputs=Input_1, outputs=outs)
    model.compile(optimizer='rmsprop', loss=loss_funcs)
    return model


def plot_history(history):
    from matplotlib import pyplot as plt

    print(history.keys())
    # plt.plot(history['loss'])
    plt.show()


if __name__ == '__main__':
    encodings = load_json('encodings.json')

    in_train, out_train = prepare_data(encodings)
    in_valid, out_valid = [], []

    model = build_model()

    history = model.fit(in_train, out_train, epochs=100)
    model.save(model_save_path)
    # model.evaluate(in_train, out_train)
    plot_history(history.history)
