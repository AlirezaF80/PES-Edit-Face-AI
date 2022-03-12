from typing import List, Dict

from extractNamesFromPics import extract_csv
from faceEncoder import FaceEncoder


def load_model(model_path):
    """
    Loads a model from a file
    """
    import tensorflow.keras as keras
    model = keras.models.load_model(model_path)
    return model


def load_json(filename):
    import json
    with open(filename) as f:
        return json.load(f)


csv_file = extract_csv('data-cleaned.csv')
appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
all_items_names: list[str] = [i.strip() for i in csv_file[0][4:]]
model_save_path = 'modelSigmoid.h5'
select_items: List[str] = load_json('items_names.json')
items_val_range: Dict[str, List[int]] = load_json('items_val_range.json')
items_types: Dict[str, Dict[str, str]] = load_json('items_types.json')

select_items_idxs: List[int] = [all_items_names.index(item) for item in select_items]
face_encoder = FaceEncoder()


def get_skin_color(race_pred):
    races_skin = {'white': 1, 'asian': 2, 'middle eastern': 3, 'latino hispanic': 3, 'indian': 4, 'black': 6}
    skin_color = 0
    for race, fac in race_pred['race'].items():
        skin_color += (fac / 100) * races_skin[race]
    return round(skin_color, 1)


def scale_from_sigmoid(val, val_range):
    min_val, max_val = val_range
    scaled_val = (val * (max_val - min_val)) + min_val
    return round(scaled_val, 1)


def categories_arr_to_val(arr, val_range):
    val = {}
    for i in range(val_range[0], val_range[1] + 1):
        fac = round(arr[i - val_range[0]], 1)
        if fac > 0:
            val[i] = fac
    return val


def clean_output(val, item_name):
    loss_type = items_types[item_name]['loss']
    if loss_type == 'categorical_crossentropy':
        return categories_arr_to_val(val, items_val_range[item_name])
    elif loss_type == 'mse':
        return scale_from_sigmoid(val[0], items_val_range[item_name])


if __name__ == '__main__':
    img_path = r"D:\Projects\Pycharm Projects\PES-Edit-Face-Maker\results\category-sigmoid hybrid model\sayadmanesh.jpg"
    encoding = [face_encoder.encode_face(img_path)]
    race_pred = face_encoder.analyze_face(img_path)
    model = load_model('modelSigmoid.h5')
    prediction = list(model.predict(encoding))

    print(img_path)
    print(f"DominantRace: {race_pred['dominant_race']}")
    print(f"RaceSkinCol: {get_skin_color(race_pred)}")
    for i in range(len(prediction)):
        raw_val = prediction[i][0]
        item_name = select_items[i]
        item_val_range = items_val_range[item_name]
        cleaned_val = clean_output(raw_val, item_name)
        print(f'{item_name}: {cleaned_val} [{item_val_range[0]} - {item_val_range[1]}]')
