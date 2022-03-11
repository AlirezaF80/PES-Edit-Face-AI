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


def scale_from_sigmoid(val, min_val, max_val):
    return val * (max_val - min_val) + min_val


values_range = load_json('values_range.json')

face_encoder = FaceEncoder()
csv_file = extract_csv('data-cleaned.csv')
# appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
appearance_items = csv_file[0][4:]

if __name__ == '__main__':
    encoding = [
        face_encoder.encode_face(r"D:\Projects\Pycharm Projects\PES-Edit-Face-Maker\results\sigmoid model\yamga.jpg")]
    model = load_model('modelSigmoid.h5')
    prediction = list(model.predict(encoding))
    for i in range(len(prediction)):
        raw_val = prediction[i][0][0]
        scaled_val = scale_from_sigmoid(raw_val, values_range[appearance_items[i]][0],
                                        values_range[appearance_items[i]][1])
        scaled_val = round(scaled_val, 1)
        print(appearance_items[i].strip(), scaled_val,
              f'{values_range[appearance_items[i]][0]} : {values_range[appearance_items[i]][1]}')
