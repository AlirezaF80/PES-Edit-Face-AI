from extractNamesFromPics import extract_csv
from faceEncoder import FaceEncoder


def load_model(model_path):
    """
    Loads a model from a file
    """
    import tensorflow.keras as keras
    model = keras.models.load_model(model_path)
    return model


face_encoder = FaceEncoder()
csv_file = extract_csv('data-cleaned.csv')
# appearances = {line[0]: [int(i) for i in line[4:]] for line in csv_file[1:]}
appearance_items = csv_file[0][4:]

if __name__ == '__main__':
    model = load_model('modelNew.h5')
    encoding = [face_encoder.encode_face(r"D:\Projects\Pycharm Projects\PES-Edit-Face-Maker\results\hardani.jpg")]
    prediction = list(model.predict(encoding))
    for i in range(len(prediction)):
        print(appearance_items[i].strip(), round(float(prediction[i])))
