import glob
import json

from tqdm import tqdm

from faceEncoder import FaceEncoder

face_encoder = FaceEncoder()


def import_encodings(file_path):
    # import encodings from json
    with open(file_path, 'r') as f:
        encodings = json.load(f)
    return encodings


def export_encodings(encodings, file_path):
    with open(file_path, 'w') as f:
        json.dump(encodings, f)


def generate_encodings(images_glob):
    encodings = {}
    for img_path in tqdm(glob.glob(images_glob)):
        player_id = img_path.split('\\')[-1][:-4]
        encodings[player_id] = face_encoder.encode_face(img_path)
    return encodings


if __name__ == '__main__':
    images_dir = r'named/*.jpg'
    print("Generating Encodings:")
    encodings = generate_encodings(images_dir)
    print("Exporting Encodings:")
    export_encodings(encodings, 'encodings.json')
