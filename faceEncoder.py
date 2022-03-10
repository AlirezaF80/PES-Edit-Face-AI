import os

import deepface.commons.distance as dst
import numpy as np
from deepface import DeepFace
from deepface.commons.functions import load_image
from deepface.detectors import FaceDetector


class FaceEncoder:
    def __init__(self, detector_backend='dlib', model_name='Facenet', normalization='Facenet'):
        self._detector_backend = detector_backend
        self._model_name = model_name
        self._normalization = normalization
        self._is_models_built = False

    def _build_models(self):
        if self._is_models_built:
            return
        self._detector = FaceDetector.build_model(self._detector_backend)  # set opencv, ssd, dlib, mtcnn or retinaface
        self._analyze_models = {'age': DeepFace.build_model('Age')}
        self._model = DeepFace.build_model(self._model_name)
        self._is_models_built = True

    @staticmethod
    def get_distance(encoding1, encoding2, metric="euclidean_l2"):
        if encoding1.model_name != encoding2.model_name:
            return -1
        img1_repr = encoding1.encoding
        img2_repr = encoding2.encoding
        # metrics = ["cosine", "euclidean", "euclidean_l2"]
        if metric == 'cosine':
            distance = dst.findCosineDistance(img1_repr, img2_repr)
        elif metric == 'euclidean':
            distance = dst.findEuclideanDistance(img1_repr, img2_repr)
        elif metric == 'euclidean_l2':
            distance = dst.findEuclideanDistance(dst.l2_normalize(img1_repr), dst.l2_normalize(img2_repr))
        else:
            raise ValueError("Invalid distance_metric passed - ", metric)
        distance = np.float64(distance)
        return distance

    def detect_faces(self, img_path):
        self._build_models()

        img = load_image(img_path)

        obj = FaceDetector.detect_faces(self._detector, self._detector_backend, img)
        return len(obj)

    def encode_face(self, img_path):
        self._build_models()

        face_num = self.detect_faces(img_path)
        if face_num != 1:
            raise ValueError(f"FaceEncoder: There are {face_num} faces in the image: '{img_path}'")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"FaceEncoder: '{img_path}' doesn't exist.")
        embedding = DeepFace.represent(img_path=img_path, model=self._model, enforce_detection=False,
                                       detector_backend=self._detector_backend, normalization=self._normalization)
        return embedding

    def analyze_face(self, img_path):
        self._build_models()

        analyzation = DeepFace.analyze(img_path, actions=['age'], models=self._analyze_models, enforce_detection=False,
                                       detector_backend=self._detector_backend, prog_bar=False)
        return analyzation

    @property
    def model_name(self):
        return self._model_name
