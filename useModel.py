from faceEncoder import FaceEncoder
from trainModel import custom_loss


def load_model(model_path):
    """
    Loads a model from a file
    """
    import tensorflow.keras as keras
    model = keras.models.load_model(model_path, custom_objects={'custom_loss': custom_loss})
    return model


face_encoder = FaceEncoder()

if __name__ == '__main__':
    model = load_model('model.h5')
    encoding = [face_encoder.encode_face('named/8639.jpg')]
    print(model.predict(encoding))
