import json
import pathlib


class EmbeddingGenerator:
    FACE_EMBEDDING_FORMAT = '.faceEmbedding'

    @staticmethod
    def _get_embedding_save_path(img_path):
        img_path_obj = pathlib.Path(img_path)
        img_name = img_path_obj.name
        embedding_name = img_name + EmbeddingGenerator.FACE_EMBEDDING_FORMAT
        return img_path_obj.parent.joinpath(embedding_name)

    @staticmethod
    def get_embedding(img_path, save_embedding=True):
        save_path = EmbeddingGenerator._get_embedding_save_path(img_path)
        if pathlib.Path(save_path).exists():
            return EmbeddingGenerator._load_embedding(save_path)
        else:
            embedding = EmbeddingGenerator._generate_embedding(str(img_path))
            if save_embedding and embedding:
                EmbeddingGenerator._save_embedding(embedding, save_path)
            return embedding

    @staticmethod
    def _generate_embedding(img_path: str):
        try:
            img_repr = DeepFaceWrapper.get_representation(img_path)
            return img_repr
        except Exception as e:
            raise ValueError(f"EmbeddingGenerator: generating embedding didn't work for: '{img_path}' - {e}")

    @staticmethod
    def _load_embedding(embedding_path):
        with open(embedding_path, 'r') as file:
            return json.load(file)

    @staticmethod
    def _save_embedding(embedding, save_path):
        if not pathlib.Path(save_path).parent.exists():
            raise ValueError(f"EmbeddingGenerator: '{save_path}' doesn't exist for saving embedding.")
        with open(save_path, 'w') as file:
            json.dump(embedding, file)
