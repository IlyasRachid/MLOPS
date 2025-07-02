import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
from sentence_transformers import SentenceTransformer
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

class PredictionPipeline:
    def __init__(self, model_path: str, embedder_path: str, genre_encoder_path: str, target_encoder_path: str):
        self.model = load_model(model_path)
        logging.info(f"Model loaded from {model_path}")
        with open(embedder_path, 'r') as f:
            self.embedder_model = SentenceTransformer(f.read().strip())
        self.genre_encoder = load_object(genre_encoder_path)
        self.target_encoder = load_object(target_encoder_path)

    def predict(self, genre_str: str, description: str):
        try:
            # Preprocess genre
            genre_list = genre_str.split(',')
            genre_encoded = self.genre_encoder.transform([genre_list]) 

            # Preprocess description
            desc_embedding = self.embedder_model.encode(description, convert_to_numpy=True)
            desc_embedding = desc_embedding.reshape(1, -1)

            # Combine features
            combined_features = np.hstack([desc_embedding, genre_encoded])

            # Make prediction
            prediction = self.model.predict(combined_features)
            pred_label = np.argmax(prediction, axis=1)

            # Decode the predicted label
            predicted_movie = self.target_encoder.inverse_transform(pred_label)

            return predicted_movie[0]
        
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":

    pipeline = PredictionPipeline(
        model_path=os.path.join('artifacts', 'models', 'model_20250702_203433.h5'),
        embedder_path=os.path.join('artifacts', 'models', 'embedder_model_20250702_203431.txt'),
        genre_encoder_path=os.path.join('artifacts', 'preprocessors', 'genre_encoder_20250702_203431.pkl'),
        target_encoder_path=os.path.join('artifacts', 'preprocessors', 'target_encoder_20250702_203431.pkl')
    )

    genre= "Drama, Crime"
    description = """Imprisoned in the 1940s for the double murder of his wife and her lover,
     upstanding banker Andy Dufresne begins a new life at the Shawshank prison, where he puts his accounting
       skills to work for an amoral warden. During his long stretch in prison, Dufresne comes to be admired by
         the other inmates -- including an older prisoner named Red -- for his integrity and unquenchable sense of hope."""
    
    prediction = pipeline.predict(genre, description)
    print(f"Predicted Movie: {prediction}")


