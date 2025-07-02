from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import os
import sys
from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sentence_transformers import SentenceTransformer
from datetime import datetime
from src.utils import save_object, handle_missing_values

@dataclass
class DataTransformationConfig:
    genre_encoder_path: str = os.path.join('artifacts', 'preprocessors', f'genre_encoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    target_encoder_path: str = os.path.join('artifacts', 'preprocessors', f'target_encoder_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    embedder_model_path: str = os.path.join('artifacts', 'models', f'embedder_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    transformed_data_path: str = os.path.join('artifacts', 'transformed_data', f'transformed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.npz')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.genre_encoder = MultiLabelBinarizer()
        self.target_encoder = LabelEncoder()
        self.embedder_model = SentenceTransformer('all-MiniLM-L6-v2')

    def save_embedder_info(self):
        try:
            metadata = "all-MiniLM-L6-v2"
            with open(self.data_transformation_config.embedder_model_path, 'w') as file:
                file.write(metadata)

            logging.info(f"Embedder model metadata saved at {self.data_transformation_config.embedder_model_path}")
            
        except Exception as e:
            raise CustomException("Error saving embedder model metadata", sys)


    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("Starting data transformation process")
            # Load the datasets
            train_df = handle_missing_values(pd.read_csv(train_path), target_col='movie_name', text_cols=['description', 'genre'])
            test_df = handle_missing_values(pd.read_csv(test_path), target_col='movie_name', text_cols=['description', 'genre'])

            logging.info("Splitting input features and target variable")
            y_train = self.target_encoder.fit_transform(train_df['movie_name'])
            seen_labels = set(self.target_encoder.classes_)
            test_df = test_df[test_df['movie_name'].isin(seen_labels)]
            y_test = self.target_encoder.transform(test_df['movie_name'])

            X_train = train_df.drop(columns=['movie_name'])
            X_test = test_df.drop(columns=['movie_name'])

            logging.info("Encoding genre column")
            genre_train = self.genre_encoder.fit_transform(X_train['genre'].apply(lambda x: x.split(',')))
            genre_test = self.genre_encoder.transform(X_test['genre'].apply(lambda x: x.split(',')))

            logging.info("Generating embeddings for description column")
            desc_train = self.embedder_model.encode(X_train['description'].tolist(), convert_to_numpy=True)
            desc_test = self.embedder_model.encode(X_test['description'].tolist(), convert_to_numpy=True)

            logging.info("Combining features into a single stacked array")
            X_train_final = np.hstack([desc_train, genre_train])
            X_test_final = np.hstack([desc_test, genre_test])

            logging.info("Saving the transformed data")
            np.savez_compressed(
                self.data_transformation_config.transformed_data_path,
                X_train=X_train_final,
                y_train=y_train,
                X_test=X_test_final,
                y_test=y_test
            )
            logging.info(f"Transformed data saved at {self.data_transformation_config.transformed_data_path}")

            logging.info("Saving encoders and embedder model")
            save_object(self.data_transformation_config.genre_encoder_path, self.genre_encoder)
            save_object(self.data_transformation_config.target_encoder_path, self.target_encoder)
            
            self.save_embedder_info()
            logging.info("Data transformation process completed successfully")
            return (
                self.data_transformation_config.transformed_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__=="__main__":
#     data_transformation = DataTransformation()
#     data_transformation.initiate_data_transformation(
#         train_path='data/train/training_data.csv-2025-07-01-06-17-21',
#         test_path='data/test/test_data.csv-2025-07-01-06-17-21'
#     )
