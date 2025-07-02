import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
import numpy as np
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from dataclasses import dataclass

from src.logger import logging
from src.exception import CustomException

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'models', f'model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.keras')

class ModelTrainer:
    def __init__(self, data_path: str):
        self.config = ModelTrainerConfig()
        self.data_path = data_path

    def load_data(self):
        try:
            data = np.load(self.data_path)
            return data['X_train'], data['y_train'], data['X_test'], data['y_test']
        except Exception as e:
            raise CustomException(e, sys)
    
    def build_model(self, input_dim: int, output_dim: int):
        try:
            model = Sequential([
                Dense(256, activation='relu', input_shape=(input_dim,)),
                Dropout(0.3),
                Dense(128, activation='relu'),
                Dropout(0.3),
                Dense(output_dim, activation='softmax' if output_dim > 1 else 'sigmoid')
            ])
            model.compile(optimizer="adam",
                          loss='sparse_categorical_crossentropy' if output_dim > 1 else 'binary_crossentropy',
                          metrics=['accuracy'])
            return model
        except Exception as e:
            raise CustomException(e, sys)
        
    def train_and_save_model(self):
        try:
            X_train, y_train, X_test, y_test = self.load_data()
            input_dim = X_train.shape[1]
            output_dim = len(np.unique(y_train))

            model = self.build_model(input_dim, output_dim)

            logging.info("Starting model training...")
            early_stopping = EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
            model_checkpoint = ModelCheckpoint(self.config.model_path, save_best_only=True, monitor='val_accuracy')

            history = model.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                epochs=50,
                                batch_size=32,
                                callbacks=[early_stopping, model_checkpoint],
                                verbose=1
                            )

            logging.info(f"Model training completed successfully and saved to {self.config.model_path}")

            # evaluate the model
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            logging.info(f"Model evaluation completed. Test loss: {test_loss}, Test accuracy: {test_accuracy}")

            return (
                self.config.model_path
            )
        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__ == "__main__":
#     trainer = ModelTrainer(data_path='artifacts/transformed_data/transformed_data_20250702_053021.npz')
#     trainer.train_and_save_model()

