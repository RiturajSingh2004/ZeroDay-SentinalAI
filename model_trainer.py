from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import Sequential, Model
from keras import Dense, Input, Dropout
import numpy as np
import pandas as pd
import pickle
import logging
import joblib

class AIModelTrainer:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("model_training.log"),
                logging.StreamHandler()
            ]
        )
    
    def preprocess_data(self, data):
        """Clean and prepare data for model training"""
        # Drop non-numeric columns or timestamp
        if 'timestamp' in data.columns:
            data = data.drop('timestamp', axis=1)
        
        # Handle missing values
        data = data.fillna(0)
        
        # Store feature columns
        self.feature_columns = data.columns.tolist()
        
        # Scale the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        self.scalers['standard'] = scaler
        
        return scaled_data, data
    
    def train_isolation_forest(self, data, contamination=0.01):
        """Train an Isolation Forest model for anomaly detection"""
        try:
            scaled_data, _ = self.preprocess_data(data)
            logging.info("Training Isolation Forest model...")
            
            # Initialize and train the model
            isolation_forest = IsolationForest(
                n_estimators=100,
                contamination=contamination,
                random_state=42,
                n_jobs=-1
            )
            isolation_forest.fit(scaled_data)
            
            # Store the model
            self.models['isolation_forest'] = isolation_forest
            logging.info("Isolation Forest model trained successfully")
            
            return True
        except Exception as e:
            logging.error(f"Error training Isolation Forest model: {str(e)}")
            return False
    
    def train_autoencoder(self, data, epochs=50, batch_size=32, validation_split=0.2):
        """Train an Autoencoder neural network for anomaly detection"""
        try:
            scaled_data, raw_data = self.preprocess_data(data)
            logging.info("Training Autoencoder model...")
            
            # Split data for training and validation
            X_train, X_val = train_test_split(scaled_data, test_size=validation_split, random_state=42)
            
            # Define model architecture
            input_dim = scaled_data.shape[1]
            encoding_dim = int(input_dim / 2)
            
            # Build the encoder
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim, activation='relu')(input_layer)
            encoder = Dropout(0.2)(encoder)
            encoder = Dense(int(encoding_dim/2), activation='relu')(encoder)
            
            # Build the decoder
            decoder = Dense(encoding_dim, activation='relu')(encoder)
            decoder = Dropout(0.2)(decoder)
            decoder = Dense(input_dim, activation='sigmoid')(decoder)
            
            # The autoencoder model
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            
            # Compile the model
            autoencoder.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train the model
            history = autoencoder.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(X_val, X_val),
                verbose=1
            )
            
            # Store the model
            self.models['autoencoder'] = autoencoder
            
            # Calculate reconstruction error threshold (used for anomaly detection)
            reconstructions = autoencoder.predict(scaled_data)
            mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
            threshold = np.percentile(mse, 95)  # 95th percentile as threshold
            self.models['autoencoder_threshold'] = threshold
            
            logging.info(f"Autoencoder model trained successfully. Threshold: {threshold:.6f}")
            return True
            
        except Exception as e:
            logging.error(f"Error training Autoencoder model: {str(e)}")
            return False
    
    def train_dbscan(self, data, eps=0.5, min_samples=5):
        """Train a DBSCAN clustering model for anomaly detection"""
        try:
            scaled_data, _ = self.preprocess_data(data)
            logging.info("Training DBSCAN model...")
            
            # Initialize and train the model
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            dbscan.fit(scaled_data)
            
            # Store the model
            self.models['dbscan'] = dbscan
            
            # Count outliers (-1 label)
            n_outliers = np.sum(dbscan.labels_ == -1)
            logging.info(f"DBSCAN model trained successfully. Found {n_outliers} outliers in training data.")
            
            return True
        except Exception as e:
            logging.error(f"Error training DBSCAN model: {str(e)}")
            return False
    
    def save_models(self, base_filename="model"):
        """Save all trained models and scalers to disk"""
        try:
            # Save ML models
            for name, model in self.models.items():
                if name == 'autoencoder':
                    model.save(f"{base_filename}_{name}.h5")
                elif name == 'autoencoder_threshold':
                    with open(f"{base_filename}_{name}.pkl", 'wb') as f:
                        pickle.dump(model, f)
                else:
                    joblib.dump(model, f"{base_filename}_{name}.joblib")
            
            # Save scalers
            for name, scaler in self.scalers.items():
                joblib.dump(scaler, f"{base_filename}_scaler_{name}.joblib")
            
            # Save feature columns
            with open(f"{base_filename}_features.pkl", 'wb') as f:
                pickle.dump(self.feature_columns, f)
                
            logging.info(f"All models and scalers saved with base filename: {base_filename}")
            return True
        except Exception as e:
            logging.error(f"Error saving models: {str(e)}")
            return False
    
    def load_models(self, base_filename="model"):
        """Load all trained models and scalers from disk"""
        try:
            # Load feature columns
            with open(f"{base_filename}_features.pkl", 'rb') as f:
                self.feature_columns = pickle.load(f)
            
            # Load ML models
            self.models['isolation_forest'] = joblib.load(f"{base_filename}_isolation_forest.joblib")
            self.models['dbscan'] = joblib.load(f"{base_filename}_dbscan.joblib")
            
            # Load autoencoder model
            self.models['autoencoder'] = keras.models.load_model(f"{base_filename}_autoencoder.h5")
            
            # Load autoencoder threshold
            with open(f"{base_filename}_autoencoder_threshold.pkl", 'rb') as f:
                self.models['autoencoder_threshold'] = pickle.load(f)
            
            # Load scalers
            self.scalers['standard'] = joblib.load(f"{base_filename}_scaler_standard.joblib")
            
            logging.info(f"All models and scalers loaded from base filename: {base_filename}")
            return True
        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False