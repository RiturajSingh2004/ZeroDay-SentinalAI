import time
import pandas as pd
import numpy as np
import logging
import threading
import json
from datetime import datetime
import queue
import tensorflow as tf

class AnomalyDetector:
    def __init__(self, data_collector, model_trainer, detection_interval=5, 
                 alert_threshold=2, consensus_threshold=0.5):
        self.data_collector = data_collector
        self.model_trainer = model_trainer
        self.detection_interval = detection_interval
        self.alert_threshold = alert_threshold  # How many consecutive anomalies before alerting
        self.consensus_threshold = consensus_threshold  # Proportion of models that must agree
        
        self.is_monitoring = False
        self.monitoring_thread = None
        self.alert_queue = queue.Queue()
        self.consecutive_anomalies = 0
        self.anomaly_scores = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("anomaly_detection.log"),
                logging.StreamHandler()
            ]
        )
    
    def detect_anomalies(self, metrics_data):
        """
        Use all trained models to detect anomalies in the current metrics
        Returns a dictionary with detection results and scores
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_results": {},
            "is_anomaly": False,
            "anomaly_score": 0.0,
            "anomaly_details": {}
        }
        
        try:
            # Ensure we have only the trained features
            metrics_df = pd.DataFrame([metrics_data])
            if 'timestamp' in metrics_df.columns:
                metrics_df = metrics_df.drop('timestamp', axis=1)
            
            # Handle missing columns
            for col in self.model_trainer.feature_columns:
                if col not in metrics_df.columns:
                    metrics_df[col] = 0
            
            # Keep only the columns used during training
            metrics_df = metrics_df[self.model_trainer.feature_columns]
            
            # Scale the data
            scaler = self.model_trainer.scalers['standard']
            scaled_data = scaler.transform(metrics_df)
            
            # Count anomaly detections
            anomaly_count = 0
            model_count = 0
            
            # Isolation Forest detection
            if 'isolation_forest' in self.model_trainer.models:
                model = self.model_trainer.models['isolation_forest']
                prediction = model.predict(scaled_data)
                # Isolation Forest returns -1 for anomalies, 1 for normal
                is_anomaly = prediction[0] == -1
                # Get decision function score (negative = more anomalous)
                score = -model.decision_function(scaled_data)[0]
                
                results["model_results"]["isolation_forest"] = {
                    "is_anomaly": bool(is_anomaly),
                    "score": float(score)
                }
                
                if is_anomaly:
                    anomaly_count += 1
                model_count += 1
            
            # Autoencoder detection
            if 'autoencoder' in self.model_trainer.models and 'autoencoder_threshold' in self.model_trainer.models:
                model = self.model_trainer.models['autoencoder']
                threshold = self.model_trainer.models['autoencoder_threshold']
                
                reconstructions = model.predict(scaled_data)
                mse = np.mean(np.power(scaled_data - reconstructions, 2), axis=1)
                is_anomaly = mse[0] > threshold
                score = float(mse[0] / threshold)  # Normalized score
                
                results["model_results"]["autoencoder"] = {
                    "is_anomaly": bool(is_anomaly),
                    "score": float(score)
                }
                
                if is_anomaly:
                    anomaly_count += 1
                model_count += 1
            
            # DBSCAN detection (for online detection, we check distance to nearest cluster)
            if 'dbscan' in self.model_trainer.models:
                model = self.model_trainer.models['dbscan']
                
                # Find distance to nearest cluster center
                cluster_centers = {}
                for label in set(model.labels_):
                    if label != -1:  # Skip noise points
                        cluster_centers[label] = np.mean(
                            self.model_trainer.scalers['standard'].inverse_transform(
                                model.components_[model.labels_ == label]
                            ), 
                            axis=0
                        )
                
                # If we have cluster centers, calculate distance
                if cluster_centers:
                    min_distance = float('inf')
                    for center in cluster_centers.values():
                        dist = np.linalg.norm(metrics_df.values[0] - center)
                        min_distance = min(min_distance, dist)
                    
                    # Determine if anomaly based on distance threshold
                    # This is a simple heuristic; you might want to calibrate this
                    distance_threshold = 3.0  # Adjust based on your data
                    is_anomaly = min_distance > distance_threshold
                    score = min_distance / distance_threshold
                    
                    results["model_results"]["dbscan"] = {
                        "is_anomaly": bool(is_anomaly),
                        "score": float(score)
                    }
                    
                    if is_anomaly:
                        anomaly_count += 1
                    model_count += 1
            
            # Calculate consensus
            if model_count > 0:
                consensus_ratio = anomaly_count / model_count
                results["is_anomaly"] = consensus_ratio >= self.consensus_threshold
                results["anomaly_score"] = consensus_ratio
            
            # If anomalous, add details about which metrics contributed
            if results["is_anomaly"]:
                # Use autoencoder reconstruction error to identify problematic metrics
                if 'autoencoder' in self.model_trainer.models:
                    original = scaled_data[0]
                    reconstruction = model.predict(scaled_data)[0]
                    
                    # Find features with highest reconstruction error
                    errors = np.power(original - reconstruction, 2)
                    feature_errors = list(zip(self.model_trainer.feature_columns, errors))
                    feature_errors.sort(key=lambda x: x[1], reverse=True)
                    
                    # Add top 5 anomalous metrics
                    for feature, error in feature_errors[:5]:
                        if error > 0.1:  # Only include significant errors
                            results["anomaly_details"][feature] = {
                                "error": float(error),
                                "value": float(metrics_df[feature].values[0]),
                                "expected": float(scaler.inverse_transform(
                                    reconstruction.reshape(1, -1)
                                )[0][self.model_trainer.feature_columns.index(feature)])
                            }
        
        except Exception as e:
            logging.error(f"Error in anomaly detection: {str(e)}")
            results["error"] = str(e)
        
        return results
    
    def monitoring_loop(self):
        """Main monitoring loop that runs in a separate thread"""
        logging.info("Starting real-time monitoring for zero-day attacks")
        
        while self.is_monitoring:
            try:
                # Collect current metrics
                current_metrics = self.data_collector.collect_system_metrics()
                
                # Detect anomalies
                results = self.detect_anomalies(current_metrics)
                
                # Update consecutive anomaly counter
                if results["is_anomaly"]:
                    self.consecutive_anomalies += 1
                    logging.warning(
                        f"Potential anomaly detected (Score: {results['anomaly_score']:.2f}, "
                        f"Consecutive: {self.consecutive_anomalies})"
                    )
                else:
                    self.consecutive_anomalies = 0
                
                # Check if we should trigger an alert
                if self.consecutive_anomalies >= self.alert_threshold:
                    alert = {
                        "timestamp": datetime.now().isoformat(),
                        "alert_type": "zero_day_attack",
                        "severity": "high" if results["anomaly_score"] > 0.8 else "medium",
                        "consecutive_detections": self.consecutive_anomalies,
                        "anomaly_score": results["anomaly_score"],
                        "details": results["anomaly_details"]
                    }
                    
                    # Add to alert queue
                    self.alert_queue.put(alert)
                    
                    # Log the alert
                    logging.error(
                        f"ALERT: Potential zero-day attack detected! "
                        f"Score: {results['anomaly_score']:.2f}, "
                        f"Consecutive detections: {self.consecutive_anomalies}"
                    )
                    
                    # Save detailed information about the alert
                    with open(f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 'w') as f:
                        json.dump(alert, f, indent=2)
                
                # Store the anomaly score history
                timestamp = datetime.now().isoformat()
                self.anomaly_scores[timestamp] = results["anomaly_score"]
                
                # Maintain only the most recent 1000 scores
                if len(self.anomaly_scores) > 1000:
                    oldest_key = next(iter(self.anomaly_scores))
                    self.anomaly_scores.pop(oldest_key)
                
                # Sleep before next detection
                time.sleep(self.detection_interval)
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.detection_interval)
        
        logging.info("Stopping real-time monitoring")
    
    def start_monitoring(self):
        """Start the monitoring process in a separate thread"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(target=self.monitoring_loop)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
            return True
        return False
    
    def stop_monitoring(self):
        """Stop the monitoring process"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        return True
    
    def get_anomaly_trend(self, window=60):
        """Get anomaly score trend for the specified window"""
        scores = list(self.anomaly_scores.values())[-window:]
        return scores if scores else []