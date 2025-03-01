import psutil
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import threading
import logging

class DataCollector:
    def __init__(self, collection_interval=5, training_duration=3600):
        self.collection_interval = collection_interval  # seconds
        self.training_duration = training_duration  # seconds
        self.metrics_df = pd.DataFrame()
        self.is_collecting = False
        self.collection_thread = None
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("data_collection.log"),
                logging.StreamHandler()
            ]
        )
    
    def collect_system_metrics(self):
        """Collect various system metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_io_read': psutil.disk_io_counters().read_bytes,
            'disk_io_write': psutil.disk_io_counters().write_bytes,
            'network_sent': psutil.net_io_counters().bytes_sent,
            'network_recv': psutil.net_io_counters().bytes_recv,
            'connection_count': len(psutil.net_connections()),
            'process_count': len(psutil.pids())
        }
        
        # Get per-process metrics for top processes
        process_metrics = {}
        for proc in sorted(psutil.process_iter(['pid', 'name', 'cpu_percent']), 
                           key=lambda p: p.info['cpu_percent'] or 0, 
                           reverse=True)[:10]:
            try:
                process_metrics[f"proc_{proc.info['pid']}_cpu"] = proc.info['cpu_percent']
                process_metrics[f"proc_{proc.info['pid']}_mem"] = proc.memory_percent()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        metrics.update(process_metrics)
        return metrics
    
    def collection_loop(self):
        """Main collection loop that runs in a separate thread"""
        start_time = time.time()
        collected_count = 0
        
        logging.info("Starting data collection for baseline creation")
        
        while self.is_collecting and (time.time() - start_time < self.training_duration):
            try:
                metrics = self.collect_system_metrics()
                self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics])], ignore_index=True)
                collected_count += 1
                
                if collected_count % 10 == 0:
                    logging.info(f"Collected {collected_count} data points. Time elapsed: {time.time() - start_time:.1f}s")
                
                time.sleep(self.collection_interval)
            except Exception as e:
                logging.error(f"Error collecting data: {str(e)}")
        
        self.is_collecting = False
        logging.info(f"Data collection completed. Total samples: {len(self.metrics_df)}")
    
    def start_collection(self):
        """Start the data collection process in a separate thread"""
        if not self.is_collecting:
            self.is_collecting = True
            self.collection_start_time = time.time()
            self.collection_thread = threading.Thread(target=self.collection_loop)
            self.collection_thread.daemon = True
            self.collection_thread.start()
            return True
        return False
    
    def stop_collection(self):
        """Stop the data collection process"""
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=10)
        return True
    
    def save_data(self, filename="system_metrics.csv"):
        """Save collected data to a CSV file"""
        if not self.metrics_df.empty:
            self.metrics_df.to_csv(filename, index=False)
            logging.info(f"Data saved to {filename}")
            return True
        logging.warning("No data to save")
        return False