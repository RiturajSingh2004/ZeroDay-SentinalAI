import logging
import argparse
import time
import json
import os
import pandas as pd

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("zero_day_framework.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("zero_day_framework")

def main():
    """Main function to run the Zero-Day Attack Mitigation Framework"""
    logger = setup_logging()
    
    parser = argparse.ArgumentParser(description="AI-Powered Zero-Day Attack Mitigation Framework")
    parser.add_argument('--train', action='store_true', help='Train AI models on baseline data')
    parser.add_argument('--collect', action='store_true', help='Collect baseline data')
    parser.add_argument('--monitor', action='store_true', help='Start monitoring and detection')
    parser.add_argument('--respond', action='store_true', help='Enable automated response')
    parser.add_argument('--dashboard', action='store_true', help='Launch the web dashboard')
    parser.add_argument('--all', action='store_true', help='Enable all components')
    args = parser.parse_args()
    
    # If no arguments are provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    logger.info("Starting AI-Powered Zero-Day Attack Mitigation Framework")
    
    # Import components
    from data_collector import DataCollector
    from model_trainer import AIModelTrainer
    from anomaly_detector import AnomalyDetector
    from response_engine import ResponseEngine
    
    # Initialize components
    data_collector = DataCollector()
    model_trainer = AIModelTrainer()
    
    # Collect baseline data if requested
    if args.collect or args.all:
        logger.info("Starting baseline data collection")
        data_collector.start_collection()
        
        try:
            # Collect data for the specified training duration
            while data_collector.is_collecting:
                time.sleep(5)
                
            logger.info(f"Collected {len(data_collector.metrics_df)} data points for baseline")
            data_collector.save_data("baseline_data.csv")
            
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
            data_collector.stop_collection()
            data_collector.save_data("baseline_data.csv")
    
    # Train AI models if requested
    if args.train or args.all:
        if os.path.exists("baseline_data.csv"):
            logger.info("Loading baseline data for training")
            data = pd.read_csv("baseline_data.csv")
            
            logger.info(f"Training AI models on {len(data)} data points")
            model_trainer.train_isolation_forest(data)
            model_trainer.train_autoencoder(data)
            model_trainer.train_dbscan(data)
            
            logger.info("Saving trained models")
            model_trainer.save_models()
        else:
            logger.error("No baseline data found. Run with --collect first.")
            return
    
    # Initialize detection and response components
    anomaly_detector = AnomalyDetector(data_collector, model_trainer)
    response_engine = ResponseEngine(anomaly_detector)
    
    # Start monitoring if requested
    if args.monitor or args.all:
        # Load models if they exist
        if os.path.exists("model_isolation_forest.joblib"):
            logger.info("Loading trained models")
            model_trainer.load_models()
            
            logger.info("Starting real-time monitoring")
            anomaly_detector.start_monitoring()
        else:
            logger.error("No trained models found. Run with --train first.")
            return
    
    # Start automated response if requested
    if args.respond or args.all:
        if anomaly_detector.is_monitoring:
            logger.info("Starting automated response engine")
            response_engine.start_responding()
        else:
            logger.error("Monitoring is not active. Run with --monitor first.")
            return
    
    # Launch the dashboard if requested
    if args.dashboard or args.all:
        logger.info("Starting web dashboard")
        # Import and run the dashboard
        from dashboard import app
        app.run_server(debug=False, host='0.0.0.0', port=8050)
    
    # Keep the main thread running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down framework")
        
        # Cleanup
        if data_collector.is_collecting:
            data_collector.stop_collection()
            
        if anomaly_detector.is_monitoring:
            anomaly_detector.stop_monitoring()
            
        if response_engine.is_responding:
            response_engine.stop_responding()
        
        logger.info("Framework shutdown complete")

if __name__ == "__main__":
    main()