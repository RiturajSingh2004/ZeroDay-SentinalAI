import logging
import time
import subprocess
import json
import os
import threading
import requests
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans

class ResponseEngine:
    def __init__(self, anomaly_detector, mitigation_config="mitigation_rules.json"):
        self.anomaly_detector = anomaly_detector
        self.mitigation_config = mitigation_config
        self.mitigation_rules = self.load_mitigation_rules()
        self.is_responding = False
        self.response_thread = None
        self.incident_history = []
        self.current_incidents = {}
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler("response_engine.log"),
                logging.StreamHandler()
            ]
        )
    
    def load_mitigation_rules(self):
        """Load mitigation rules from configuration file"""
        try:
            if os.path.exists(self.mitigation_config):
                with open(self.mitigation_config, 'r') as f:
                    return json.load(f)
            else:
                # Default rules if file doesn't exist
                default_rules = {
                    "rules": [
                        {
                            "name": "CPU Overload",
                            "conditions": {
                                "cpu_percent": ">80"
                            },
                            "actions": [
                                {
                                    "type": "command",
                                    "command": "ps -eo pid,ppid,cmd,%mem,%cpu --sort=-%cpu | head -n 10"
                                },
                                {
                                    "type": "alert",
                                    "message": "High CPU usage detected"
                                }
                            ]
                        },
                        {
                            "name": "Network Anomaly",
                            "conditions": {
                                "network_sent": ">10000000",
                                "network_recv": ">10000000"
                            },
                            "actions": [
                                {
                                    "type": "command",
                                    "command": "netstat -tunap | grep ESTABLISHED | wc -l"
                                },
                                {
                                    "type": "alert",
                                    "message": "Unusual network activity detected"
                                }
                            ]
                        }
                    ]
                }
                
                # Save default rules
                with open(self.mitigation_config, 'w') as f:
                    json.dump(default_rules, f, indent=2)
                
                return default_rules
                
        except Exception as e:
            logging.error(f"Error loading mitigation rules: {str(e)}")
            return {"rules": []}
    
    def evaluate_condition(self, condition, value):
        """Evaluate a condition against a value"""
        if isinstance(condition, str):
            if condition.startswith('>'):
                threshold = float(condition[1:])
                return value > threshold
            elif condition.startswith('<'):
                threshold = float(condition[1:])
                return value < threshold
            elif condition.startswith('=='):
                threshold = float(condition[2:])
                return value == threshold
            else:
                return False
        return False
    
    def execute_action(self, action, context):
        """Execute a mitigation action"""
        try:
            action_type = action.get("type", "")
            
            if action_type == "command":
                command = action.get("command", "")
                if command:
                    logging.info(f"Executing command: {command}")
                    result = subprocess.run(command, shell=True, capture_output=True, text=True)
                    logging.info(f"Command output: {result.stdout}")
                    return {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                
            elif action_type == "alert":
                message = action.get("message", "")
                severity = action.get("severity", "medium")
                
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "message": message,
                    "severity": severity,
                    "context": context
                }
                
                logging.warning(f"ALERT: {message} (Severity: {severity})")
                
                # You could add code here to send the alert via email, SMS, etc.
                return {
                    "success": True,
                    "alert": alert
                }
                
            elif action_type == "process":
                pid = action.get("pid", "")
                action_name = action.get("action", "")
                
                if pid and action_name == "kill":
                    logging.warning(f"Killing process with PID {pid}")
                    result = subprocess.run(f"kill -9 {pid}", shell=True, capture_output=True, text=True)
                    return {
                        "success": result.returncode == 0,
                        "output": result.stdout,
                        "error": result.stderr
                    }
                
            elif action_type == "network":
                ip = action.get("ip", "")
                action_name = action.get("action", "")
                
                if ip and action_name == "block":
                    logging.warning(f"Blocking IP address {ip}")
                    # This would typically use iptables or other firewall tools
                    # For demo purposes, we'll just log it
                    return {
                        "success": True,
                        "message": f"Blocked IP address {ip} (simulated)"
                    }
                
            return {"success": False, "error": "Unknown action or missing parameters"}
            
        except Exception as e:
            logging.error(f"Error executing action: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def apply_mitigation(self, alert):
        """Apply appropriate mitigation based on the alert and rules"""
        incident_id = f"incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create incident record
        incident = {
            "id": incident_id,
            "timestamp": datetime.now().isoformat(),
            "alert": alert,
            "status": "in_progress",
            "mitigation_actions": [],
            "resolution": None
        }
        
        self.current_incidents[incident_id] = incident
        
        try:
            # Check each rule
            for rule in self.mitigation_rules.get("rules", []):
                rule_matched = True
                
                # Check if conditions match
                for metric, condition in rule.get("conditions", {}).items():
                    if metric in alert.get("details", {}):
                        metric_value = alert["details"][metric].get("value", 0)
                        if not self.evaluate_condition(condition, metric_value):
                            rule_matched = False
                            break
                    else:
                        rule_matched = False
                        break
                
                # If rule matches, execute actions
                if rule_matched:
                    logging.info(f"Rule '{rule.get('name')}' matched for incident {incident_id}")
                    
                    # Execute each action in the rule
                    for action in rule.get("actions", []):
                        action_result = self.execute_action(action, {
                            "incident_id": incident_id,
                            "alert": alert
                        })
                        
                        # Record action result
                        incident["mitigation_actions"].append({
                            "timestamp": datetime.now().isoformat(),
                            "action": action,
                            "result": action_result
                        })
            
                # If no rules matched, apply default response
                if not incident["mitigation_actions"]:
                    logging.info(f"No specific rules matched for incident {incident_id}, applying default response")
                    
                    # Default action: collect diagnostic information
                    default_action = {
                        "type": "command",
                        "command": "ps aux | sort -nrk 3,3 | head -n 10"
                    }
                    
                    action_result = self.execute_action(default_action, {                    
                        "incident_id": incident_id,
                        "alert": alert
                        })
                    incident["mitigation_actions"].append({
                    "timestamp": datetime.now().isoformat(),
                    "action": default_action,
                    "result": action_result
                })
            
            # Update incident status
            incident["status"] = "mitigated"
            
            # Add to history and remove from current incidents
            self.incident_history.append(incident)
            self.current_incidents.pop(incident_id, None)
            
            # Save incident record
            with open(f"incident_{incident_id}.json", 'w') as f:
                json.dump(incident, f, indent=2)
            
            logging.info(f"Incident {incident_id} has been mitigated")
            return incident
            
        except Exception as e:
            logging.error(f"Error applying mitigation for incident {incident_id}: {str(e)}")
            incident["status"] = "error"
            incident["resolution"] = str(e)
            return incident

    def response_loop(self):
        """Main response loop that runs in a separate thread"""
        logging.info("Starting automated response engine")
        
        while self.is_responding:
            try:
                # Check for new alerts in the queue
                if not self.anomaly_detector.alert_queue.empty():
                    alert = self.anomaly_detector.alert_queue.get()
                    logging.info(f"Processing new alert: {alert['alert_type']} (Severity: {alert['severity']})")
                    
                    # Apply mitigation
                    incident = self.apply_mitigation(alert)
                    
                    # Mark alert as processed
                    self.anomaly_detector.alert_queue.task_done()
                
                # Sleep before checking again
                time.sleep(1)
                
            except Exception as e:
                logging.error(f"Error in response loop: {str(e)}")
                time.sleep(5)
        
        logging.info("Stopping automated response engine")

    def start_responding(self):
        """Start the response engine in a separate thread"""
        if not self.is_responding:
            self.is_responding = True
            self.response_thread = threading.Thread(target=self.response_loop)
            self.response_thread.daemon = True
            self.response_thread.start()
            return True
        return False

    def stop_responding(self):
        """Stop the response engine"""
        self.is_responding = False
        if self.response_thread:
            self.response_thread.join(timeout=10)
        return True

    def cluster_incidents(self, n_clusters=3):
        """
        Use AI to cluster similar incidents for pattern recognition
        Returns cluster assignments for each incident
        """
        if len(self.incident_history) < n_clusters:
            return []
        
        # Extract features from incidents
        incidents_text = []
        for incident in self.incident_history:
            # Convert incident data to text representation
            text = []
            text.append(f"severity:{incident['alert'].get('severity', '')}")
            
            for detail_key, detail_value in incident['alert'].get('details', {}).items():
                if isinstance(detail_value, dict) and 'value' in detail_value:
                    text.append(f"{detail_key}:{detail_value['value']}")
            
            incidents_text.append(' '.join(text))
        
        # Vectorize text data
        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(incidents_text)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(incidents_text)), random_state=42)
        clusters = kmeans.fit_predict(X)
        
        # Return cluster assignments
        return list(clusters)

    def get_incident_summary(self):
        """Generate a summary of incidents and their patterns"""
        total_incidents = len(self.incident_history)
        current_incidents = len(self.current_incidents)
        
        if total_incidents == 0:
            return {"total": 0, "current": 0, "summary": "No incidents recorded"}
        
        # Get incident clusters
        clusters = self.cluster_incidents() if total_incidents >= 3 else []
        
        # Calculate incident statistics
        severity_counts = {}
        for incident in self.incident_history:
            severity = incident['alert'].get('severity', 'unknown')
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Identify common patterns in incidents
        common_details = {}
        for incident in self.incident_history:
            for detail_key in incident['alert'].get('details', {}):
                common_details[detail_key] = common_details.get(detail_key, 0) + 1
        
        # Sort by frequency
        common_details = sorted(common_details.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "total": total_incidents,
            "current": current_incidents,
            "by_severity": severity_counts,
            "common_indicators": common_details[:5],
            "clusters": len(set(clusters)) if clusters else 0
        }
                