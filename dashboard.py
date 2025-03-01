# Web Dashboard Component
import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import sys

# Initialize the Dash app
app = dash.Dash(__name__, title="Zero-Day Attack Mitigation Dashboard")

# Load framework components
# These would be initialized in a production system
# For the demo, we'll simulate some data
def load_mock_data():
    # Simulate anomaly scores over time
    now = datetime.now()
    times = [(now - timedelta(minutes=i)).isoformat() for i in range(120, 0, -1)]
    
    # Generate some realistic looking anomaly scores
    base_scores = np.random.normal(0.2, 0.05, len(times))
    
    # Add some anomalous periods
    for i in range(70, 80):
        base_scores[i] = np.random.normal(0.7, 0.1)
    
    for i in range(100, 105):
        base_scores[i] = np.random.normal(0.9, 0.05)
    
    anomaly_scores = {t: max(0, min(1, s)) for t, s in zip(times, base_scores)}
    
    # Generate some mock incidents
    incidents = []
    if os.path.exists("mock_incidents.json"):
        with open("mock_incidents.json", "r") as f:
            incidents = json.load(f)
    else:
        # Create a few mock incidents
        incident_times = [times[75], times[102]]
        for i, t in enumerate(incident_times):
            severity = "high" if i == 1 else "medium"
            incidents.append({
                "id": f"incident_{i}",
                "timestamp": t,
                "alert": {
                    "alert_type": "zero_day_attack",
                    "severity": severity,
                    "anomaly_score": 0.9 if severity == "high" else 0.7,
                    "details": {
                        "cpu_percent": {"value": 92.5 if severity == "high" else 78.3},
                        "network_sent": {"value": 15000000 if severity == "high" else 8000000}
                    }
                },
                "status": "mitigated",
                "mitigation_actions": []
            })
        
        with open("mock_incidents.json", "w") as f:
            json.dump(incidents, f)
    
    return anomaly_scores, incidents

# Load mock data for the demo
anomaly_scores, incidents = load_mock_data()

# Define the dashboard layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("AI-Powered Zero-Day Attack Mitigation Framework"),
        html.H3("Real-time Attack Detection and Response Dashboard"),
    ], className='header'),
    
    # System Status
    html.Div([
        html.Div([
            html.H4("System Status"),
            html.Div(id="system-status", children=[
                html.Span("MONITORING", className="status-active"),
                html.Span(" | Models: ", style={"margin-left": "10px"}),
                html.Span("Isolation Forest, Autoencoder, DBSCAN", className="status-info")
            ])
        ], className='status-card'),
        
        html.Div([
            html.H4("Alert Summary"),
            html.Div(id="alert-summary", children=[
                html.Div([
                    html.Span("Total Incidents: "),
                    html.Span(f"{len(incidents)}", className="alert-number")
                ]),
                html.Div([
                    html.Span("Current Anomaly Score: "),
                    html.Span(f"{list(anomaly_scores.values())[-1]:.2f}", 
                              className="alert-score", 
                              id="current-score")
                ]),
                html.Div([
                    html.Span("Status: "),
                    html.Span("Normal", className="status-normal", id="current-status")
                ])
            ])
        ], className='status-card')
    ], className='status-row'),
    
    # Anomaly Score Chart
    html.Div([
        html.H4("Real-time Anomaly Detection"),
        dcc.Graph(id="anomaly-chart"),
        dcc.Interval(
            id='interval-component',
            interval=5*1000,  # in milliseconds
            n_intervals=0
        )
    ], className='chart-card'),
    
    # Recent Incidents
    html.Div([
        html.H4("Recent Security Incidents"),
        html.Div(id="incidents-table")
    ], className='incidents-card'),
    
    # Model Performance
    html.Div([
        html.H4("AI Model Performance"),
        html.Div([
            dcc.Graph(id="model-performance-chart")
        ])
    ], className='chart-card'),
    
    # Footer
    html.Div([
        html.P("AI-Powered Zero-Day Attack Mitigation Framework - Demo Version"),
        html.P("Update Frequency: 5 seconds")
    ], className='footer')
], className='dashboard-container')

# Callback to update the anomaly chart
@app.callback(
    [Output("anomaly-chart", "figure"),
     Output("current-score", "children"),
     Output("current-status", "children"),
     Output("current-status", "className"),
     Output("incidents-table", "children"),
     Output("model-performance-chart", "figure")],
    [Input("interval-component", "n_intervals")]
)
def update_metrics(n):
    # In a real system, this would pull live data
    # For the demo, we'll use our mock data
    
    # Prepare time series data
    times = list(anomaly_scores.keys())
    scores = list(anomaly_scores.values())
    
    # Create the anomaly chart
    df = pd.DataFrame({
        'time': pd.to_datetime(times),
        'score': scores
    })
    
    # Set color thresholds
    df['color'] = 'green'
    df.loc[df['score'] > 0.5, 'color'] = 'orange'
    df.loc[df['score'] > 0.7, 'color'] = 'red'
    
    # Create anomaly chart
    fig = px.line(df, x='time', y='score', 
                 title="Anomaly Detection Score (Real-time)",
                 labels={'time': 'Time', 'score': 'Anomaly Score'})
    
    fig.update_traces(line=dict(color='blue', width=2))
    
    # Add threshold lines
    fig.add_shape(type="line", x0=df['time'].min(), x1=df['time'].max(),
                 y0=0.7, y1=0.7, line=dict(color="red", width=2, dash="dash"))
    
    fig.add_shape(type="line", x0=df['time'].min(), x1=df['time'].max(),
                 y0=0.5, y1=0.5, line=dict(color="orange", width=2, dash="dash"))
    
    # Add colored points
    for color in df['color'].unique():
        mask = df['color'] == color
        fig.add_trace(go.Scatter(
            x=df[mask]['time'],
            y=df[mask]['score'],
            mode='markers',
            marker=dict(color=color, size=8),
            name=f"{color.capitalize()} Level",
            showlegend=True
        ))
    
    # Highlight incident points
    incident_times = [pd.to_datetime(inc['timestamp']) for inc in incidents]
    incident_scores = [inc['alert']['anomaly_score'] for inc in incidents]
    
    if incident_times:
        fig.add_trace(go.Scatter(
            x=incident_times,
            y=incident_scores,
            mode='markers',
            marker=dict(color='black', size=12, symbol='x'),
            name="Incidents",
            showlegend=True
        ))
    
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest"
    )
    
    # Current score and status
    current_score = f"{scores[-1]:.2f}"
    
    if scores[-1] > 0.7:
        current_status = "Critical"
        status_class = "status-critical"
    elif scores[-1] > 0.5:
        current_status = "Warning"
        status_class = "status-warning"
    else:
        current_status = "Normal"
        status_class = "status-normal"
    
    # Create incidents table
    incidents_sorted = sorted(incidents, key=lambda x: x['timestamp'], reverse=True)
    
    if incidents_sorted:
        incidents_table = html.Table([
            html.Thead(html.Tr([
                html.Th("Time"),
                html.Th("Severity"),
                html.Th("Anomaly Score"),
                html.Th("Status"),
                html.Th("Details")
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(datetime.fromisoformat(inc['timestamp']).strftime('%Y-%m-%d %H:%M:%S')),
                    html.Td(inc['alert']['severity'].upper(), className=f"severity-{inc['alert']['severity']}"),
                    html.Td(f"{inc['alert']['anomaly_score']:.2f}"),
                    html.Td(inc['status'].upper()),
                    html.Td(", ".join([f"{k}: {v['value']}" for k, v in inc['alert']['details'].items()]))
                ]) for inc in incidents_sorted[:5]
            ])
        ], className="incidents-table")
    else:
        incidents_table = html.Div("No incidents recorded", className="no-incidents")
    
    # Create model performance chart
    # In a real system, this would show actual model performance metrics
    model_names = ['Isolation Forest', 'Autoencoder', 'DBSCAN', 'Ensemble']
    precision = [0.92, 0.89, 0.85, 0.94]
    recall = [0.88, 0.91, 0.82, 0.93]
    f1_scores = [0.90, 0.90, 0.83, 0.93]
    
    model_df = pd.DataFrame({
        'Model': model_names * 3,
        'Metric': ['Precision'] * 4 + ['Recall'] * 4 + ['F1-Score'] * 4,
        'Value': precision + recall + f1_scores
    })
    
    model_fig = px.bar(model_df, x='Model', y='Value', color='Metric', barmode='group',
                  title="AI Model Performance Metrics",
                  labels={'Value': 'Score', 'Model': 'Model Name'})
    
    model_fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10)
    )
    
    return fig, current_score, current_status, status_class, incidents_table, model_fig

# Define CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Zero-Day Attack Mitigation Dashboard</title>
        {%metas%}
        {%favicon%}
        {%css%}
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background-color: #f5f7fa;
                margin: 0;
                padding: 20px;
                color: #333;
            }
            
            .dashboard-container {
                max-width: 1200px;
                margin: 0 auto;
            }
            
            .header {
                text-align: center;
                margin-bottom: 20px;
            }
            
            .header h1 {
                color: #2c3e50;
                margin-bottom: 5px;
            }
            
            .header h3 {
                color: #7f8c8d;
                font-weight: normal;
                margin-top: 0;
            }
            
            .status-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
            }
            
            .status-card {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                width: 48%;
            }
            
            .status-card h4 {
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 1px solid #ecf0f1;
                padding-bottom: 10px;
            }
            
            .chart-card, .incidents-card {
                background-color: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                margin-bottom: 20px;
            }
            
            .chart-card h4, .incidents-card h4 {
                margin-top: 0;
                color: #2c3e50;
                border-bottom: 1px solid #ecf0f1;
                padding-bottom: 10px;
            }
            
            .footer {
                text-align: center;
                margin-top: 30px;
                color: #7f8c8d;
                font-size: 12px;
            }
            
            .status-active {
                color: #27ae60;
                font-weight: bold;
            }
            
            .status-inactive {
                color: #e74c3c;
                font-weight: bold;
            }
            
            .status-info {
                color: #3498db;
            }
            
            .alert-number {
                font-weight: bold;
                color: #3498db;
            }
            
            .alert-score {
                font-weight: bold;
            }
            
            .status-normal {
                color: #27ae60;
                font-weight: bold;
            }
            
            .status-warning {
                color: #f39c12;
                font-weight: bold;
            }
            
            .status-critical {
                color: #e74c3c;
                font-weight: bold;
            }
            
            .incidents-table {
                width: 100%;
                border-collapse: collapse;
            }
            
            .incidents-table th, .incidents-table td {
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ecf0f1;
            }
            
            .incidents-table th {
                background-color: #f8f9fa;
                font-weight: 600;
            }
            
            .severity-high {
                color: #e74c3c;
                font-weight: bold;
            }
            
            .severity-medium {
                color: #f39c12;
                font-weight: bold;
            }
            
            .severity-low {
                color: #3498db;
                font-weight: bold;
            }
            
            .no-incidents {
                padding: 20px;
                text-align: center;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Main application
def main():
    """Main function to start the framework"""
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    try:
        from data_collector import DataCollector
        from model_trainer import AIModelTrainer
        from anomaly_detector import AnomalyDetector
        from response_engine import ResponseEngine
    except ImportError as e:
        print(f"Error importing modules: {e}")
        print("Make sure all required modules are in the same directory")
        return

    # Initialize components
    data_collector = DataCollector()
    model_trainer = AIModelTrainer()
    anomaly_detector = AnomalyDetector(data_collector, model_trainer)
    response_engine = ResponseEngine(anomaly_detector)
    
    # Start the dashboard
    app.run_server(debug=True, port=8050)

if __name__ == "__main__":
    main()