from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
import logging
from flask_wtf.csrf import CSRFProtect
from google.cloud import aiplatform, datastore, storage
from uuid import uuid4
import base64
import dash
from dash import dash_table
from dash import dcc, html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


load_dotenv()  # Load environment variables from .env

app = Flask(__name__)
csrf = CSRFProtect(app)

# Load configuration values
app.config['SECRET_KEY'] = 'sillygoosesecretkey'
app.config['VERSION'] = os.getenv('VERSION')
app.config['ENV'] = os.getenv('ENV')
app.config['MAPS_API_KEY'] = os.getenv('MAPS_API_KEY')
app.config['IPINFO_TOKEN'] = os.getenv('IPINFO_TOKEN')

csrf._exempt_views.add('dash.dash.dispatch')

if app.config['ENV'] == 'production':
    logging.basicConfig(level=logging.ERROR)
else:
    logging.basicConfig(level=logging.DEBUG)

project_id = "739182013691"
location = "us-central1"
endpoint_ids = {
    "blurring":"2109928728841682944",
    "spectral":"8519676898496741376",
    "augmix":"7424176289138868224",
    "standard":"1254244799641288704",
    "oracle":"7018852322675523584",
    
}

if app.config['ENV'] == 'production':
    aiplatform.init(project=project_id, location=location)
    endpoints = {k: aiplatform.Endpoint(v) for k, v in endpoint_ids.items()}

    client = datastore.Client()
    storage_client = storage.Client()
else:
    from dev_data import fake_feedback_data, fake_metamodel_data, fake_probabilities_data

@app.route('/')
def index():
    return render_template('index.html', maps_api_key=app.config['MAPS_API_KEY'], ipinfo_token=app.config['IPINFO_TOKEN'])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

@app.route('/process-image', methods=['POST'])
def process_image():
    # Get the uploaded image from the request
    request_json = request.get_json()
    base64_image_string = request_json.get('image')
    
    # Check and remove prefix if present (e.g., data:image/png;base64,)
    if base64_image_string.startswith("data:image"):
        base64_image_string = base64_image_string.split(",")[1]
    
    # Send image data to the Vertex AI endpoint (AI Platform)
    try:
        # Format the image data as a list of instances (assuming it's a single image)
        if app.config['ENV'] == 'production':
            instances = [{"data": base64_image_string}]
        
            # Send prediction request to the AI platform
            probabilities = {}
            for name, endpoint in endpoints.items():
                response = endpoint.predict(instances=instances)
                
                # Assuming response contains the prediction results in `predictions`
                probabilities[name] = response.predictions[0]
            metamodel_dict = load_optimal_params()
        else:
            probabilities = fake_probabilities_data
            metamodel_dict = fake_metamodel_data
        
        metamodel_params = dict(zip(metamodel_dict['features'], metamodel_dict['param_values']))
        feature_names = list(probabilities.keys())
        decision_function_sum = metamodel_params['intercept']
        for key in feature_names:
            decision_function_sum+=metamodel_params[key]*probabilities[key]
        metamodel_probability = sigmoid(decision_function_sum)
        
        # Adding the predictions to the response
        request_json['probabilities'] = probabilities
        request_json['metamodel_probability'] = metamodel_probability
        # Adding the predictions to the response
        request_json['probabilities'] = probabilities
        return jsonify(request_json), 200
    
    except Exception as e:
        # Handle errors
        return jsonify({"error": str(e)}), 500


@app.route('/store-feedback', methods=['POST'])
def store_feedback():
    if app.config['ENV'] != 'production':
        return jsonify({"msg": "skipping"}), 200
    request_json = request.get_json()

    # Check if 'image' key is in the request
    if 'image' in request_json:
        base64_image_string = request_json['image']
        image_name = f"feedback_image_{uuid4().hex}.png"  # Generate a unique image name

        # Upload the image to Cloud Storage
        bucket_name = 'solarscan-images'  # Replace with your bucket name
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(image_name)

        # Convert the base64 string to binary data
        image_data = base64.b64decode(base64_image_string)
        
        # Upload image data to Cloud Storage
        blob.upload_from_string(image_data, content_type='image/png')

        # Get the URL or path to the image in Cloud Storage
        image_url = blob.public_url  # Public URL or use a signed URL for private access

        # Remove the 'image' key from the original request_json to avoid storing large data
        del request_json['image']
        
        # Prepare the feedback to be stored in Datastore
        feedback_entity = datastore.Entity(client.key('feedback'))  # Create a Datastore entity
        feedback_entity.update({
            'timestamp': datetime.now().isoformat(),
            'image_url': image_url  # Store the Cloud Storage URL for the image
        })
        
        feedback_entity.update(request_json)

        # Save the entity to Datastore
        try:
            client.put(feedback_entity)
            return jsonify({"message": "Feedback stored successfully!"}), 200
        except Exception as e:
            return jsonify({"error": str(e), 'request': request.get_json()}), 500
    
    else:
        return jsonify({"error": "Image key is required."}), 400

def get_model_metrics(results):
    model_metrics = {}

    thresholds = np.linspace(0, 1, 21)  # Generate thresholds from 0 to 1 in steps of 0.05
    
    for entity in results:
        feedback = entity['feedback']  # Assuming 'feedback' is a property in your Datastore
        probabilities = entity['probabilities']  # Assuming 'probabilities' is a dictionary
        
        for model, prob in probabilities.items():
            if model not in model_metrics:
                model_metrics[model] = {'TPs': np.zeros(len(thresholds)),
                                          'TNs': np.zeros(len(thresholds)),
                                          'Ps': np.zeros(len(thresholds)),
                                          'Ns': np.zeros(len(thresholds)),
                                          }
            # Iterate over thresholds to calculate TP, TN rates for each
            for i, threshold in enumerate(thresholds):
                if feedback == 1 and prob >= threshold:
                    # True Positive (TP)
                    model_metrics[model]["TPs"][i]+=1
                    model_metrics[model]["Ps"][i]+=1
                elif feedback == 0 and prob < threshold:
                    # True Negative (TN)
                    model_metrics[model]["TNs"][i]+=1
                    model_metrics[model]["Ns"][i]+=1
                elif feedback == 1 and prob < threshold:
                    # False Negative (FN)
                    model_metrics[model]["Ps"][i]+=1
                elif feedback == 0 and prob >= threshold:
                    # False Positive (FP)
                    model_metrics[model]["Ns"][i]+=1

    # Now, calculate the TP and TN rates by averaging over the thresholds
    for model, metrics in model_metrics.items():
        model_metrics[model]["TP_rate"] = metrics["TPs"]/metrics["Ps"]
        model_metrics[model]["TN_rate"] = metrics["TNs"]/metrics["Ns"]

    return thresholds, model_metrics

def find_optimal_params(results):
    # Create numpy arrays for features (probabilities) and labels (feedback)
    feature_names = list(results[0]['probabilities'].keys())
    features = np.array([list(result['probabilities'].values()) for result in results])
    labels = np.array([result['feedback'] for result in results])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

    # Initialize and train the logistic regression model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    params = model.coef_[0]  # This will be a 1D array of coefficients
    params = model.coef_[0].tolist()
    params.append(model.intercept_[0])
    feature_names.append('intercept')


    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test,y_pred)

    result_data = {'features': feature_names, 
            'param_values': params, 
            'accuracy':accuracy,
            'f1_score':f1,
            'checkpoint_time':datetime.now().isoformat(),}
    
    if app.config['ENV'] == 'production':
        metamodel_entity = datastore.Entity(client.key('metamodel'))  # Create a Datastore entity
        metamodel_entity.update(result_data)
        client.put(metamodel_entity)

    return result_data

def load_optimal_params():
    if app.config['ENV'] != 'production':
        return fake_metamodel_data
    query = client.query(kind='metamodel')

    # Order by the 'timestamp' field in descending order (newest first)
    query.order = ['-checkpoint_time']  # Use '-' to reverse the order (descending)

    # Fetch the entities
    results = list(query.fetch(limit=1))  # Fetch only the most recent entity

    if results:
        # Return the properties of the first (most recent) entity as a dictionary
        return dict(results[0])
    else:
        return {}

dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')

# Dash layout with improved structure
dash_app.layout = html.Div(
    style={'fontFamily': 'Arial, sans-serif', 'padding': '30px'},  # Global styling
    children=[
        html.Div(
            children=[
                html.H1("Model Performance Dashboard", style={'textAlign': 'center', 'color': '#333', 'marginBottom': '30px'}),
                html.P(
                    "This dashboard displays the performance of machine learning models based on their True Positive and True Negative rates. "
                    "The graphs below show how these rates change across different thresholds, helping to evaluate model accuracy at various levels.",
                    style={'textAlign': 'center', 'fontSize': '16px', 'color': '#666', 'marginBottom': '30px'}
                ),
            ]
        ),
        
        html.Div(
            children=[
                # Two columns for the charts (using Flexbox for layout)
                html.Div(
                    children=[dcc.Graph(id='tp-chart')],
                    style={'flex': '1', 'padding': '10px', 'border': '2px solid #D0D0D0', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)'}
                ),
                html.Div(
                    children=[dcc.Graph(id='tn-chart')],
                    style={'flex': '1', 'padding': '10px', 'border': '2px solid #D0D0D0', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)'}
                ),
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}
        ),
        
        # Add a table to display logistic regression parameters and accuracy
        html.Div(
            children=[
                # Performance Metrics Table
                html.Div(
                    children=[
                        html.H2("Optimal Model Performance Metrics", style={'textAlign': 'center', 'color': '#333', 'marginTop': '50px'}),
                        dash_table.DataTable(
                            id='performance-metrics-table',
                            columns=[
                                {'name': 'Metric', 'id': 'metric'},
                                {'name': 'Value', 'id': 'value'},
                            ],
                            style_table={'width': '60%', 'margin': '30px auto'},
                            style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'},
                            style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '14px'},
                            style_data={'backgroundColor': '#f9f9f9'},
                        ),
                    ],
                    style={'flex': '1', 'padding': '10px', 'border': '2px solid #D0D0D0', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)'}
                ),
                # Parameters Table
                html.Div(
                    children=[
                        html.H2("Logistic Regression Parameters", style={'textAlign': 'center', 'color': '#333', 'marginTop': '50px'}),
                        dash_table.DataTable(
                            id='params-table',
                            columns=[
                                {'name': 'Feature', 'id': 'feature'},
                                {'name': 'Value', 'id': 'param_value'},
                            ],
                            style_table={'width': '60%', 'margin': '30px auto'},
                            style_header={'backgroundColor': '#f4f4f4', 'fontWeight': 'bold'},
                            style_cell={'padding': '10px', 'textAlign': 'center', 'fontSize': '14px'},
                            style_data={'backgroundColor': '#f9f9f9'},
                        ),
                    ],
                    style={'flex': '1', 'padding': '10px', 'border': '2px solid #D0D0D0', 'borderRadius': '10px', 'boxShadow': '0 2px 5px rgba(0, 0, 0, 0.1)'}
                ),
            ],
            style={'display': 'flex', 'justifyContent': 'space-between', 'flexWrap': 'wrap'}
        ),

        # Button to trigger the rerun of finding optimal parameters
        html.Div(
            children=[
                html.Button(
                    "Rerun Finding Optimal Params", 
                    id='rerun-btn', 
                    n_clicks=0, 
                    style={'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#4CAF50', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
                ),
            ],
            style={'textAlign': 'center', 'marginTop': '20px'}
        ),

        # Toggle button for history graph
        html.Div(
            children=[
                html.Button(
                    "Toggle Accuracy & F1 Score History", 
                    id='toggle-history-btn', 
                    n_clicks=0, 
                    style={'padding': '10px 20px', 'fontSize': '16px', 'backgroundColor': '#FF9800', 'color': 'white', 'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer'}
                ),
            ],
            style={'textAlign': 'center', 'marginTop': '20px'}
        ),
        
        # History of Accuracy and F1 Score graph (Initially hidden)
        html.Div(
            children=[
                dcc.Graph(id='history-chart'),
            ],
            id='history-container',  # Container for history chart
            style={'display': 'none', 'marginTop': '30px'},  # Hidden initially
        ),
    ]
)

# Dash callback to update dashboard
@dash_app.callback(
    [Output('tp-chart', 'figure'),
     Output('tn-chart', 'figure'),
     Output('performance-metrics-table', 'data'),
     Output('params-table', 'data'),
     Output('history-chart', 'figure')],
    [Input('rerun-btn', 'n_clicks')]
)
def update_dashboard(n_clicks):
    # Ensure that the callback is triggered when the button is clicked
    if app.config['ENV'] == 'production':
        query = client.query(kind='feedback')
        results = list(query.fetch())  # Fetch all results
    else:
        results = fake_feedback_data

    thresholds, model_metrics = get_model_metrics(results)

    # Get the optimal parameters and accuracy using your get_optimal_params() function
    if n_clicks > 0:
        model_performance = find_optimal_params(results)  # This returns a dictionary with keys: 'features', 'params', 'accuracy'
    else:
        model_performance = load_optimal_params()

    # Metrics for performance table
    performance_data = [
        {'metric': 'Checkpoint Time', 'value': model_performance['checkpoint_time']},
        {'metric': 'Accuracy', 'value': f"{model_performance['accuracy']:0.3f}"},
        {'metric': 'F1 Score', 'value': f"{model_performance['f1_score']:0.3f}"},
    ]

    # Parameters for parameter table
    params_data = [{'feature': feature, 'param_value': f"{value:0.3f}"} for feature, value in zip(model_performance['features'], model_performance['param_values'])]
    
    # Create figure for True Positive Rate vs Thresholds
    bold_colors = ['#FF6F61', '#6B5B95', '#88B04B', '#F7B7A3', '#F6D02F']
    tp_rate_fig = {
        'data': [
            go.Scatter(
                x=thresholds,
                y=model_metrics[model]["TP_rate"],
                mode='lines+markers',
                name=f'{model}',
                line={'width': 3, 'color': bold_colors[i % len(bold_colors)]},
                marker={'size': 8, 'color': bold_colors[i % len(bold_colors)]}
            )
            for i, model in enumerate(model_metrics)
        ],
        'layout': go.Layout(
            title='True Positive Rate vs Thresholds',
            title_x=0.5,
            xaxis={'title': 'Threshold', 'tickangle': 45},
            yaxis={'title': 'True Positive Rate'},
            template='plotly_white',
            plot_bgcolor='#f7f7f7',
            paper_bgcolor='#FFFFFF',
            font={'color': '#333'},
            showlegend=True,
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
        )
    }

    # Create figure for True Negative Rate vs Thresholds
    tn_rate_fig = {
        'data': [
            go.Scatter(
                x=thresholds,
                y=model_metrics[model]["TN_rate"],
                mode='lines+markers',
                name=f'{model}',
                line={'width': 3, 'color': bold_colors[i % len(bold_colors)]},
                marker={'size': 8, 'color': bold_colors[i % len(bold_colors)]}
            )
            for i, model in enumerate(model_metrics)
        ],
        'layout': go.Layout(
            title='True Negative Rate vs Thresholds',
            title_x=0.5,
            xaxis={'title': 'Threshold', 'tickangle': 45},
            yaxis={'title': 'True Negative Rate'},
            template='plotly_white',
            plot_bgcolor='#f7f7f7',
            paper_bgcolor='#FFFFFF',
            font={'color': '#333'},
            showlegend=True,
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
        )
    }

    if app.config['ENV'] == 'production':
        query = client.query(kind='metamodel')
        query.order = ['checkpoint_time'] 
        history_data = []

        # Loop through the results to extract the required fields
        for result in query.fetch():
            # Extract checkpoint_time, accuracy, and f1_score
            checkpoint_time = result.get('checkpoint_time')
            accuracy = result.get('accuracy')
            f1_score = result.get('f1_score')
            
            # Append to history data
            if checkpoint_time:
                history_data.append({
                    'date': checkpoint_time,  # Format as string (YYYY-MM-DD)
                    'accuracy': accuracy,
                    'f1_score': f1_score
                })
    else:
        history_data = [
            {'date':fake_metamodel_data['checkpoint_time'],
             'accuracy':fake_metamodel_data['accuracy'],
             'f1_score':fake_metamodel_data['f1_score'],
             } for i in range(3)]
    
    # Make the history fig
    history_fig = {
        'data': [
            go.Scatter(
                x=[data['date'] for data in history_data],
                y=[data['accuracy'] for data in history_data],
                mode='lines+markers',
                name='Accuracy',
                line={'color': '#4CAF50'},
            ),
            go.Scatter(
                x=[data['date'] for data in history_data],
                y=[data['f1_score'] for data in history_data],
                mode='lines+markers',
                name='F1 Score',
                line={'color': '#FF9800'},
            ),
        ],
        'layout': go.Layout(
            title='Metamodel Accuracy and F1 Score History',
            title_x=0.5,
            xaxis={'title': 'Date', 'tickangle': 45},
            yaxis={'title': 'Score'},
            template='plotly_white',
            plot_bgcolor='#f7f7f7',
            paper_bgcolor='#FFFFFF',
            font={'color': '#333'},
            showlegend=True,
            margin={'t': 50, 'b': 50, 'l': 50, 'r': 50},
        )
    }

    # Return the figures and the table data
    return tp_rate_fig, tn_rate_fig, performance_data, params_data, history_fig

@dash_app.callback(
    Output('history-container', 'style'),
    Input('toggle-history-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_history_graph(n_clicks):
    if n_clicks % 2 == 1:  # Toggle on
        return {'display': 'block'}
    else:  # Toggle off
        return {'display': 'none'}

if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], host='0.0.0.0', port=5000)
