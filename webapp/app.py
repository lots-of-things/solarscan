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
from dash.dependencies import Input, Output
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
import requests
import threading
from dashboard import dash_app_layout, generate_dash_outputs

load_dotenv()  # Load environment variables from .env

app = Flask(__name__)
csrf = CSRFProtect(app)

# Load configuration values
app.config['SECRET_KEY'] = 'sillygoosesecretkey'
app.config['VERSION'] = os.getenv('VERSION')
app.config['ENV'] = os.getenv('ENV')
app.config['MAPS_API_KEY'] = os.getenv('MAPS_API_KEY')
app.config['IPINFO_TOKEN'] = os.getenv('IPINFO_TOKEN')
app.config['BACKEND_TYPE'] = os.getenv('BACKEND_TYPE')

csrf._exempt_views.add('dash.dash.dispatch')

if app.config['ENV'] == 'production':
    logging.basicConfig(level=logging.ERROR)
else:
    logging.basicConfig(level=logging.DEBUG)

project_id = "739182013691"
location = "us-central1"


if app.config['ENV'] == 'production':
    model_names = {
        "blurring",
        "spectral",
        "augmix",
        "standard",
        "oracle",
    }
    client = datastore.Client()
    storage_client = storage.Client()
else:
    model_names = {
        "blurring",
        "spectral",
    }
    import time
    from dev_data import fake_feedback_data, fake_metamodel_data, fake_history_data

if app.config['BACKEND_TYPE'] == 'vertex':
    
    aiplatform.init(project=project_id, location=location)
    all_endpoints = aiplatform.Endpoint.list()

    # Iterate through the endpoints and their deployed models
    endpoints = {}
    for endpoint in all_endpoints:
        # List deployed models for this endpoint
        for deployed_model in endpoint.list_models():
            # Match the model name with the deployed model
            model_name = aiplatform.Model(deployed_model.model).display_name
            endpoints[model_name] = endpoint

    
    def call_backend(base64_image_string):
        # Format the image data as a list of instances (assuming it's a single image)
        instances = [{"data": base64_image_string}]
        
        probabilities = {}
        for name, endpoint in endpoints.items():
            response = endpoint.predict(instances=instances)
            
            # Assuming response contains the prediction results in `predictions`
            probabilities[name] = response.predictions[0]
        return probabilities
elif app.config['BACKEND_TYPE'] == 'app':
    if app.config['ENV'] == 'production':
        BACKEND_URL = "http://backend-dot-solarscan.appspot.com"
    else:
        BACKEND_URL = "http://127.0.0.1:5050"
    def call_backend(base64_image_string):
        probabilities = {}
        for model_name in model_names:
            request_data = {'model_name': model_name,
                    'base64_image_string': base64_image_string}
            response = requests.post(f'{BACKEND_URL}/warmup', json=request_data)
            print(response)
            backend_data = response.json()
            probabilities[model_name]=backend_data
        return probabilities
    
def warmup_backend():
    # Send the GET request to the /warmup endpoint
    requests.get(f'{BACKEND_URL}/warmup')

@app.route('/')
def index():
    if app.config['BACKEND_TYPE'] == 'app':
        thread = threading.Thread(target=warmup_backend)
        thread.daemon = True  # Make the thread a daemon thread so it exits when the main program exits
        thread.start()
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
    # Send prediction request to the AI platform
    probabilities = call_backend(base64_image_string)
    
    if app.config['ENV'] == 'production':
        metamodel_dict = load_optimal_params()
    else:
        time.sleep(2)
        # image_data = base64.b64decode(base64_image_string)
        # file_path = "output_image.jpg"
        # with open(file_path, "wb") as file:
        #     file.write(image_data)
        metamodel_dict = fake_metamodel_data
    
    metamodel_params = dict(zip(metamodel_dict['features'], metamodel_dict['param_values']))
    feature_names = list(probabilities.keys())
    decision_function_sum = metamodel_params['intercept']
    for key in feature_names:
        decision_function_sum+=metamodel_params[key]*probabilities[key]
    metamodel_probability = sigmoid(decision_function_sum)

    if app.config['ENV'] == 'production':
        prediction_entity = datastore.Entity(client.key('detection'))  # Create a Datastore entity
        prediction_entity.update({
            'timestamp': datetime.now().isoformat(),
            'north': request_json['north'],
            'south': request_json['south'],
            'east': request_json['east'],
            'west': request_json['west'],
            'probabilities': probabilities,
            'metamodel_probability': metamodel_probability
        })
        try:
            client.put(prediction_entity)
        except:
            pass
    
    # Adding the predictions to the response
    request_json['probabilities'] = probabilities
    request_json['metamodel_probability'] = metamodel_probability
    # Adding the predictions to the response
    request_json['probabilities'] = probabilities
    return jsonify(request_json), 200


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
    feedback_results = []
    model_probs = {}

    thresholds = np.linspace(0, 1, 21)  # Generate thresholds from 0 to 1 in steps of 0.05
    
    for entity in results:
        feedback = entity['feedback']  # Assuming 'feedback' is a property in your Datastore
        probabilities = entity['probabilities']  # Assuming 'probabilities' is a dictionary
        feedback_results.append(feedback)
        for model, prob in probabilities.items():
            if model not in model_metrics:
                model_probs[model] = []
                model_metrics[model] = {'TPs': np.zeros(len(thresholds)),
                                          'TNs': np.zeros(len(thresholds)),
                                          'Ps': np.zeros(len(thresholds)),
                                          'Ns': np.zeros(len(thresholds)),
                                          }
            model_probs[model].append(prob)
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

    return thresholds, model_metrics, feedback_results, model_probs

def find_optimal_params(feedback_results, model_probs):
    # Create numpy arrays for features (probabilities) and labels (feedback)
    features = np.array([v for _, v in model_probs.items()]).T
    labels = np.array(feedback_results)

    model = LogisticRegression()
    model.fit(features, labels)

    # Get the coefficients and intercept after fitting the model on the whole dataset
    feature_names = list(model_probs.keys())
    feature_names.append('intercept')
    
    params = model.coef_[0].tolist()
    params.append(model.intercept_[0])

    # Perform cross-validation for accuracy and F1 score
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Get cross-validation scores for accuracy and F1 score
    accuracy_scores = cross_val_score(model, features, labels, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, features, labels, cv=cv, scoring='f1')
    balanced_accuracy_scores = cross_val_score(model, features, labels, cv=cv, scoring='balanced_accuracy')

    # Calculate the mean accuracy and mean F1 score across the folds
    accuracy = np.mean(accuracy_scores)
    f1 = np.mean(f1_scores)
    balanced_accuracy = np.mean(balanced_accuracy_scores)

    result_data = {'features': feature_names, 
            'param_values': params, 
            'accuracy':accuracy,
            'f1_score':f1,
            'balanced_accuracy':balanced_accuracy,
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
dash_app.layout = dash_app_layout

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
        time.sleep(2)
        results = fake_feedback_data

    thresholds, model_metrics, feedback_results, model_probs = get_model_metrics(results)

    # Get the optimal parameters and accuracy using your get_optimal_params() function
    if n_clicks > 0:
        model_performance = find_optimal_params(feedback_results, model_probs)  # This returns a dictionary with keys: 'features', 'params', 'accuracy'
    else:
        model_performance = load_optimal_params()

    # Metrics for performance table
    performance_data = [
        {'metric': 'Evaluation Date', 'value': model_performance['checkpoint_time'][:10]},
    ]
    metrics_to_show = {'Accuracy': 'accuracy', 'Balanced Accuracy': 'balanced_accuracy', 'F1 Score': 'f1_score' }

    for metric_name, metric_key in metrics_to_show.items():
        if metric_key in model_performance:
            performance_data.append({'metric': metric_name, 'value': f"{model_performance[metric_key]:0.3f}"})


    if app.config['ENV'] == 'production':
        query = client.query(kind='metamodel')
        query.order = ['checkpoint_time'] 
        history_data = []

        # Loop through the results to extract the required fields
        for result in query.fetch():
            # Extract checkpoint_time, accuracy, and f1_score
            checkpoint_time = result.get('checkpoint_time')

            # Append to history data
            if checkpoint_time:
                next_record = {
                    'date': checkpoint_time,  # Format as string (YYYY-MM-DD)
                }
                for _, metric_key in metrics_to_show.items():
                    if metric_key in result:
                        next_record[metric_key] = result[metric_key]
                history_data.append(next_record) 
    else:
        history_data = fake_history_data

    return generate_dash_outputs(thresholds, model_metrics, model_performance, performance_data, history_data, metrics_to_show)
    

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
    app.run(debug=app.config['DEBUG'], port=5000)
