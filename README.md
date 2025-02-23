# PV detection model webapp

Code for [solarscan.uw.r.appspot.com](https://solarscan.uw.r.appspot.com), a map app that detects whether there's PV panels at a given location based on google's sattelite imagery.  Built on top of [gabrielkasmi's](https://github.com/gabrielkasmi) models trained in [robust_pv_mapping](https://github.com/gabrielkasmi/robust_pv_mapping) using the [BDAPPV dataset](https://zenodo.org/records/12179554).

## Usage and Future Directions

The current app is meant to display real-time tuning of a meta-model based on several pretrained CNN models using a validation dataset from Google Maps images.  Users can view the output of the existing meta-model on any location by clicking on the map and selecting "Detect Panels."  Afterwards, they can submit feedback on whether they could observe a panel in the region of interest.  This feedback is stored and from the [/dashboard](https://solarscan.uw.r.appspot.com/dashboard) endpoint you can view each model's True Positive and True Negative performance for a given cutoff value.  The meta-model seeks to find a combination of models that maximizes prediction accuracy.  By clicking "Rerun Finding Optimal Params" the stored validation data is fit and new metamodel parameters are stored for use in the app.  At the bottom the history of metamodel performance at a given point in time is traced.

In the future, as more labelled validation data is collected, it should become possible to extract the stored dataset and use it to train new more robust CNN models which can be incorporated into this app.  These models on Vertex AI can be used to then map large areas for solar installations.

## Set-up

There are two separate processes to getting this up and running.

1. Getting `robust_pv_mapping` models onto Vertex AI
2. Deploy webapp to Google App Engine

### 1. Getting `robust_pv_mapping` models onto Vertex AI

##### Download pretrained models

This app uses pretrained models which can be found on [here on Zenodo](https://zenodo.org/records/14673918).  Download the whole `models.zip` and place the `*.pth` files into the `models/` directory.

##### Generate .mar packages 

Install the requirements needed for building models, handling data, and generating `.mar` archives needed for  

```pip install -r models/requirements.txt```

```pip install torch-model-archiver```


##### (optional) test .mar locally

Install torchserve to test locally

```pip install torchserve```

After installation start it up 

```torchserve --start --model-store models/vertex_archives --models model_name=model_blurring.mar```

```cat key_file.json```
find the inference token to pass to torchserve.

```
curl -X POST "http://localhost:8080/predictions/model_name" \                                
  -H "Authorization: Bearer APH29-dw" \
  -H "Content-Type: application/json" \
  -d @ts_ex.json
```

You should see a number around `~0.99`.  You can put other base64 encoded images into the same format as `ts_ex.json`.  If this is working without error, your models should be good to upload to.

##### Load into Vertex AI

1. Enable Vertex AI in a GCP project
2. Upoad `vertex_archives/` to a Cloud Storage bucket. 
3. Go to [Vertex AI Model Registry](https://console.cloud.google.com/vertex-ai/models) and "Import" each of the models from Cloud Storage ([instructions](https://cloud.google.com/vertex-ai/docs/model-registry/import-model/)).  Note that these are all PyTorch models.
4. Deploy all the models. 
    - You can do it manually as per [these instructions](https://cloud.google.com/vertex-ai/docs/general/deployment#google-cloud-console).  I also set up a [shared resource pool](https://cloud.google.com/vertex-ai/docs/predictions/model-co-hosting) so I didn't need a dedicated server for each model.  Even still keeping the smallest machine up for a day still costs about $3.
    - Since Vertex doesn't allow autoscaling to 0 instances, in order to make it easier to kill all the model deployment endpoints, I wrote the scipt in `deploy_models_to_vertex.py` to iterate through the list of models and deploy them. and then tear them down.  It does take a long time to deploy the models so it doesn't really work in a "real-time" sense, but this gives you the option to run it just during demo time and shut it down shortly after.

I found that the online endpoint tested seemed to be broken for PyTorch models, but I was able to test using the aiplatform api like so:

```
with open('ts_ex.json', 'r') as f:
    base64_image_string = json.load(f)['data'] 

from google.cloud import aiplatform

project_id = "XXXX"
location = "us-central1"
endpoint_id = "YYYY"

aiplatform.init(project=project_id, location=location)
endpoint = aiplatform.Endpoint(endpoint_id)
response = endpoint.predict(instances=[{"data": base64_image_string}])
print(response.predictions)
```

If that prints something like `~0.99` then your model is deployed and working correctly!

### 2. Deploy webapp to Google App Engine

To run the app, you'll need to enable App Engine, Cloud Datastore, Maps Javascript API, Maps Static API, and Geocoding API for your GCP project, and get an API key that you can use.   You'll also need to setup a [ipinfo.io](https://ipinfo.io) account and get a token.

##### Local Test

You can test locally just by running `python app.py` from within the `webapp` directory.  It will just run locally without connecting to app engine or datastore using dummy data, but it will try to use the Maps Javascript/Static/Geocoding API key and the ipinfo.io token.  You'll need to add those to the .env file where I've left them blank.  You can view it from [http://127.0.0.1/5000](http://127.0.0.1/5000)

##### Deploying to app engine

Similar to the local test you'll need the Maps API key and ipinfo.io token, but you'll need to add them to the blanks in app.yaml this time.  If you have the GCP CLI installed you can just run `gcloud app deploy` from within the webapp directory and the whole thing should deploy to your project URL in a few minutes.