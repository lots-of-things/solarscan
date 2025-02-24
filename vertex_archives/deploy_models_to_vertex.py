from google.cloud import aiplatform
# Example usage
project_id = "739182013691"
location = "us-central1"
aiplatform.init(project=project_id, location=location)


# Function to deploy a shared resource pool
def deploy_shared_resource_pool(resource_pool_name):
    """
    Deploy a shared resource pool that will be used for all the models.
    """
    # For simplicity, we'll use the shared resource pool as a "model" endpoint for multiple models.
    # Create a new endpoint as the resource pool.
    aiplatform.DeploymentResourcePool.create(
        deployment_resource_pool_id=resource_pool_name,  # User-specified ID
        machine_type="e2-standard-2",  # Machine type
        min_replica_count=1,  # Minimum number of replicas
        max_replica_count=1,  # Maximum number of replicas
    )

# Function to find the shared resource pool (endpoint) by its name
def get_deployment_pool_by_name(pool_name):
    """
    Find the shared resource pool (endpoint) by its name.
    """

    for drp in aiplatform.DeploymentResourcePool.list():
        if drp.name == pool_name:
            return drp
    return None
 
# Function to deploy models to endpoints based on the list of model names
def deploy_endpoints(model_names, resource_pool_name):
    """
    Turn on (deploy) endpoints for all models in the list by using a shared resource pool.
    """
    drp = get_deployment_pool_by_name(resource_pool_name)
    
    for model_name in model_names:
        # Deploy the model to the shared resource pool endpoint
        model_by_name = aiplatform.Model.list(filter=f'display_name="{model_name}"')
        model = aiplatform.Model(model_name=model_by_name[0].name)
        model.deploy(deployment_resource_pool=drp)
        

# Function to undeploy all models at the endpoint and delete the endpoint if empty
def undeploy_and_delete_endpoints(pool_name):
    """
    Undeploy all models at the endpoint shared pool, and delete the endpoint if no models remain deployed.
    """
    # Find the shared pool (endpoint) by its name
    drp = get_deployment_pool_by_name(pool_name)

    for drp_model in drp.query_deployed_models()[0]:
        endpoint = aiplatform.Endpoint(drp_model.endpoint.split('/')[-1])
        endpoint.undeploy_all()
        endpoint.delete()

    drp.delete()
    

model_names = [
    "blurring",
    "spectral",
    "augmix",
    "standard",
    "oracle",
]
resource_pool_name = "shared-resource-pool"
deploy_shared_resource_pool(resource_pool_name)
    
# Step 2: Deploy all models to the shared resource pool
deploy_endpoints(model_names, resource_pool_name)

# Step 3: Undeploy the models (turn off endpoints)
# undeploy_and_delete_endpoints(resource_pool_name)