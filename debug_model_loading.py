import mlflow
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)

def debug_model_loading():
    """Debug script to test model loading from both MLflow and local file."""
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()
    
    try:
        # Test MLflow connection
        experiments = client.search_experiments()
        print(f"✅ Connected to MLflow. Found {len(experiments)} experiments:")
        for exp in experiments:
            print(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Get the most recent run
        experiment_id = "0"  # Default experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id], 
            order_by=["start_time DESC"], 
            max_results=5
        )
        
        print(f"\n📊 Found {len(runs)} recent runs:")
        for i, run in enumerate(runs):
            print(f"  {i+1}. Run ID: {run.info.run_id}")
            print(f"     Status: {run.info.status}")
            print(f"     Start time: {run.info.start_time}")
            
            # Check artifacts
            try:
                artifacts = client.list_artifacts(run.info.run_id)
                print(f"     Artifacts: {[art.path for art in artifacts]}")
                
                # Try to load model if it exists
                for artifact in artifacts:
                    if artifact.path == "model":
                        try:
                            model_uri = f"runs:/{run.info.run_id}/model"
                            print(f"     Attempting to load: {model_uri}")
                            model = mlflow.sklearn.load_model(model_uri)
                            print(f"     ✅ Successfully loaded model from MLflow!")
                            print(f"     Model type: {type(model)}")
                            return model
                        except Exception as e:
                            print(f"     ❌ Failed to load model: {e}")
                            
            except Exception as e:
                print(f"     ❌ Error listing artifacts: {e}")
            print()
    
    except Exception as e:
        print(f"❌ MLflow connection error: {e}")
    
    # Try local model loading
    print("🔄 Trying local model loading...")
    try:
        if os.path.exists('delay_model.pkl'):
            model = joblib.load('delay_model.pkl')
            print(f"✅ Successfully loaded local model!")
            print(f"Model type: {type(model)}")
            return model
        else:
            print("❌ Local delay_model.pkl not found")
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
    
    print("❌ No valid model found in MLflow or locally")
    return None

if __name__ == "__main__":
    debug_model_loading()