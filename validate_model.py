import mlflow
import joblib
import pandas as pd
import yaml
from sklearn.metrics import f1_score, classification_report
import os
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_config(config):
    """Validate required configuration parameters."""
    required_params = [
        'excel_file_path', 'sheet_names', 'target_column', 
        'test_size', 'random_state'
    ]
    
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        logging.error(f"Missing required config parameters: {missing_params}")
        return False
    
    # Validate sheet names
    required_sheets = ['orders', 'freight', 'warehouse_costs', 'plant_ports']
    missing_sheets = [sheet for sheet in required_sheets if sheet not in config['sheet_names']]
    if missing_sheets:
        logging.error(f"Missing required sheet names: {missing_sheets}")
        return False
    
    return True

def load_test_data(config):
    """Loads a consistent, held-out test set for validation using the SAME preprocessing as training."""
    try:
        # Import functions from training script
        from train_model import unify_data, engineer_features
        
        logging.info("Loading and processing data using training script functions...")
        
        # Load and process data exactly as in training
        master_df = unify_data(config)
        if master_df is None:
            logging.error("Failed to load unified data")
            return None, None
            
        X, y = engineer_features(master_df, config)
        if X is None or y is None:
            logging.error("Failed to engineer features")
            return None, None
        
        # Use EXACT same train/test split as training (same random_state)
        _, X_test, _, y_test = train_test_split(
            X, y, 
            test_size=config['test_size'], 
            random_state=config['random_state'],  # CRITICAL: Same random state as training
            stratify=y
        )
        
        logging.info(f"Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        logging.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        return X_test, y_test
        
    except ImportError as e:
        logging.error(f"Cannot import from train_model.py: {e}")
        logging.error("Make sure train_model.py is in the same directory and has no syntax errors")
        return None, None
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        return None, None

def load_local_model_as_fallback():
    """Load the local .pkl file as a fallback if MLflow fails."""
    try:
        if os.path.exists('delay_model.pkl'):
            logging.info("Loading local delay_model.pkl as fallback...")
            model = joblib.load('delay_model.pkl')
            logging.info("Successfully loaded local model")
            return model
        else:
            logging.warning("Local delay_model.pkl not found")
            return None
    except Exception as e:
        logging.error(f"Error loading local model: {e}")
        return None

def get_model_from_run(client, run_id):
    """Safely load a model from a specific run with enhanced error handling."""
    try:
        # First check if the run exists
        run = client.get_run(run_id)
        logging.info(f"Found run: {run_id}")
        
        # Check if model artifacts exist
        artifacts = client.list_artifacts(run_id)
        logging.info(f"Available artifacts in run {run_id}:")
        for artifact in artifacts:
            logging.info(f"  - {artifact.path}")
        
        # Look for model artifact - your training script uses 'model'
        model_artifact_paths = ["model", "sklearn-model", "trained-model"]
        model_path = None
        
        for artifact in artifacts:
            if artifact.path in model_artifact_paths:
                model_path = artifact.path
                break
        
        if model_path is None:
            logging.error(f"No model artifact found in run {run_id}")
            logging.error("Expected artifact paths: model, sklearn-model, or trained-model")
            logging.error("Available artifacts:")
            for artifact in artifacts:
                logging.error(f"  - {artifact.path}")
            return None
        
        # Try to load the model
        model_uri = f"runs:/{run_id}/{model_path}"
        logging.info(f"Attempting to load model from: {model_uri}")
        
        model = mlflow.sklearn.load_model(model_uri)
        logging.info(f"Successfully loaded model from run: {run_id}")
        return model
        
    except mlflow.exceptions.MlflowException as e:
        logging.error(f"MLflow error loading model from run {run_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error loading model from run {run_id}: {e}")
        return None

def find_valid_candidate_model(client, experiment_name="supply_chain_delay_prediction", max_runs=20):
    """Find the most recent run that has a valid model artifact."""
    try:
        # Get all experiments and look for our experiment
        experiments = client.search_experiments()
        logging.info(f"Available experiments:")
        for exp in experiments:
            logging.info(f"  - {exp.name} (ID: {exp.experiment_id})")
        
        # Try to find the experiment by name
        experiment_id = None
        for exp in experiments:
            if exp.name == experiment_name:
                experiment_id = exp.experiment_id
                break
        
        if experiment_id is None:
            logging.warning(f"Experiment '{experiment_name}' not found, using default experiment")
            experiment_id = "0"
        
        logging.info(f"Using experiment ID: {experiment_id}")
        
        # Get all runs from the experiment
        runs = client.search_runs(
            experiment_ids=[experiment_id], 
            order_by=["start_time DESC"], 
            max_results=max_runs
        )
        
        if not runs:
            logging.error(f"No runs found in experiment {experiment_id}")
            return None, None
            
        logging.info(f"Found {len(runs)} runs in experiment, checking for valid models...")
        
        # Check each run for a valid model
        for i, run in enumerate(runs):
            run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName", "Unknown")
            run_status = run.info.status
            
            logging.info(f"Checking run {i+1}/{len(runs)}: {run_id} ({run_name}) - Status: {run_status}")
            
            # Skip failed runs
            if run_status != "FINISHED":
                logging.info(f"  Skipping run {run_id} - Status: {run_status}")
                continue
            
            model = get_model_from_run(client, run_id)
            if model is not None:
                logging.info(f"Found valid model in run: {run_id}")
                return model, run_id
        
        logging.error("No valid model found in any runs")
        return None, None
        
    except Exception as e:
        logging.error(f"Error searching for candidate model: {e}")
        return None, None

def get_most_recent_run_id(client, experiment_name="supply_chain_delay_prediction"):
    """Get the most recent run ID for fallback model loading."""
    try:
        experiments = client.search_experiments()
        experiment_id = None
        
        for exp in experiments:
            if exp.name == experiment_name:
                experiment_id = exp.experiment_id
                break
        
        if experiment_id is None:
            experiment_id = "0"
        
        runs = client.search_runs(
            experiment_ids=[experiment_id], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        
        if runs:
            return runs[0].info.run_id
        return None
        
    except Exception as e:
        logging.error(f"Error getting most recent run: {e}")
        return None

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate a model and return comprehensive metrics."""
    try:
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Get detailed classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        logging.info(f"{model_name} Performance:")
        logging.info(f"  F1-score (weighted): {f1:.4f}")
        logging.info(f"  Accuracy: {report['accuracy']:.4f}")
        logging.info(f"  Precision (weighted): {report['weighted avg']['precision']:.4f}")
        logging.info(f"  Recall (weighted): {report['weighted avg']['recall']:.4f}")
        
        return f1, report
    except Exception as e:
        logging.error(f"Error evaluating {model_name}: {e}")
        return None, None

def validate_and_promote_model():
    """
    Compares the latest candidate model to the production model and promotes it if it's better.
    """
    logging.info("Starting model validation and promotion process...")
    
    # Load configuration
    try:
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logging.error("config.yaml not found. Please create a config.yaml file.")
        return
    except Exception as e:
        logging.error(f"Error loading config: {e}")
        return
    
    # Validate configuration
    if not validate_config(config):
        logging.error("Configuration validation failed")
        return

    # Set up MLflow
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()
    
    # Test connection
    try:
        experiments = client.search_experiments()
        logging.info(f"Connected to MLflow. Found {len(experiments)} experiments.")
    except Exception as e:
        logging.error(f"Cannot connect to MLflow server: {e}")
        logging.info("Make sure MLflow server is running: mlflow server --host 127.0.0.1 --port 5000")
        return
    
    model_name = "SupplyChainDelayPredictor"
    
    # 1. Get a valid candidate model
    candidate_model, candidate_run_id = find_valid_candidate_model(client)
    
    # If MLflow model loading fails, try local model as fallback
    if candidate_model is None:
        logging.warning("Could not find valid model in MLflow, trying local model...")
        candidate_model = load_local_model_as_fallback()
        if candidate_model is None:
            logging.error("Could not find a valid candidate model in MLflow or locally.")
            logging.info("Please run train_model.py first to create a trained model.")
            return
        else:
            # Get the most recent run ID for tracking purposes
            candidate_run_id = get_most_recent_run_id(client)
            logging.info(f"Using local model with run ID: {candidate_run_id}")
    
    # 2. Get the current production model
    production_model = None
    production_version = None
    try:
        latest_versions = client.get_latest_versions(name=model_name, stages=["Production"])
        if latest_versions:
            production_version = latest_versions[0]
            production_model_uri = production_version.source
            production_model = mlflow.sklearn.load_model(production_model_uri)
            logging.info(f"Loaded production model version {production_version.version}")
        else:
            logging.info("No production model found in registry")
    except mlflow.exceptions.RestException as e:
        logging.info(f"Model registry not found or no production model: {e}")
        logging.info("This is normal for first-time setup")
    except Exception as e:
        logging.error(f"Error loading production model: {e}")

    # 3. Evaluate both models on the same test set
    X_test, y_test = load_test_data(config)
    if X_test is None or y_test is None:
        logging.error("Could not load test data. Aborting validation.")
        return

    # Evaluate candidate model
    candidate_f1, candidate_report = evaluate_model(candidate_model, X_test, y_test, "Candidate")
    if candidate_f1 is None:
        logging.error("Failed to evaluate candidate model")
        return

    # 4. Compare and promote
    should_promote = False
    
    if production_model is not None:
        # Compare with existing production model
        production_f1, production_report = evaluate_model(production_model, X_test, y_test, "Production")
        
        if production_f1 is not None:
            logging.info(f"\nModel Comparison:")
            logging.info(f"Candidate F1-score: {candidate_f1:.4f}")
            logging.info(f"Production F1-score: {production_f1:.4f}")
            logging.info(f"Improvement: {candidate_f1 - production_f1:.4f}")
            
            if candidate_f1 > production_f1:
                logging.info("✅ Candidate model performs better than production model")
                should_promote = True
            else:
                logging.info("❌ Candidate model does not outperform production model")
        else:
            logging.warning("Could not evaluate production model, promoting candidate")
            should_promote = True
    else:
        # No production model exists, promote the candidate
        logging.info("No existing production model. Promoting candidate to Production.")
        should_promote = True
    
    # 5. Promote model if needed
    if should_promote and candidate_run_id is not None:
        try:
            logging.info("🚀 Promoting candidate model to Production...")
            
            # Create registered model if it doesn't exist
            try:
                registered_model = client.get_registered_model(model_name)
                logging.info(f"Using existing registered model: {model_name}")
            except mlflow.exceptions.RestException:
                client.create_registered_model(model_name)
                logging.info(f"Created new registered model: {model_name}")
            
            # Register the new version
            model_version = client.create_model_version(
                name=model_name,
                source=f"runs:/{candidate_run_id}/model",
                run_id=candidate_run_id,
                description=f"Model promoted with F1-score: {candidate_f1:.4f}"
            )
            
            # Transition old production model to "Archived"
            if production_version:
                client.transition_model_version_stage(
                    name=model_name, 
                    version=production_version.version, 
                    stage="Archived",
                    archive_existing_versions=False
                )
                logging.info(f"Archived old production model version {production_version.version}")
            
            # Promote the new candidate model
            client.transition_model_version_stage(
                name=model_name, 
                version=model_version.version, 
                stage="Production"
            )
            
            logging.info(f"✅ Successfully promoted model version {model_version.version} to Production")
            logging.info(f"📊 New production model F1-score: {candidate_f1:.4f}")
            
        except Exception as e:
            logging.error(f"❌ Error promoting model: {e}")
            logging.error("Model evaluation completed but promotion failed")
    else:
        if should_promote:
            logging.warning("🔄 Model should be promoted but no run ID available")
        else:
            logging.info("🔄 No model promotion needed")

    logging.info("Model validation and promotion process completed")

if __name__ == "__main__":
    validate_and_promote_model()