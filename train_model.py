import pandas as pd
import yaml
import os
import logging
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from category_encoders import TargetEncoder
from pydantic import BaseModel, ValidationError, Field, ConfigDict
from typing import List, Union, Optional
import re

# --- Setup: Structured Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log", mode='w'), # Overwrite log file each run
        logging.StreamHandler()
    ]
)

# --- 1. Data Validation Schema (The System's Immune System) ---
class OrderRow(BaseModel):
    """Pydantic model to define the expected schema for the raw orders data AFTER cleaning."""
    model_config = ConfigDict(extra='ignore')

    order_id: int
    order_date: str
    origin_port: str
    carrier: str
    tpt: int
    service_level: str
    ship_ahead_day_count: int
    ship_late_day_count: int
    customer: str
    product_id: int
    plant_code: str
    destination_port: str
    unit_quantity: int
    weight: float  # Fixed: changed from weight_kilograms to weight
    unit_price: Optional[float] = None  # Made optional in case it's missing
    freight_cost_usd: Optional[float] = None  # Made optional in case it's missing

def validate_raw_data(df: pd.DataFrame) -> bool:
    """Validates the cleaned dataframe against the Pydantic schema."""
    logging.info("Validating cleaned data against the defined schema...")
    logging.info(f"Available columns: {df.columns.tolist()}")
    
    try:
        df_val = df.copy()
        
        # Convert data types to match schema expectations
        df_val['order_id'] = df_val['order_id'].astype(int)
        df_val['order_date'] = df_val['order_date'].astype(str)
        df_val['tpt'] = df_val['tpt'].astype(int)
        df_val['ship_ahead_day_count'] = df_val['ship_ahead_day_count'].astype(int)
        df_val['ship_late_day_count'] = df_val['ship_late_day_count'].astype(int)
        df_val['product_id'] = df_val['product_id'].astype(int)
        df_val['unit_quantity'] = df_val['unit_quantity'].astype(int)
        df_val['weight'] = df_val['weight'].astype(float)
        
        # Handle optional columns
        if 'unit_price' in df_val.columns:
            df_val['unit_price'] = df_val['unit_price'].astype(float)
        if 'freight_cost_usd' in df_val.columns:
            df_val['freight_cost_usd'] = df_val['freight_cost_usd'].astype(float)
        
        # Validate a sample of rows (first 100 or all if less than 100)
        sample_size = min(100, len(df_val))
        sample_data = df_val.head(sample_size)
        
        _ = [OrderRow(**row) for row in sample_data.to_dict(orient='records')]
        logging.info(f"Data validation successful for {sample_size} sample rows. Schema is correct.")
        return True
        
    except ValidationError as e:
        logging.error("FATAL: Data validation failed. The data does not match the required schema.")
        logging.error(f"Validation error details: {e}")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred during data validation: {e}")
        return False


# --- 2. Modular Functions ---
def load_config(config_path="config.yaml"):
    """Loads the configuration file."""
    logging.info(f"Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logging.error(f"FATAL: Could not load {config_path}: {e}")
        return None

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes all column names to a consistent snake_case format.
    This version uses simple string replacement for maximum reliability.
    """
    logging.info("Standardizing column names...")
    cols = df.columns
    new_cols = []
    for col in cols:
        new_col = str(col).lower()          # 1. Convert to lowercase
        new_col = new_col.replace(' ', '_')   # 2. Replace spaces with underscores
        new_col = new_col.replace('(', '')    # 3. Remove opening parenthesis
        new_col = new_col.replace(')', '')    # 4. Remove closing parenthesis
        new_col = new_col.replace('-', '_')   # 5. Replace hyphens with underscores
        new_col = new_col.replace('.', '_')   # 6. Replace dots with underscores
        new_col = new_col.replace('/', '_')   # 7. Replace forward slashes with underscores
        # Remove multiple underscores
        new_col = re.sub(r'_+', '_', new_col)
        # Remove leading/trailing underscores
        new_col = new_col.strip('_')
        new_cols.append(new_col)
    df.columns = new_cols
    logging.info(f"Cleaned column names: {df.columns.tolist()}")
    return df

def diagnose_data_columns(df, name="dataframe"):
    """Diagnostic function to understand the structure of your data."""
    logging.info(f"=== DIAGNOSTIC INFO FOR {name.upper()} ===")
    logging.info(f"Shape: {df.shape}")
    logging.info(f"Columns ({len(df.columns)}): {df.columns.tolist()}")
    
    # Look for potential date columns
    date_like_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'delivery', 'scheduled', 'delivered'])]
    if date_like_cols:
        logging.info(f"Potential date columns: {date_like_cols}")
    
    # Look for potential weight columns  
    weight_like_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['weight', 'kg', 'kilogram'])]
    if weight_like_cols:
        logging.info(f"Potential weight columns: {weight_like_cols}")
    
    # Look for potential cost columns
    cost_like_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['cost', 'price', 'freight', 'usd'])]
    if cost_like_cols:
        logging.info(f"Potential cost columns: {cost_like_cols}")
    
    # Show first few rows
    logging.info(f"First 3 rows:\n{df.head(3)}")
    logging.info("=== END DIAGNOSTIC INFO ===\n")

def unify_data(config):
    """Loads, cleans, and merges multiple data sources from a single Excel file."""
    logging.info("--- Starting Data Unification from Excel file ---")
    try:
        file_path = config['excel_file_path']
        sheets = config['sheet_names']

        # Load orders data
        orders_df = pd.read_excel(file_path, sheet_name=sheets['orders'])
        
        # Clean column names immediately after loading
        orders_df = clean_column_names(orders_df)
        
        # Add diagnostic information
        diagnose_data_columns(orders_df, "Orders Data")
        
        # Now, run validation on the cleaned dataframe
        if not validate_raw_data(orders_df):
            logging.error("Data validation failed. Continuing with processing but be aware of potential issues.")
            # Don't return None, continue processing
            
        # Load and clean other dataframes
        freight_df = clean_column_names(pd.read_excel(file_path, sheet_name=sheets['freight']))
        wh_costs_df = clean_column_names(pd.read_excel(file_path, sheet_name=sheets['warehouse_costs']))
        plant_ports_df = clean_column_names(pd.read_excel(file_path, sheet_name=sheets['plant_ports']))
        
        logging.info(f"Orders columns: {orders_df.columns.tolist()}")
        logging.info(f"Plant ports columns: {plant_ports_df.columns.tolist()}")
        logging.info(f"Warehouse costs columns: {wh_costs_df.columns.tolist()}")
        logging.info(f"Freight columns: {freight_df.columns.tolist()}")
        
        # Merge dataframes
        master_df = pd.merge(orders_df, plant_ports_df, on='plant_code', how='left')
        
        # Rename warehouse costs column to match
        if 'wh' in wh_costs_df.columns:
            wh_costs_df.rename(columns={'wh': 'plant_code'}, inplace=True)
        
        master_df = pd.merge(master_df, wh_costs_df, on='plant_code', how='left')
        
        # Handle freight merge - check for correct column names
        freight_origin_col = 'orig_port_cd' if 'orig_port_cd' in freight_df.columns else 'origin_port'
        master_df = pd.merge(master_df, freight_df, left_on=['carrier', 'origin_port'], 
                           right_on=['carrier', freight_origin_col], how='left')
        
        logging.info(f"Final master dataframe shape: {master_df.shape}")
        logging.info(f"Final master dataframe columns: {master_df.columns.tolist()}")
        
        # Add diagnostic for final merged data
        diagnose_data_columns(master_df, "Final Master Data")
        
        logging.info("Data unification successful.")
        return master_df
        
    except FileNotFoundError:
        logging.error(f"FATAL: Excel file not found at '{config['excel_file_path']}'. Please check the path in config.yaml.")
        return None
    except Exception as e:
        logging.error(f"FATAL: Failed during data unification: {e}")
        return None

def engineer_features(df, config):
    """Creates target variable and new features based on available data."""
    logging.info("--- Starting Feature Engineering ---")
    df_eng = df.copy()
    
    logging.info(f"Input dataframe columns: {df_eng.columns.tolist()}")
    
    # Check if we have ship_ahead_day_count and ship_late_day_count
    if 'ship_ahead_day_count' not in df_eng.columns or 'ship_late_day_count' not in df_eng.columns:
        logging.error("Cannot find ship_ahead_day_count or ship_late_day_count columns. These are required for target creation.")
        logging.error("Available columns: " + str(df_eng.columns.tolist()))
        return None, None
    
    # Create target variable based on ship_late_day_count
    # If ship_late_day_count > 0, it's considered delayed
    df_eng[config['target_column']] = (df_eng['ship_late_day_count'] > 0).astype(int)
    
    # Create delivery delay feature (can be negative for early deliveries)
    df_eng['delivery_delay'] = df_eng['ship_late_day_count'] - df_eng['ship_ahead_day_count']
    
    # Convert order_date to datetime if it exists
    if 'order_date' in df_eng.columns:
        df_eng['order_date'] = pd.to_datetime(df_eng['order_date'], errors='coerce')
        
        # Extract date features if order_date is available
        df_eng['order_month'] = df_eng['order_date'].dt.month
        df_eng['order_day_of_week'] = df_eng['order_date'].dt.dayofweek
        df_eng['order_quarter'] = df_eng['order_date'].dt.quarter
    
    # Create cost-related features
    cost_col = 'cost_unit' if 'cost_unit' in df_eng.columns else None
    if cost_col:
        df_eng['cost_per_unit'] = df_eng[cost_col]
        # Create cost per kg if weight is available
        if 'weight' in df_eng.columns:
            df_eng['cost_per_kg'] = df_eng['cost_per_unit'] / df_eng['weight'].replace(0, 1)
        
        # Fill missing values
        df_eng['cost_per_unit'].fillna(df_eng['cost_per_unit'].median(), inplace=True)
        if 'cost_per_kg' in df_eng.columns:
            df_eng['cost_per_kg'].fillna(df_eng['cost_per_kg'].median(), inplace=True)
    else:
        logging.warning("No cost column found. Creating dummy cost features.")
        df_eng['cost_per_unit'] = 10.0  # Default value
        if 'weight' in df_eng.columns:
            df_eng['cost_per_kg'] = 10.0 / df_eng['weight'].replace(0, 1)
    
    # Create weight-related features
    if 'weight' in df_eng.columns:
        df_eng['weight_category'] = pd.cut(df_eng['weight'], 
                                          bins=[0, 100, 500, 1000, float('inf')], 
                                          labels=['Light', 'Medium', 'Heavy', 'VeryHeavy'])
        df_eng['weight_category'] = df_eng['weight_category'].astype(str)
    
    # Create TPT (transit time) categories
    if 'tpt' in df_eng.columns:
        df_eng['tpt_category'] = pd.cut(df_eng['tpt'], 
                                       bins=[0, 7, 14, 30, float('inf')], 
                                       labels=['Fast', 'Medium', 'Slow', 'VerySlow'])
        df_eng['tpt_category'] = df_eng['tpt_category'].astype(str)
    
    # Create quantity-related features
    if 'unit_quantity' in df_eng.columns:
        df_eng['quantity_category'] = pd.cut(df_eng['unit_quantity'], 
                                            bins=[0, 10, 50, 100, float('inf')], 
                                            labels=['Small', 'Medium', 'Large', 'VeryLarge'])
        df_eng['quantity_category'] = df_eng['quantity_category'].astype(str)
    
    # Fill missing values for categorical columns
    categorical_cols = ['origin_port', 'carrier', 'service_level', 'customer', 'plant_code', 'destination_port']
    for col in categorical_cols:
        if col in df_eng.columns:
            df_eng[col].fillna('Unknown', inplace=True)
    
    # Define features to use for modeling
    base_features = [
        'tpt', 'ship_ahead_day_count', 'unit_quantity', 'weight',
        'origin_port', 'carrier', 'service_level', 'customer', 'plant_code', 'destination_port'
    ]
    
    # Add engineered features
    engineered_features = [
        'delivery_delay', 'cost_per_unit'
    ]
    
    # Add date features if available
    if 'order_date' in df_eng.columns and not df_eng['order_date'].isna().all():
        engineered_features.extend(['order_month', 'order_day_of_week', 'order_quarter'])
    
    # Add categorical features if created
    if 'weight_category' in df_eng.columns:
        engineered_features.append('weight_category')
    if 'tpt_category' in df_eng.columns:
        engineered_features.append('tpt_category')
    if 'quantity_category' in df_eng.columns:
        engineered_features.append('quantity_category')
    if 'cost_per_kg' in df_eng.columns:
        engineered_features.append('cost_per_kg')
    
    all_features = base_features + engineered_features
    
    # Check which features actually exist in the dataframe
    available_features = [col for col in all_features if col in df_eng.columns]
    missing_features = [col for col in all_features if col not in df_eng.columns]
    
    if missing_features:
        logging.warning(f"Missing features (will be skipped): {missing_features}")
    
    logging.info(f"Using features: {available_features}")
    
    # Remove rows with missing target
    df_eng = df_eng.dropna(subset=[config['target_column']])
    
    X = df_eng[available_features]
    y = df_eng[config['target_column']]
    
    logging.info(f"Feature matrix shape: {X.shape}")
    logging.info(f"Target variable distribution: {y.value_counts().to_dict()}")
    
    # Check if we have enough data for both classes
    if len(y.unique()) < 2:
        logging.error("Target variable has only one class. Cannot train a binary classifier.")
        return None, None
    
    return X, y

def train_model(X_train, y_train, config):
    """Builds and trains the model pipeline with hyperparameter tuning."""
    logging.info("--- Starting Model Training ---")
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    logging.info(f"Numerical features: {numerical_features}")
    logging.info(f"Categorical features: {categorical_features}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', TargetEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Calculate scale_pos_weight for class imbalance
    if y_train.value_counts().get(1, 0) > 0:
        scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    else:
        scale_pos_weight = 1
    
    logging.info(f"Scale pos weight: {scale_pos_weight}")
    
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        scale_pos_weight=scale_pos_weight,
        random_state=config['random_state']
    )
    
    ml_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', model)
    ])
    
    search = RandomizedSearchCV(
        ml_pipeline,
        param_distributions=config['param_distributions'],
        n_iter=config['n_iter'],
        cv=config['cv'],
        scoring=config['scoring'],
        n_jobs=-1,
        random_state=config['random_state'],
        verbose=1
    )
    
    search.fit(X_train, y_train)
    logging.info(f"Best parameters found: {search.best_params_}")
    return search.best_estimator_, search.best_score_

def log_model_diagnostics(model_pipeline, X_test, y_test):
    """Generates, saves, and logs diagnostic plots and reports for the model."""
    logging.info("--- Generating and Logging Model Diagnostics ---")
    
    try:
        # Get feature importances
        importances = model_pipeline.named_steps['classifier'].feature_importances_
        
        # Get feature names after preprocessing
        preprocessor = model_pipeline.named_steps['preprocessor']
        
        # Get numerical feature names
        num_features = list(preprocessor.transformers_[0][2])
        
        # Get categorical feature names
        cat_features = list(preprocessor.transformers_[1][2])
        
        # Combine all feature names
        all_feature_names = num_features + cat_features
        
        # Ensure we have the right number of feature names
        if len(all_feature_names) != len(importances):
            logging.warning(f"Feature name count ({len(all_feature_names)}) doesn't match importance count ({len(importances)})")
            # Create generic feature names if mismatch
            all_feature_names = [f"feature_{i}" for i in range(len(importances))]
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(15)
        
        # Plot feature importance
        fig_imp, ax_imp = plt.subplots(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importance_df, ax=ax_imp)
        ax_imp.set_title('Top 15 Feature Importances')
        ax_imp.set_xlabel('Importance')
        ax_imp.set_ylabel('Features')
        plt.tight_layout()
        feature_importance_path = "feature_importance.png"
        fig_imp.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(feature_importance_path, "diagnostics")
        plt.close(fig_imp)
        
        logging.info("Feature importance plot generated successfully.")
        
    except Exception as e:
        logging.warning(f"Could not generate feature importance plot: {e}")
        logging.warning(f"Error details: {str(e)}")
        
        # Fallback: create a simple plot with generic names
        try:
            importances = model_pipeline.named_steps['classifier'].feature_importances_
            simple_names = [f"feature_{i}" for i in range(len(importances))]
            importance_df = pd.DataFrame({
                'feature': simple_names,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            fig_imp, ax_imp = plt.subplots(figsize=(10, 8))
            sns.barplot(x='importance', y='feature', data=importance_df, ax=ax_imp)
            ax_imp.set_title('Top 15 Feature Importances (Generic Names)')
            plt.tight_layout()
            feature_importance_path = "feature_importance_fallback.png"
            fig_imp.savefig(feature_importance_path)
            mlflow.log_artifact(feature_importance_path, "diagnostics")
            plt.close(fig_imp)
            logging.info("Fallback feature importance plot generated successfully.")
            
        except Exception as e2:
            logging.error(f"Even fallback feature importance plot failed: {e2}")

    # Generate confusion matrix
    y_pred = model_pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm, ax_cm = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm, 
                xticklabels=['On-Time', 'Delayed'], yticklabels=['On-Time', 'Delayed'])
    ax_cm.set_xlabel('Predicted')
    ax_cm.set_ylabel('Actual')
    ax_cm.set_title('Confusion Matrix')
    confusion_matrix_path = "confusion_matrix.png"
    fig_cm.savefig(confusion_matrix_path)
    mlflow.log_artifact(confusion_matrix_path, "diagnostics")
    plt.close(fig_cm)

    # Generate classification report
    report_str = classification_report(y_test, y_pred)
    report_path = "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write(report_str)
    mlflow.log_artifact(report_path, "diagnostics")
    
    logging.info("Model diagnostics generated successfully.")

def main():
    """Main function to orchestrate the entire training pipeline."""
    logging.info("Starting Supply Chain Delay Prediction Model Training...")
    
    # Load configuration
    config = load_config()
    if config is None:
        logging.error("Failed to load configuration. Exiting.")
        return

    # CRITICAL: Set up MLflow tracking
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Test MLflow connection
    try:
        mlflow.set_experiment("supply_chain_delay_prediction")
        logging.info("MLflow connection established successfully.")
    except Exception as e:
        logging.error(f"Cannot connect to MLflow server: {e}")
        logging.error("Please ensure MLflow server is running: mlflow server --host 127.0.0.1 --port 5000")
        return

    # Start MLflow run
    with mlflow.start_run():
        try:
            # Log configuration parameters
            mlflow.log_params(config.get('param_distributions', {}))
            mlflow.log_param("n_iter", config.get('n_iter', 10))
            mlflow.log_param("cv", config.get('cv', 3))
            mlflow.log_param("test_size", config.get('test_size', 0.2))
            mlflow.log_param("random_state", config.get('random_state', 42))

            # Data processing
            master_df = unify_data(config)
            if master_df is None:
                logging.error("Stopping pipeline due to data unification failure.")
                return

            X, y = engineer_features(master_df, config)
            if X is None or y is None:
                logging.error("Stopping pipeline due to feature engineering failure.")
                return
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config.get('test_size', 0.2), 
                random_state=config.get('random_state', 42), 
                stratify=y
            )
            
            logging.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

            # Train model
            best_pipeline, best_val_score = train_model(X_train, y_train, config)
            
            # Log validation score
            mlflow.log_metric(f"best_validation_{config.get('scoring', 'f1')}", best_val_score)

            # Generate predictions and log test metrics
            y_pred = best_pipeline.predict(X_test)
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            
            # Log test metrics
            mlflow.log_metric("test_f1_score_weighted", report_dict['weighted avg']['f1-score'])
            mlflow.log_metric("test_precision_weighted", report_dict['weighted avg']['precision'])
            mlflow.log_metric("test_recall_weighted", report_dict['weighted avg']['recall'])
            mlflow.log_metric("test_accuracy", report_dict['accuracy'])
            
            # Generate and log diagnostics
            log_model_diagnostics(best_pipeline, X_test, y_test)
            
            # Save model locally
            model_output_path = config.get('model_output_path', 'supply_chain_model.pkl')
            joblib.dump(best_pipeline, model_output_path)
            
            # CRITICAL: Log model to MLflow
            mlflow.sklearn.log_model(
                sk_model=best_pipeline, 
                artifact_path="model",
                signature=mlflow.models.infer_signature(X_test, best_pipeline.predict(X_test)),
                registered_model_name="SupplyChainDelayPredictor"
            )
            
            run_id = mlflow.active_run().info.run_id
            logging.info(f"Model training completed successfully!")
            logging.info(f"Run ID: {run_id}")
            logging.info(f"Model saved locally to: {model_output_path}")
            logging.info("Model logged to MLflow with artifact path: 'model'")
            logging.info("To view results, run: mlflow ui")
            
        except Exception as e:
            logging.error(f"Training pipeline failed: {e}")
            raise

if __name__ == "__main__":
    main()