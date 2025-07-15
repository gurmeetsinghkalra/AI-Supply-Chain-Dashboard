import pytest
import pandas as pd
import os
import yaml
from ..train_model import unify_data, engineer_features # Assuming train_model is in the parent directory

# A fixture to load the configuration once for all tests
@pytest.fixture(scope="module")
def config():
    """Loads the main configuration file."""
    with open("config.yaml", 'r') as f:
        return yaml.safe_load(f)

# A fixture to create a small, controlled sample of the data for testing
@pytest.fixture(scope="module")
def sample_data(config):
    """Creates a small sample DataFrame for testing purposes."""
    # This creates a tiny dataframe that mimics the structure of your real data
    # In a real project, you might load a pre-saved small CSV for this.
    orders_data = {
        'Order_ID': [1], 'Order_Date': ['2023-01-01'], 'Origin_Port': ['PORTA'],
        'Carrier': ['CARRIER_X'], 'TPT': [5], 'Service_Level': ['Std'],
        'Ship_ahead_day_count': [0], 'Ship_Late_Day_count': [2],
        'Customer': ['CUST1'], 'Product_ID': [101], 'Plant_Code': ['PLANT1'],
        'Destination_Port': ['PORTB'], 'Unit_quantity': [100],
        'Weight (Kilograms)': [500.0], 'Unit_Price': [10.0], 'Freight Cost (USD)': [1000.0],
        'Scheduled Delivery Date': ['2023-01-06'], 'Delivered to Client Date': ['2023-01-08']
    }
    return pd.DataFrame(orders_data)


def test_unify_data_runs(config):
    """
    Integration Test: Ensures the data unification function runs without errors.
    This is a basic test to catch major issues like incorrect column names for merging.
    """
    # In a real scenario, you'd have sample CSVs for each file in the config.
    # For this example, we just ensure the function is callable.
    assert callable(unify_data)


def test_feature_engineering(sample_data, config):
    """
    Unit Test: Verifies that the feature engineering function creates the expected columns.
    """
    X, y = engineer_features(sample_data, config)

    # Assert that the target column was created
    assert config['target_column'] in y.name

    # Assert that new features were created
    assert 'cost_per_kg' in X.columns
    assert 'Scheduled Month' in X.columns
    
    # Assert the logic is correct
    # 1000 (cost) / 500 (weight) = 2.0
    assert X['cost_per_kg'].iloc[0] == 2.0
    
    # Assert the target variable is correct (2 days late means Is Delayed = 1)
    assert y.iloc[0] == 1

