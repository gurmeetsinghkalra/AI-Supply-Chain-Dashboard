# AI-Supply-Chain-Dashboard
🚀 AI-Powered Supply Chain Command Center
This project is a state-of-the-art, end-to-end Machine Learning Operations (MLOps) system designed to predict shipment delays in a complex supply chain. It goes beyond a simple dashboard by implementing a full, automated pipeline for data validation, model training, and deployment, ensuring the system is robust, reproducible, and always using the most accurate AI model.

✨ Key Features
Automated ML Pipeline: A production-grade pipeline that unifies multiple data sources, engineers advanced features, and trains a high-performance XGBoost model.

Intelligent Data Validation: The system is protected by a Pydantic validation layer that automatically checks raw data for schema correctness, preventing a "garbage in, garbage out" scenario.

Experiment Tracking with MLflow: Every training run is logged, versioned, and tracked. This includes parameters, performance metrics, and diagnostic plots (like feature importance and confusion matrix), ensuring full reproducibility.

Automated Model Validation & Promotion: A dedicated script (validate_model.py) automatically compares new "candidate" models against the current "production" model and promotes the winner to the MLflow Model Registry.

CI/CD Ready: The project includes a template for a GitHub Actions workflow (retrain.yml) to fully automate the testing, training, and validation process.

Interactive Streamlit Dashboard: A user-friendly "Command Center" that loads the live production model from the MLflow Registry to provide real-time delay predictions and actionable recommendations.

🛠️ Tech Stack
Backend & ML: Python, Pandas, Scikit-learn, XGBoost

MLOps & Automation: MLflow, Pydantic, GitHub Actions

Dashboard: Streamlit

Data & Configuration: YAML, Excel

Dependency Management: Conda, Pip

📁 Project Structure
SUPPLY_CHAIN_DASHBOARD/
│
├── .conda/                   # Local Conda environment files
├── .github/
│   └── workflows/
│       └── retrain.yml       # CI/CD automation workflow
├── data/
│   └── Supply chain logistics problem.xlsx  # Source data file
├── tests/
│   └── test_pipeline.py      # Automated tests for the pipeline
├── app.py                    # The Streamlit user-facing dashboard
├── config.yaml               # Central configuration for the entire project
├── delay_model.pkl           # Output file for the trained model
├── requirements.txt          # Python dependencies
├── train_model.py            # The main data engineering and model training pipeline
├── training.log              # Log file for the training process
└── validate_model.py         # Script for automated model validation and promotion

⚙️ Setup and Installation
Follow these steps to set up and run the project locally.

1. Prerequisites
Ensure you have Anaconda or Miniconda installed.

Place your Supply chain logistics problem.xlsx file inside the data/ directory.

2. Create and Activate Conda Environment
Open your terminal (Anaconda Prompt on Windows) and navigate to the project's root directory.

# Create a new Conda environment from the requirements file
# (This creates an environment named 'supplychain' with python 3.9)
conda create --name supplychain python=3.9

# Activate the new environment
conda activate supplychain

3. Install Dependencies
With your environment active, install all the required Python libraries.

pip install -r requirements.txt

🚀 How to Run the System
This project runs as a system of three interconnected components. You will need to open three separate terminals, and activate the supplychain Conda environment in each one.

Terminal 1: Start the MLflow Server (The "Brain")

This server tracks all experiments and manages the model registry. It must be running first.

# Run MLflow as a Python module for reliability
python -m mlflow server --host 127.0.0.1 --port 5000

Keep this terminal running in the background.

Terminal 2: Run the Training & Validation Pipeline (The "Factory")

This will execute the full data engineering and model training process. The first time you run this, it will create and promote your first production model.

# First, train the model
python train_model.py

# Next, validate the model and promote it to production
python validate_model.py

Terminal 3: Launch the Streamlit Dashboard (The "Storefront")

This starts the user-facing application. It will automatically connect to the MLflow server and load the best available model.

streamlit run app.py

You can now open your web browser and navigate to http://localhost:8501 to use the Command Center.
