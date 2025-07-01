

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "mlproject"

list_of_files = [
    f"src/__init__.py",
    f"src/{project_name}/__init__.py",  
    f"src/{project_name}/data_collection.py",
    f"src/{project_name}/financial_data.py",
    f"src/{project_name}/stock_sentiment.py",
    f"src/{project_name}/model_building.py",
    f"src/{project_name}/model_evaluation.py",
    f"src/{project_name}/utilities.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/config/config.yaml",

    f"src/pipeline/__init__.py",
    f"src/pipeline/predict_pipeline.py",
    f"src/pipeline/train_pipeline.py",
    f"src/exception.py",

    f"notebooks/Model_Training.ipynb",
    "notebooks/Data",

    "requirements.txt"
]

for file_path in list_of_files:
    file_path = Path(file_path)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created directory: {file_path.parent}")

    if not file_path.exists() or file_path.stat().st_size == 0:
        file_path.touch()
        logging.info(f"Created empty file: {file_path}")
    else:
        logging.info(f"File already exists: {file_path}")
