# reorganize_project.py
import os
import shutil
from pathlib import Path

# Create directory structure
dirs = {
    "notebooks": ["01_data_exploration.ipynb", "02_dataset_integration.ipynb", 
                 "03_nutrition_database.ipynb", "04_data_pipeline.ipynb", 
                 "05_model_training.ipynb"],
    "data": ["food101_df.pkl", "food101_processed.pkl", "food101_nutrition_database.json", 
            "food101_label_mappings.json", "food101_train_dataset", "food101_val_dataset"],
    "models": ["food101_model_final*.h5", "food101_model.tflite", "food101_model_quantized.tflite"]
}

# Create directories
for directory in dirs.keys():
    os.makedirs(directory, exist_ok=True)
    print(f"Created directory: {directory}")

# Move files to appropriate directories
for directory, files in dirs.items():
    for file_pattern in files:
        # Handle wildcards
        if "*" in file_pattern:
            import glob
            matching_files = glob.glob(file_pattern)
            for file in matching_files:
                if os.path.exists(file):
                    dest = os.path.join(directory, os.path.basename(file))
                    shutil.move(file, dest)
                    print(f"Moved {file} to {dest}")
        else:
            if os.path.exists(file_pattern):
                dest = os.path.join(directory, file_pattern)
                shutil.move(file_pattern, dest)
                print(f"Moved {file_pattern} to {dest}")

# Update import paths in notebooks
def update_notebook_paths(notebook_path, old_prefix="", new_prefix=""):
    import json
    
    notebook_path = os.path.join("notebooks", notebook_path)
    if not os.path.exists(notebook_path):
        return
        
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    modified = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            for i, line in enumerate(cell['source']):
                if 'read_pickle(' in line or 'to_pickle(' in line or 'open(' in line:
                    new_line = line.replace(f'"{old_prefix}', f'"data/{new_prefix}')
                    new_line = new_line.replace(f"'{old_prefix}", f"'data/{new_prefix}")
                    if new_line != line:
                        cell['source'][i] = new_line
                        modified = True
    
    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        print(f"Updated paths in {notebook_path}")

# Update paths in each notebook
for notebook in dirs["notebooks"]:
    update_notebook_paths(notebook)

print("Project reorganization complete!")