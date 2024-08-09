import os
import yaml
from datetime import datetime
from ultralytics import YOLO
from clearml import Task

def capitalize_first_five(s):
    if len(s) < 4:
        return s.upper()
    else:
        return s[:4].upper() + s[4:]

def select_config():
    config_history_folder = 'config_history'
    yaml_files = [f for f in os.listdir(config_history_folder) if f.endswith('.yaml')]
    
    if not yaml_files:
        print("No YAML files found in the config_history folder.")
        return None

    print("Available configuration files:")
    for i, file in enumerate(yaml_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("Enter the number of the configuration file to use: "))
            if 1 <= choice <= len(yaml_files):
                return os.path.join(config_history_folder, yaml_files[choice - 1])
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Select configuration file
config_file = select_config()
if config_file is None:
    print("No configuration file selected. Exiting.")
    exit()

# Load configuration from selected YAML file
with open(config_file, 'r') as config_file:
    config = yaml.safe_load(config_file)

# Extract parameters from config
settings = config['settings']
clearml_settings = config['clearml']
training_params = config['training']
model_params = config['model']
export_params = config['export']

# Get the current time and format it using the format from config
current_time = datetime.now().strftime(clearml_settings['task_name_format'])

# Generate the task_name based on the format
task_name = f"{settings['model_type']}-{current_time}"

# Initialize ClearML Task with the generated task_name
task = Task.init(project_name=clearml_settings['project_name'], task_name=task_name.capitalize(), 
                 tags=[capitalize_first_five(settings['model_type'])])

# Update model parameters with the actual dataset name
model_params['data_dir'] = model_params['data_dir'].format(dataset=settings['dataset'])

# Log dataset parameters to ClearML
task.connect(model_params)

# Construct absolute paths for the dataset directories
data_dir = os.path.abspath(model_params['data_dir'])
train_images_dir = os.path.join(data_dir, model_params['train_images_dir'])
val_images_dir = os.path.join(data_dir, model_params['val_images_dir'])
test_images_dir = os.path.join(data_dir, model_params['test_images_dir'])

# Check if the directories exist
if not os.path.exists(train_images_dir):
    raise FileNotFoundError(f"Training images directory not found: {train_images_dir}")
if not os.path.exists(val_images_dir):
    raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
if not os.path.exists(test_images_dir):
    raise FileNotFoundError(f"Test images directory not found: {test_images_dir}")

# Create the data.yaml file required by YOLO
data_yaml_content = f"""
train: {train_images_dir}
val: {val_images_dir}
test: {test_images_dir}

nc: {len(model_params['classes'])}
names: {model_params['classes']}
"""

data_yaml_path = os.path.join(data_dir, 'data.yaml')
with open(data_yaml_path, 'w') as file:
    file.write(data_yaml_content)

# Log data.yaml path to ClearML
task.upload_artifact('data.yaml', data_yaml_path)

# Add task_name to training parameters
training_params['name'] = task_name

# Log training parameters to ClearML
task.connect(training_params)

# Train YOLOv10 model
model = YOLO(f"{settings['model_type']}.pt")  # Load the YOLOv10 model

# Train the model
model.train(data=data_yaml_path, **training_params)

# Evaluate the model
metrics = model.val(data=data_yaml_path)

# Export the model using parameters from config
model.export(data=data_yaml_path, **export_params)

# Log export parameters to ClearML
task.connect({'export_params': export_params})

# Close the ClearML task
task.close()
