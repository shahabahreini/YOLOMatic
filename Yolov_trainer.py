import os
import yaml
from datetime import datetime
from ultralytics import YOLO
from clearml import Task


def load_dataset_config(dataset_name):
    """Load dataset configuration from data.yaml file."""
    # Look for data.yaml in the dataset folder

    # Get the absolute path for the dataset directory
    dataset_path = os.path.abspath(os.path.join("datasets", dataset_name))
    # Get the absolute path for the data.yaml file
    data_yaml_path = os.path.abspath(os.path.join(dataset_path, "data.yaml"))

    if not os.path.exists(data_yaml_path):
        raise FileNotFoundError(f"data.yaml not found in {dataset_path}")

    with open(data_yaml_path, "r") as file:
        dataset_config = yaml.safe_load(file)

    # Convert relative paths to absolute paths within the dataset directory
    base_path = os.path.dirname(data_yaml_path)
    for key in ["train", "val", "test"]:
        # Strip any leading slashes and join with base path
        relative_path = dataset_config[key].lstrip("/")
        dataset_config[key] = os.path.join(base_path, relative_path)

    return dataset_config, data_yaml_path, dataset_path


def verify_directories(dataset_config):
    """Verify that all required directories exist."""
    missing_dirs = []
    for dir_type, dir_path in [
        ("Training", dataset_config["train"]),
        ("Validation", dataset_config["val"]),
        ("Test", dataset_config["test"]),
    ]:
        if not os.path.exists(dir_path):
            missing_dirs.append(f"{dir_type} directory: {dir_path}")

    if missing_dirs:
        raise FileNotFoundError(
            "The following directories are missing:\n" + "\n".join(missing_dirs)
        )


def select_config():
    """Select configuration file from configs directory."""
    config_folder = "configs"
    if not os.path.exists(config_folder):
        raise FileNotFoundError(f"Config folder '{config_folder}' not found.")

    yaml_files = [f for f in os.listdir(config_folder) if f.endswith(".yaml")]

    if not yaml_files:
        raise FileNotFoundError("No YAML files found in the configs folder.")

    print("\nAvailable configuration files:")
    for i, file in enumerate(yaml_files, 1):
        print(f"{i}. {file}")

    while True:
        try:
            choice = int(input("\nEnter the number of the configuration file to use: "))
            if 1 <= choice <= len(yaml_files):
                selected_file = os.path.join(config_folder, yaml_files[choice - 1])
                print(f"\nSelected configuration: {yaml_files[choice - 1]}")
                return selected_file
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def verify_model_file(model_name):
    """Verify if the model file exists and load it, or download if it's a YOLO model."""
    try:
        # Check if it's a YOLO model (e.g., yolov8n, yolov8s, etc.)
        if model_name.lower().startswith("yolo"):
            # Convert model name to correct format (e.g., YOLO11n -> yolov8n)
            # Extract the version number and size
            version = model_name[4:6]  # Extract '11' from 'YOLO11n'
            size = model_name[-1].lower()  # Extract 'n' and convert to lowercase

            # Convert to standard YOLO format
            standard_name = f"yolo{version}{size}"
            print(f"\nConverting model name {model_name} to {standard_name}")

            # Try to load or download the model
            model = YOLO(standard_name + ".pt")
            print(f"\nSuccessfully loaded/downloaded model: {standard_name}")
            return model
        else:
            # Handle custom model files
            if not os.path.exists(model_name):
                raise FileNotFoundError(f"Model file {model_name} not found")
            model = YOLO(model_name)
            print(f"\nSuccessfully loaded model: {model_name}")
            return model

    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        return None


def capitalize_first_five(s):
    """Capitalize first five characters of a string."""
    if len(s) < 4:
        return s.upper()
    else:
        return s[:4].upper() + s[4:]


def print_config_summary(config, dataset_config):
    """Print a summary of the loaded configurations."""
    print("\nConfiguration Summary:")
    print("=====================")
    print(f"Model: {config['settings']['model']}")
    print(f"Dataset: {config['settings']['dataset']}")
    print(f"Project Name: {config['clearml']['project_name']}")

    print("\nTraining Parameters:")
    print(f"Batch Size: {config['training']['batch']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Device: {config['training']['device']}")

    print("\nDataset Information:")
    print(f"Number of Classes: {dataset_config['nc']}")
    print(f"Classes: {dataset_config['names']}")
    if "roboflow" in dataset_config:
        print("\nRoboflow Information:")
        print(f"Workspace: {dataset_config['roboflow']['workspace']}")
        print(f"Project: {dataset_config['roboflow']['project']}")
        print(f"Version: {dataset_config['roboflow']['version']}")


# Main execution
def main():
    try:
        # Select configuration file
        config_file = select_config()
        if config_file is None:
            print("No configuration file selected. Exiting.")
            return

        # Load configuration from selected YAML file
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Extract parameters from config
        settings = config["settings"]
        clearml_settings = config["clearml"]
        training_params = config["training"]
        export_params = config["export"]

        # Load dataset configuration
        dataset_config, data_yaml_path, dataset_path = load_dataset_config(
            settings["dataset"]
        )

        # Print configuration summary
        print_config_summary(config, dataset_config)

        # Verify dataset directories
        # verify_directories(dataset_config)

        # Verify and load the model
        model = verify_model_file(settings["model"])
        if model is None:
            print("Model verification failed. Exiting.")
            return

        # Initialize ClearML Task
        current_time = datetime.now().strftime(clearml_settings["task_name_format"])
        task_name = f"{settings['model']}-{current_time}"

        task = Task.init(
            project_name=clearml_settings["project_name"],
            task_name=task_name,
            tags=[capitalize_first_five(settings["model"])],
        )

        # Log configurations to ClearML
        task.connect(
            {
                "dataset_config": dataset_config,
                "training_params": training_params,
                "export_params": export_params,
            }
        )

        # Train the model
        print("\nStarting training...")
        os.environ["YOLO_DATASET_DIR"] = os.path.abspath("datasets/Oxford Pets")
        model.train(data=data_yaml_path, **training_params)

        # Evaluate the model
        print("\nStarting validation...")
        metrics = model.val(data=data_yaml_path)

        # Export the model
        print("\nExporting model...")
        model.export(**export_params)

        print("\nTraining completed successfully!")

    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
        import traceback

        print(traceback.format_exc())
    finally:
        # Ensure ClearML task is closed
        if "task" in locals():
            task.close()


if __name__ == "__main__":
    main()
