import os
from ruamel.yaml import YAML
from termcolor import colored, cprint
from shutil import copy2
from datetime import datetime
from terminaltables import SingleTable

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_stylized_header(text):
    width = 60
    padding = (width - len(text)) // 2
    border = "+" + "=" * (width - 2) + "+"
    
    print("\n" + colored(border, "cyan"))
    print(colored("|" + " " * (width - 2) + "|", "cyan"))
    print(colored("|" + " " * padding + text + " " * (width - 2 - padding - len(text)) + "|", "cyan", attrs=["bold"]))
    print(colored("|" + " " * (width - 2) + "|", "cyan"))
    print(colored(border, "cyan") + "\n")

def print_model_info(model_data):
    headers = list(model_data[0].keys())
    table_data = [headers]
    for row in model_data:
        table_data.append([row[col] for col in headers])

    table = SingleTable(table_data)
    table.inner_heading_row_border = True
    table.inner_row_border = True
    table.justify_columns = {i: 'center' for i in range(len(headers))}

    # Color the header row
    for i in range(len(headers)):
        table.table_data[0][i] = colored(table.table_data[0][i], 'cyan', attrs=['bold'])

    # Color alternating rows
    for i in range(1, len(table_data)):
        color = 'yellow' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table.table_data[i][j] = colored(str(table.table_data[i][j]), color)

    # Set a title for the table
    table.title = colored(f" {headers[0]} Comparison ", 'green', attrs=['bold'])

    print(table.table)

def get_user_choice(options):
    for i, option in enumerate(options, 1):
        cprint(f"{i}. {option}", "yellow")
    
    while True:
        try:
            choice = input(colored("\nEnter the number of your choice: ", "green", attrs=["bold"]))
            choice = int(choice)
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                cprint("❌ Invalid choice. Please try again.", "red", attrs=["bold"])
        except ValueError:
            cprint("❌ Invalid input. Please enter a number.", "red", attrs=["bold"])

def get_folder_size(folder_path):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    return total_size

def format_size(size_in_bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024.0:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024.0

def list_datasets():
    datasets_folder = 'datasets'
    if not os.path.exists(datasets_folder):
        os.makedirs(datasets_folder)
        cprint(f"✨ '{datasets_folder}' folder created. Please add COCO or any other compatible dataset into it.", "yellow")
        return None

    datasets = []
    for folder in os.listdir(datasets_folder):
        folder_path = os.path.join(datasets_folder, folder)
        if os.path.isdir(folder_path):
            size = get_folder_size(folder_path)
            datasets.append({"name": folder, "size": format_size(size)})

    if not datasets:
        cprint(f"❌ No datasets found in '{datasets_folder}' folder.", "red")
        return None

    print_stylized_header("Available Datasets")
    print_model_info(datasets)
    
    dataset_names = [d["name"] for d in datasets]
    return get_user_choice(dataset_names)

def print_summary(model_type, dataset):
    print_stylized_header("Summary of Selected Options")
    print(f"Model Type: {colored(model_type, 'yellow', attrs=['bold'])}")
    print(f"Dataset: {colored(dataset, 'yellow', attrs=['bold'])}")

def update_config(model_type, dataset):
    config_file = 'config.yaml'
    config_history_folder = 'config_history'
    
    if not os.path.exists(config_history_folder):
        os.makedirs(config_history_folder)
    
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.indent(mapping=2, sequence=4, offset=2)

    with open(config_file, 'r') as file:
        config = yaml.load(file)

    config['settings']['model_type'] = model_type
    config['settings']['dataset'] = dataset

    with open(config_file, 'w') as file:
        yaml.dump(config, file)

    backup_file = os.path.join(config_history_folder, f'config_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml')
    copy2(config_file, backup_file)
    cprint(f"\n✅ Backup created: {backup_file}", "green")

    cprint(f"✅ Config updated with model type: {model_type} and dataset: {dataset}", "green")

# Define all model data in a dictionary
model_data_dict = {
    "yolov11": [
        {"Model": "YOLO11n", "Input Size": 640, "APval": 39.5, "FLOPs (B)": 6.5, "Latency (ms)": 1.5, "params (M)": 2.6, "CPU ONNX (ms)": 56.1},
        {"Model": "YOLO11s", "Input Size": 640, "APval": 47.0, "FLOPs (B)": 21.5, "Latency (ms)": 2.5, "params (M)": 9.4, "CPU ONNX (ms)": 90.0},
        {"Model": "YOLO11m", "Input Size": 640, "APval": 51.5, "FLOPs (B)": 68.0, "Latency (ms)": 4.7, "params (M)": 20.1, "CPU ONNX (ms)": 183.2},
        {"Model": "YOLO11l", "Input Size": 640, "APval": 53.4, "FLOPs (B)": 86.9, "Latency (ms)": 6.2, "params (M)": 25.3, "CPU ONNX (ms)": 238.6},
        {"Model": "YOLO11x", "Input Size": 640, "APval": 54.7, "FLOPs (B)": 194.9, "Latency (ms)": 11.3, "params (M)": 56.9, "CPU ONNX (ms)": 462.8},
    ],
    "yolov10": [
        {"Model": "YOLOv10N", "Input Size": 640, "APval": 38.5, "FLOPs (G)": 6.7, "Latency (ms)": 1.84},
        {"Model": "YOLOv10S", "Input Size": 640, "APval": 46.3, "FLOPs (G)": 21.6, "Latency (ms)": 2.49},
        {"Model": "YOLOv10M", "Input Size": 640, "APval": 51.1, "FLOPs (G)": 59.1, "Latency (ms)": 4.74},
        {"Model": "YOLOv10B", "Input Size": 640, "APval": 52.5, "FLOPs (G)": 92.0, "Latency (ms)": 5.74},
        {"Model": "YOLOv10L", "Input Size": 640, "APval": 53.2, "FLOPs (G)": 120.3, "Latency (ms)": 7.28},
        {"Model": "YOLOv10X", "Input Size": 640, "APval": 54.4, "FLOPs (G)": 160.4, "Latency (ms)": 10.70},
    ],
    "yolov9": [
        {"Model": "YOLOv9t", "size (pixels)": 640, "mAPval 50-95": 38.3, "mAPval 50": 53.1, "params (M)": 2.0, "FLOPs (B)": 7.7},
        {"Model": "YOLOv9s", "size (pixels)": 640, "mAPval 50-95": 46.8, "mAPval 50": 63.4, "params (M)": 7.2, "FLOPs (B)": 26.7},
        {"Model": "YOLOv9m", "size (pixels)": 640, "mAPval 50-95": 51.4, "mAPval 50": 68.1, "params (M)": 20.1, "FLOPs (B)": 76.8},
        {"Model": "YOLOv9c", "size (pixels)": 640, "mAPval 50-95": 53.0, "mAPval 50": 70.2, "params (M)": 25.5, "FLOPs (B)": 102.8},
        {"Model": "YOLOv9e", "size (pixels)": 640, "mAPval 50-95": 55.6, "mAPval 50": 72.8, "params (M)": 58.1, "FLOPs (B)": 192.5},
    ],
    "yolov8": [
        {"Model": "YOLOv8n", "size (pixels)": 640, "mAPval 50-95": 37.3, "Speed CPU ONNX (ms)": 80.4, "Speed A100 TensorRT (ms)": 0.99, "params (M)": 3.2, "FLOPs (B)": 8.7},
        {"Model": "YOLOv8s", "size (pixels)": 640, "mAPval 50-95": 44.9, "Speed CPU ONNX (ms)": 128.4, "Speed A100 TensorRT (ms)": 1.20, "params (M)": 11.2, "FLOPs (B)": 28.6},
        {"Model": "YOLOv8m", "size (pixels)": 640, "mAPval 50-95": 50.2, "Speed CPU ONNX (ms)": 234.7, "Speed A100 TensorRT (ms)": 1.83, "params (M)": 25.9, "FLOPs (B)": 78.9},
        {"Model": "YOLOv8l", "size (pixels)": 640, "mAPval 50-95": 52.9, "Speed CPU ONNX (ms)": 375.2, "Speed A100 TensorRT (ms)": 2.39, "params (M)": 43.7, "FLOPs (B)": 165.2},
        {"Model": "YOLOv8x", "size (pixels)": 640, "mAPval 50-95": 53.9, "Speed CPU ONNX (ms)": 479.1, "Speed A100 TensorRT (ms)": 3.53, "params (M)": 68.2, "FLOPs (B)": 257.8},
    ],
    "yolox": [
        {"Model": "YOLOX-S", "Params (M)": 9.0, "FLOPs (G)": 26.8, "Size (pixels)": 640, "FPS": 102, "APtest / val 50-95": "40.5% / 40.5%"},
        {"Model": "YOLOX-M", "Params (M)": 25.3, "FLOPs (G)": 73.8, "Size (pixels)": 640, "FPS": 81, "APtest / val 50-95": "47.2% / 46.9%"},
        {"Model": "YOLOX-L", "Params (M)": 54.2, "FLOPs (G)": 155.6, "Size (pixels)": 640, "FPS": 69, "APtest / val 50-95": "50.1% / 49.7%"},
        {"Model": "YOLOX-X", "Params (M)": 99.1, "FLOPs (G)": 281.9, "Size (pixels)": 640, "FPS": 58, "APtest / val 50-95": "51.5% / 51.1%"},
    ],
}

def main():
    while True:
        clear_screen()
        print_stylized_header("YOLO Model Selector")
        options = ["YOLOv11", "YOLOv10", "YOLOv9", "YOLOv8", "YOLOX", "Exit"]
        choice = get_user_choice(options)

        if choice == "Exit":
            print_stylized_header("Thank you for using YOLO Model Selector")
            cprint("👋 Goodbye!", "cyan", attrs=["bold"])
            break

        clear_screen()
        print_stylized_header(f"{choice} Models")

        model_data = model_data_dict[choice.lower()]
        print_model_info(model_data)

        model_choice = get_user_choice([model["Model"] for model in model_data])
        
        clear_screen()
        print_stylized_header(f"Properties of {model_choice}")
        model_info = next(model for model in model_data if model["Model"] == model_choice)
        print_model_info([model_info])

        dataset_choice = list_datasets()
        if dataset_choice:
            print_summary(model_choice, dataset_choice)
            update_config(model_choice, dataset_choice)
        
        input(colored("\nPress Enter to continue...", "green", attrs=["bold"]))

if __name__ == "__main__":
    main()
