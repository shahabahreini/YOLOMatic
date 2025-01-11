import torch
from datetime import datetime
import yaml
from super_gradients.training import Trainer, models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import (
    PPYoloEPostPredictionCallback,
)
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train,
    coco_detection_yolo_format_val,
)
from super_gradients.conversion import DetectionOutputFormatMode
from super_gradients.conversion.conversion_enums import ExportQuantizationMode
from torch.utils.data import DataLoader
from clearml import Task, Logger
import os


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def __getattr__(self, name):
        return self.config.get(name)


class ClearMLCallback:
    def __init__(self, task):
        self.task = task
        self.logger = Logger.current_logger()

    def on_validation_end(self, trainer, metrics, **kwargs):
        epoch = trainer.epoch
        for key, value in metrics.items():
            self.logger.report_scalar(
                title="metrics", series=key, value=value, iteration=epoch
            )


def main():
    # Load configuration
    cfg = Config("config.yaml")

    # Set environment variables
    os.environ["CONSOLE_LOG_FILE"] = cfg.experiment["console_log_file"]

    # Setup experiment name
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"{cfg.experiment['name_prefix']}-{current_time}"

    # Initialize ClearML
    task = Task.init(
        project_name=cfg.clearml["project_name"],
        task_name=experiment_name,
        tags=cfg.clearml["tags"],
    )
    task.set_parameters(
        {"user": cfg.experiment["user"], "experiment": cfg.experiment["description"]}
    )

    # Setup trainer
    torch.backends.quantized.engine = "qnnpack"
    trainer = Trainer(
        experiment_name=experiment_name, ckpt_root_dir=cfg.experiment["checkpoint_dir"]
    )

    # Prepare dataset parameters
    dataset_params = {
        "data_dir": cfg.dataset["base_dir"],
        "classes": cfg.dataset["classes"],
    }
    task.connect(dataset_params)

    # Create data loaders
    train_data = coco_detection_yolo_format_train(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": cfg.dataset["structure"]["train"]["images"],
            "labels_dir": cfg.dataset["structure"]["train"]["labels"],
            "classes": dataset_params["classes"],
        },
        dataloader_params={
            "batch_size": cfg.training["batch_size"],
            "num_workers": cfg.training["num_workers"],
        },
    )

    val_data = coco_detection_yolo_format_val(
        dataset_params={
            "data_dir": dataset_params["data_dir"],
            "images_dir": cfg.dataset["structure"]["valid"]["images"],
            "labels_dir": cfg.dataset["structure"]["valid"]["labels"],
            "classes": dataset_params["classes"],
        },
        dataloader_params={
            "batch_size": cfg.training["batch_size"],
            "num_workers": cfg.training["num_workers"],
        },
    )

    # Prepare training parameters
    train_params = {
        "warmup_initial_lr": cfg.training["learning_rate"]["warmup_initial_lr"],
        "initial_lr": cfg.training["learning_rate"]["initial_lr"],
        "zero_weight_decay_on_bias_and_bn": cfg.training["optimizer"][
            "zero_weight_decay_on_bias_and_bn"
        ],
        "lr_warmup_epochs": cfg.training["learning_rate"]["warmup_epochs"],
        "warmup_mode": cfg.training["learning_rate"]["warmup_mode"],
        "loss_scale_value": cfg.training["mixed_precision"]["loss_scale_value"],
        "mixed_precision_enabled": cfg.training["mixed_precision"]["enabled"],
        "mixed_precision_config": {
            "loss_scale_method": cfg.training["mixed_precision"]["loss_scale_method"]
        },
        "checkpoint_params": cfg.training["checkpoint"],
        "mixed_precision_dtype": getattr(
            torch, cfg.training["mixed_precision"]["dtype"]
        ),
        "optimizer_params": {"weight_decay": cfg.training["optimizer"]["weight_decay"]},
        "ema": cfg.training["ema"]["enabled"],
        "ema_params": {
            "decay": cfg.training["ema"]["decay"],
            "decay_type": cfg.training["ema"]["decay_type"],
        },
        "max_epochs": cfg.training["max_epochs"],
        "mixed_precision": cfg.training["mixed_precision"]["enabled"],
        "loss": PPYoloELoss(
            use_static_assigner=cfg.model["loss"]["use_static_assigner"],
            num_classes=len(cfg.dataset["classes"]),
            reg_max=cfg.model["loss"]["reg_max"],
        ),
        "valid_metrics_list": [
            DetectionMetrics_050(
                score_thres=cfg.model["metrics"]["score_threshold"],
                top_k_predictions=cfg.model["metrics"]["top_k_predictions"],
                num_cls=len(cfg.dataset["classes"]),
                normalize_targets=True,
                post_prediction_callback=PPYoloEPostPredictionCallback(
                    score_threshold=cfg.model["metrics"]["post_prediction"][
                        "score_threshold"
                    ],
                    nms_top_k=cfg.model["metrics"]["post_prediction"]["nms_top_k"],
                    max_predictions=cfg.model["metrics"]["post_prediction"][
                        "max_predictions"
                    ],
                    nms_threshold=cfg.model["metrics"]["post_prediction"][
                        "nms_threshold"
                    ],
                ),
            )
        ],
        "metric_to_watch": "mAP@0.50",
    }

    task.connect(train_params)

    # Initialize model
    model = models.get(
        cfg.model["name"],
        num_classes=len(cfg.dataset["classes"]),
        pretrained_weights=cfg.model["pretrained_weights"],
    )

    # Train model
    clearml_callback = ClearMLCallback(task)
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data,
        callbacks=[clearml_callback],
    )

    # Export model
    try:
        export_result = model.export(
            cfg.export["output_name"],
            output_predictions_format=DetectionOutputFormatMode.FLAT_FORMAT,
            quantization_mode=ExportQuantizationMode.INT8,
        )
        print(export_result)
        Logger.current_logger().report_text(f"Export result: {export_result}")
    except Exception as e:
        print(f"Export failed: {e}")

    # Calibration
    dummy_calibration_dataset = [
        torch.randn((3, 640, 640), dtype=torch.float32)
        for _ in range(cfg.export["calibration"]["num_samples"])
    ]
    dummy_calibration_loader = DataLoader(
        dummy_calibration_dataset,
        batch_size=cfg.export["calibration"]["batch_size"],
        num_workers=cfg.export["calibration"]["num_workers"],
    )


if __name__ == "__main__":
    main()
