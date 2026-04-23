import os
from datetime import datetime
from typing import Optional

import yaml
from clearml import Logger, Task

try:
    from src.cli.run import get_user_choice
    from src.utils.ml_dependencies import import_module_or_raise, import_torch
    from src.utils.training_preflight import resolve_training_device
except ImportError:
    from utils.ml_dependencies import import_module_or_raise, import_torch
    from utils.training_preflight import resolve_training_device

    try:
        from cli.run import get_user_choice
    except ImportError:

        def get_user_choice(
            options,
            allow_back=False,
            title="Select an Option",
            text="Use ↑↓ keys to navigate, Enter to select:",
            model_data=None,
        ):
            return options[0]


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


def initialize_clearml_task(project_name, task_name, tags):
    try:
        return Task.init(
            project_name=project_name,
            task_name=task_name,
            tags=tags,
        )
    except Exception as error:
        print(f"ClearML is not configured: {error}")
        selection = get_user_choice(
            ["Continue Without ClearML", "Cancel Training"],
            title="ClearML Setup Required",
            text="Use ↑↓ keys to choose whether to continue without ClearML or cancel training:",
        )
        if selection == "Cancel Training":
            return False
        return None


def main(config_path: Optional[str] = None):
    cfg = Config(config_path or "config.yaml")
    requested_device = cfg.training.get("device")
    device_resolution = resolve_training_device(requested_device, prefer_gpu=True)
    if device_resolution.cancelled:
        print("Training cancelled.")
        return

    torch = import_torch()
    training_module = import_module_or_raise("super_gradients.training")
    dataloaders_module = import_module_or_raise(
        "super_gradients.training.dataloaders.dataloaders"
    )
    losses_module = import_module_or_raise("super_gradients.training.losses")
    metrics_module = import_module_or_raise("super_gradients.training.metrics")
    pp_yolo_module = import_module_or_raise(
        "super_gradients.training.models.detection_models.pp_yolo_e"
    )
    conversion_module = import_module_or_raise("super_gradients.conversion")
    conversion_enums_module = import_module_or_raise(
        "super_gradients.conversion.conversion_enums"
    )
    data_loader_module = import_module_or_raise("torch.utils.data")

    trainer_class = training_module.Trainer
    models = training_module.models
    coco_detection_yolo_format_train = (
        dataloaders_module.coco_detection_yolo_format_train
    )
    coco_detection_yolo_format_val = dataloaders_module.coco_detection_yolo_format_val
    pp_yolo_e_loss = losses_module.PPYoloELoss
    detection_metrics = metrics_module.DetectionMetrics_050
    pp_yolo_callback = pp_yolo_module.PPYoloEPostPredictionCallback
    detection_output_format_mode = conversion_module.DetectionOutputFormatMode
    export_quantization_mode = conversion_enums_module.ExportQuantizationMode
    data_loader_class = data_loader_module.DataLoader

    os.environ["CONSOLE_LOG_FILE"] = cfg.experiment["console_log_file"]

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
    experiment_name = f"{cfg.experiment['name_prefix']}-{current_time}"

    task = initialize_clearml_task(
        project_name=cfg.clearml["project_name"],
        task_name=experiment_name,
        tags=cfg.clearml["tags"],
    )
    if task is False:
        print("Training cancelled.")
        return
    if task is not None:
        task.set_parameters(
            {
                "user": cfg.experiment["user"],
                "experiment": cfg.experiment["description"],
            }
        )

    torch.backends.quantized.engine = "qnnpack"
    trainer = trainer_class(
        experiment_name=experiment_name, ckpt_root_dir=cfg.experiment["checkpoint_dir"]
    )

    dataset_params = {
        "data_dir": cfg.dataset["base_dir"],
        "classes": cfg.dataset["classes"],
    }
    if task is not None:
        task.connect(dataset_params)

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
        "loss": pp_yolo_e_loss(
            use_static_assigner=cfg.model["loss"]["use_static_assigner"],
            num_classes=len(cfg.dataset["classes"]),
            reg_max=cfg.model["loss"]["reg_max"],
        ),
        "valid_metrics_list": [
            detection_metrics(
                score_thres=cfg.model["metrics"]["score_threshold"],
                top_k_predictions=cfg.model["metrics"]["top_k_predictions"],
                num_cls=len(cfg.dataset["classes"]),
                normalize_targets=True,
                post_prediction_callback=pp_yolo_callback(
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

    if task is not None:
        task.connect(train_params)

    model = models.get(
        cfg.model["name"],
        num_classes=len(cfg.dataset["classes"]),
        pretrained_weights=cfg.model["pretrained_weights"],
    )

    callbacks = [ClearMLCallback(task)] if task is not None else []
    trainer.train(
        model=model,
        training_params=train_params,
        train_loader=train_data,
        valid_loader=val_data,
        callbacks=callbacks,
    )

    try:
        export_result = model.export(
            cfg.export["output_name"],
            output_predictions_format=detection_output_format_mode.FLAT_FORMAT,
            quantization_mode=export_quantization_mode.INT8,
        )
        print(export_result)
        Logger.current_logger().report_text(f"Export result: {export_result}")
    except Exception as e:
        print(f"Export failed: {e}")

    dummy_calibration_dataset = [
        torch.randn((3, 640, 640), dtype=torch.float32)
        for _ in range(cfg.export["calibration"]["num_samples"])
    ]
    data_loader_class(
        dummy_calibration_dataset,
        batch_size=cfg.export["calibration"]["batch_size"],
        num_workers=cfg.export["calibration"]["num_workers"],
    )


if __name__ == "__main__":
    main()
