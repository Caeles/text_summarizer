from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    root_dir: Path
    source_URL: Path
    local_data_file: Path
    unzip_dir: Path

@dataclass
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: Path

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: Path
    num_train_epochs: int
    warmup_steps: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    logging_steps: int
    evaluation_strategy: str
    eval_steps: int
    save_strategy: str
    save_steps: float
    save_total_limit: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    gradient_accumulation_steps: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path