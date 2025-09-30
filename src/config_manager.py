#!/usr/bin/env python3
"""
Configuration Management System

Hydra-based configuration system for flexible experiment management.
Supports YAML configs with command-line overrides for maximum flexibility.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field
import logging
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.config_store import ConfigStore
import torch

logger = logging.getLogger(__name__)

@dataclass
class ExperimentConfig:
    """Experiment metadata configuration"""
    name: str = "unstable_singularity_detection"
    version: str = "1.0.0"
    author: str = "Flamehaven Research"
    description: str = "DeepMind fluid dynamics implementation"
    tags: List[str] = field(default_factory=list)

@dataclass
class GlobalConfig:
    """Global system configuration"""
    precision: str = "float64"
    device: str = "auto"
    random_seed: int = 42
    log_level: str = "INFO"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

@dataclass
class PerformanceConfig:
    """Performance optimization settings"""
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False
    compile: bool = False

@dataclass
class LoggingConfig:
    """Logging system configuration"""
    enable_mlflow: bool = False
    enable_wandb: bool = False
    log_every_n_steps: int = 100
    save_frequency: int = 1000

@dataclass
class ReproducibilityConfig:
    """Reproducibility settings"""
    deterministic: bool = True
    benchmark: bool = False
    set_torch_deterministic: bool = True

class ConfigManager:
    """
    Central configuration management system using Hydra

    Features:
    - YAML-based configuration files
    - Command-line parameter overrides
    - Environment-specific configurations
    - Automatic validation and type checking
    """

    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager

        Args:
            config_dir: Directory containing config files. If None, uses default.
        """
        if config_dir is None:
            # Get project root directory
            project_root = Path(__file__).parent.parent
            config_dir = str(project_root / "configs")

        self.config_dir = Path(config_dir).resolve()
        self.config_store = ConfigStore.instance()
        self._register_schemas()

        logger.info(f"Initialized ConfigManager with config_dir: {self.config_dir}")

    def _register_schemas(self):
        """Register configuration schemas with Hydra"""
        cs = self.config_store

        # Register dataclass schemas
        cs.store(name="experiment_schema", node=ExperimentConfig)
        cs.store(name="global_schema", node=GlobalConfig)
        cs.store(name="performance_schema", node=PerformanceConfig)
        cs.store(name="logging_schema", node=LoggingConfig)
        cs.store(name="reproducibility_schema", node=ReproducibilityConfig)

    def load_config(self,
                   config_name: str = "base",
                   overrides: Optional[list] = None,
                   return_hydra_config: bool = False) -> Union[DictConfig, Any]:
        """
        Load configuration with optional overrides

        Args:
            config_name: Name of the base config file (without .yaml)
            overrides: List of parameter overrides (e.g., ["training.epochs=5000"])
            return_hydra_config: If True, return raw Hydra config

        Returns:
            Loaded and validated configuration
        """
        if overrides is None:
            overrides = []

        try:
            # Initialize Hydra with config directory
            with initialize_config_dir(config_dir=str(self.config_dir)):
                cfg = compose(config_name=f"{config_name}.yaml", overrides=overrides)

                # Apply configuration
                self._apply_global_config(cfg.get("global", {}))

                if return_hydra_config:
                    return cfg
                else:
                    return self._convert_to_objects(cfg)

        except Exception as e:
            logger.error(f"Failed to load config '{config_name}': {e}")
            raise

    def _apply_global_config(self, global_cfg: DictConfig):
        """Apply global configuration settings"""

        # Set random seed
        if "random_seed" in global_cfg:
            seed = global_cfg.random_seed
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            logger.info(f"Set random seed to {seed}")

        # Set device
        if "device" in global_cfg:
            device = global_cfg.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            logger.info(f"Using device: {self.device}")

        # Set precision
        if "precision" in global_cfg:
            precision = global_cfg.precision
            if precision == "float64":
                torch.set_default_dtype(torch.float64)
            elif precision == "float32":
                torch.set_default_dtype(torch.float32)
            logger.info(f"Set default precision to {precision}")

        # Create output directories
        for dir_key in ["output_dir", "checkpoint_dir"]:
            if dir_key in global_cfg:
                dir_path = Path(global_cfg[dir_key])
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")

        # Set logging level
        if "log_level" in global_cfg:
            level = getattr(logging, global_cfg.log_level.upper())
            logging.getLogger().setLevel(level)

    def _convert_to_objects(self, cfg: DictConfig) -> Dict[str, Any]:
        """Convert Hydra config to instantiated objects where specified"""
        result = {}

        for key, value in cfg.items():
            if isinstance(value, DictConfig):
                if "_target_" in value:
                    # This is an object to instantiate
                    try:
                        obj = hydra.utils.instantiate(value)
                        result[key] = obj
                        logger.info(f"Instantiated {key}: {value._target_}")
                    except Exception as e:
                        logger.warning(f"Failed to instantiate {key}: {e}")
                        result[key] = value
                else:
                    # Recursive conversion
                    result[key] = self._convert_to_objects(value)
            else:
                result[key] = value

        return result

    def save_config(self, cfg: DictConfig, save_path: str):
        """Save configuration to file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, 'w') as f:
            OmegaConf.save(cfg, f)

        logger.info(f"Saved configuration to {save_path}")

    def validate_config(self, cfg: DictConfig) -> bool:
        """Validate configuration against schemas"""
        try:
            # Basic validation checks
            required_sections = ["global", "experiment"]
            for section in required_sections:
                if section not in cfg:
                    logger.error(f"Missing required section: {section}")
                    return False

            # Validate precision setting
            if "precision" in cfg.get("global", {}):
                valid_precisions = ["float32", "float64"]
                if cfg.get("global", {}).get("precision") not in valid_precisions:
                    logger.error(f"Invalid precision: {cfg.get('global', {}).get('precision')}")
                    return False

            # Validate device setting
            if "device" in cfg.get("global", {}):
                valid_devices = ["auto", "cpu", "cuda"]
                if cfg.get("global", {}).get("device") not in valid_devices:
                    logger.error(f"Invalid device: {cfg.get('global', {}).get('device')}")
                    return False

            logger.info("Configuration validation passed")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def create_experiment_config(self,
                               base_config: str = "base",
                               experiment_name: str = None,
                               overrides: list = None) -> DictConfig:
        """Create a new experiment configuration"""

        # Load base config
        cfg = self.load_config(base_config, overrides, return_hydra_config=True)

        # Update experiment metadata
        if experiment_name:
            cfg.experiment.name = experiment_name

        # Add timestamp
        from datetime import datetime
        cfg.experiment.timestamp = datetime.now().isoformat()

        # Validate
        if not self.validate_config(cfg):
            raise ValueError("Configuration validation failed")

        return cfg

    def get_config_schema(self, schema_name: str) -> Any:
        """Get registered configuration schema"""
        try:
            return self.config_store.get_schema(schema_name)
        except Exception as e:
            logger.error(f"Schema '{schema_name}' not found: {e}")
            return None

# Global config manager instance
config_manager = ConfigManager()

def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    return config_manager

# Convenience functions
def load_config(config_name: str = "base", overrides: list = None):
    """Load configuration using global manager"""
    return config_manager.load_config(config_name, overrides)

def create_experiment_config(experiment_name: str,
                           base_config: str = "base",
                           overrides: list = None):
    """Create experiment configuration using global manager"""
    return config_manager.create_experiment_config(
        base_config, experiment_name, overrides
    )

if __name__ == "__main__":
    # Example usage
    print("Testing Configuration Manager...")

    # Load base configuration
    cfg = load_config("base")
    print(f"Loaded base config with experiment: {cfg['experiment'].name}")

    # Create experiment configuration
    exp_cfg = create_experiment_config(
        "test_experiment",
        overrides=["global.random_seed=12345", "detector.precision_target=1e-14"]
    )
    print(f"Created experiment config: {exp_cfg.experiment.name}")

    print("Configuration manager test completed!")