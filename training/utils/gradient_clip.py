"""Per-module gradient clipping for DDP training."""
from typing import Optional

import torch.nn as nn


class GradientClipper:
    """Clip gradients of specific sub-modules, verifying full coverage.

    Each config entry is ``{module_name: str | list[str], max_norm: float,
    norm_type: int}``. ``setup_clipping`` collects the trainable parameters
    matched by each entry and asserts that *every* trainable parameter is
    covered by some entry (so nothing silently escapes clipping).
    """

    def __init__(self, configs, *args, **kwargs):
        self.configs = []
        self.params_to_clip_by_config = None
        self.is_initialized = False

        for config in configs:
            module_names = config["module_name"]
            if isinstance(module_names, str):
                module_names = [module_names]

            self.configs.append({
                "module_names": module_names,
                "max_norm": float(config["max_norm"]) if config["max_norm"] is not None else None,
                "norm_type": config.get("norm_type", 2),
            })

    def setup_clipping(self, model: nn.Module) -> None:
        """Resolve the parameters to clip once, up front, and validate coverage."""
        params_to_clip_by_config = []
        all_clipped_params = set()

        for config in self.configs:
            current_config_params = []
            for name, param in model.named_parameters():
                if param.requires_grad:
                    for module_name in config["module_names"]:
                        if module_name in name:
                            current_config_params.append(param)
                            all_clipped_params.add(param)
                            break
            params_to_clip_by_config.append((config, current_config_params))

        remaining_params = [
            name for name, param in model.named_parameters()
            if param.requires_grad and param not in all_clipped_params
        ]
        if len(remaining_params) > 0:
            print(f"Found {len(remaining_params)} parameters that won't be clipped")
            print(remaining_params)
            raise ValueError("Some parameters are not configured for gradient clipping")

        self.params_to_clip_by_config = params_to_clip_by_config
        self.is_initialized = True

    def __call__(self, model: nn.Module) -> Optional[dict]:
        """Clip gradients and return the per-config gradient norms."""
        if not self.is_initialized:
            raise RuntimeError("GradientClipper must be initialized with setup_clipping() before use")

        grad_norms = {}
        for config, params_to_clip in self.params_to_clip_by_config:
            if not params_to_clip or config["max_norm"] is None:
                continue

            grad_norm = nn.utils.clip_grad_norm_(
                params_to_clip,
                max_norm=config["max_norm"],
                norm_type=config["norm_type"],
            )
            if grad_norm is None:
                continue
            grad_norms[",".join(config["module_names"])] = grad_norm.item()

        return grad_norms
