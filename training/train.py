"""Training entry point.

Launched with ``torchrun`` (see ``README.md``). Hydra composes the config tree
under ``../configs/`` (the shared project config root, so the model config is
reused verbatim from inference) and the resolved config is passed straight into
:class:`trainer.Trainer`.
"""
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from trainer import Trainer

torch._inductor.config.fx_graph_cache = True


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    trainer = Trainer(**cfg)
    trainer.run()


if __name__ == "__main__":
    main()
