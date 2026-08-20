#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies import PreTrainedPolicy


def make_optimizer_and_scheduler(
    cfg: TrainPipelineConfig, policy: PreTrainedPolicy
) -> tuple[Optimizer, LRScheduler | None]:
    """Generates the optimizer and scheduler based on configs.

    Args:
        cfg (TrainPipelineConfig): The training config that contains optimizer and scheduler configs
        policy (PreTrainedPolicy): The policy config from which parameters and presets must be taken from.

    Returns:
        tuple[Optimizer, LRScheduler | None]: The couple (Optimizer, Scheduler). Scheduler can be `None`.
    """
    params = policy.get_optim_params() if cfg.use_policy_training_preset else policy.parameters()
    if cfg.optimizer is None:
        raise ValueError("Optimizer config is required but not provided in TrainPipelineConfig")
    # LEROBOT_BACKBONE_LR_SCALE=0.1: fine-tune the pretrained vision backbone at a
    # scaled LR while the rest of the network trains at the regular LR (DP paper
    # section 5.4 prescription: "10x smaller vs diffusion policy network").
    import logging
    import os

    scale = float(os.environ.get("LEROBOT_BACKBONE_LR_SCALE", "0"))
    if scale > 0:
        named = [(n, p) for n, p in policy.named_parameters() if p.requires_grad]
        backbone = [p for n, p in named if "rgb_encoder" in n and ".backbone." in n]
        bb_ids = {id(p) for p in backbone}
        rest = [p for _, p in named if id(p) not in bb_ids]
        assert backbone, "LEROBOT_BACKBONE_LR_SCALE set but no rgb_encoder.backbone params found"
        base_lr = cfg.optimizer.lr
        params = [{"params": rest}, {"params": backbone, "lr": base_lr * scale}]
        logging.info(
            f"Backbone LR scale {scale}: {len(backbone)} backbone tensors at "
            f"lr={base_lr * scale:.2e}, {len(rest)} tensors at lr={base_lr:.2e}"
        )
    optimizer = cfg.optimizer.build(params)
    lr_scheduler = cfg.scheduler.build(optimizer, cfg.steps) if cfg.scheduler is not None else None
    return optimizer, lr_scheduler
