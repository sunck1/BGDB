import torch
import numpy as np
import functools
import os
import yaml
from argparse import Namespace

from improved_diffusion import dist_util
from improved_diffusion.resample import (
    create_named_schedule_sampler,
    LossAwareSampler,
)
from script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)

from network.utils import _SimpleSegmentationModel as DeeplabSimpleSeg

def approx_standard_normal_cdf_patched(x):
    scale = x.new_tensor(np.sqrt(2.0 / np.pi))
    return 0.5 * (1.0 + torch.tanh(scale * (x + 0.044715 * torch.pow(x, 3))))

def save_opts_to_yaml(opts, filename="config.yaml"):
    """
    Save argparse.Namespace (opts) into BGDB/config.yaml
    """
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)

    # Namespace → dict
    if isinstance(opts, Namespace):
        opts_dict = vars(opts)
    else:
        opts_dict = opts

    with open(save_path, "w") as f:
        yaml.dump(opts_dict, f, default_flow_style=False)

    print(f"[INFO] Saved options to: {save_path}")

def load_yaml_if_exists():
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if os.path.isfile(cfg_path):
        with open(cfg_path, "r") as f:
            return yaml.safe_load(f)
    return {}

def merge_config():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=0.1,
        weight_decay=1e-4,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=16,
        ema_rate="0.9999",
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())

    yaml_cfg = load_yaml_if_exists()
    if yaml_cfg:
        print("[INFO] Loaded config.yaml and overriding defaults:")
        print(yaml_cfg)
        defaults.update(yaml_cfg)

    return defaults


class MySimpleSegmentationModel(DeeplabSimpleSeg):

    def __init__(self, backbone, classifier):
        super(MySimpleSegmentationModel, self).__init__(backbone, classifier)
        self.backbone = backbone
        self.classifier = classifier

        cfg = merge_config()
        self.microbatch = cfg.get("microbatch", 16)

        dist_util.setup_dist()

        diff_keys = list(model_and_diffusion_defaults().keys())
        diff_kwargs = {k: cfg[k] for k in diff_keys if k in cfg}

        if "num_classes" in cfg:
            diff_kwargs["num_classes"] = cfg["num_classes"]
        else:
            raise ValueError(
                "num_classes not found in config; please set it in config.yaml."
            )

        self.model, self.diffusion = create_model_and_diffusion(**diff_kwargs)

        sampler_name = cfg.get("schedule_sampler", "uniform")
        self.schedule_sampler = create_named_schedule_sampler(
            sampler_name, self.diffusion
        )

        self.ddp_model = self.model
        self.use_ddp = False
        self.cond = None

    def forward(self, x):
        return super(MySimpleSegmentationModel, self).forward(x)

    def forward_train(self, x):
        seg = self.forward(x)
        loss = self.forward_backward(seg, self.cond)
        return seg, loss

    def forward_backward(self, batch, cond):
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            
            micro_cond = None
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
        return loss