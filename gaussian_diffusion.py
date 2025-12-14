import os
import sys
import numpy as np
import torch as th

BASE_DIR = os.path.dirname(os.path.abspath(__file__))              # BGDB/
IMPROVED_REPO_DIR = os.path.join(BASE_DIR, "improved-diffusion")   # BGDB/improved-diffusion

if IMPROVED_REPO_DIR not in sys.path:
    sys.path.append(IMPROVED_REPO_DIR)

from improved_diffusion import gaussian_diffusion as _base_gd
from improved_diffusion.gaussian_diffusion import *

class GaussianDiffusion(_base_gd.GaussianDiffusion):
    def p_sample(
        self,
        model,
        x,
        t,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
    ):
        out = self.p_mean_variance(
            model,
            x,
            t,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
        )

        noise = th.randn_like(x)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0

        std = th.exp(0.5 * out["log_variance"])
        sample = out["mean"] + nonzero_mask * std * noise

        return {
            "sample": sample,
            "pred_xstart": out["pred_xstart"],
            "mean": out["mean"],
            "variance": nonzero_mask * std,
        }

    def p_sample_loop(
        self,
        model,
        shape,
        noise=None,
        clip_denoised=True,
        denoised_fn=None,
        model_kwargs=None,
        device=None,
        progress=False,
    ):
        """
        生成样本。

        和原版差异：
          - 原版: return final["sample"]
          - 这里: return final["sample"], final["mean"], final["variance"]
        """
        final = None
        for sample in self.p_sample_loop_progressive(
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
        ):
            final = sample

        return final["sample"], final["mean"], final["variance"]