# python main.py --model deeplabv3plus_mobilenet --crop_val --lr 0.01 --crop_size 256 --batch_size 16 --output_stride 16 --data_root /orange/xujie/chengkun/chengkun/cityscapes --dataset cityscapes --random_seed 7

import os
import sys
import random
import importlib.util
import warnings
import numpy as np

import torch
import torch.nn as nn
from torch.utils import data
# from PIL import Image
# import matplotlib
# import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ======================
# Change Path
# ======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))            # BGDB
DEEPLAB_DIR = os.path.join(BASE_DIR, "DeepLabV3")                # BGDB/DeepLabV3
IMPROVED_REPO_DIR = os.path.join(BASE_DIR, "improved-diffusion") # BGDB/improved-diffusion

def _ensure_in_sys_path(path, prepend=False):
    if path not in sys.path:
        if prepend:
            sys.path.insert(0, path)
        else:
            sys.path.append(path)

# 让 DeepLabV3 在前，BGDB / improved-diffusion 在后
_ensure_in_sys_path(DEEPLAB_DIR, prepend=True)
_ensure_in_sys_path(BASE_DIR)
_ensure_in_sys_path(IMPROVED_REPO_DIR)

# ======================
# Patch improved-diffusion submodule
# ======================

def _load_root_module(module_name: str, file_name: str):
    file_path = os.path.join(BASE_DIR, file_name)
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

root_gd = _load_root_module("root_gaussian_diffusion", "gaussian_diffusion.py")
sys.modules["improved_diffusion.gaussian_diffusion"] = root_gd
print("[Patch] improved_diffusion.gaussian_diffusion -> root.gaussian_diffusion")

import dist_util as root_dist_util  # BGDB/dist_util.py
sys.modules["improved_diffusion.dist_util"] = root_dist_util
print("[Patch] improved_diffusion.dist_util -> root.dist_util")
root_utils = _load_root_module("root_utils", "utils.py")

# ======================
# DeepLabV3 related import
# ======================
import network                      # BGDB/DeepLabV3/network
import utils                        # BGDB/DeepLabV3/utils
from metrics import StreamSegMetrics
from utils.visualizer import Visualizer
from main import get_dataset, validate, get_argparser

# ======================
# Patch: replace CDF and Segmentation
# ======================

def patch_diffusion_losses_cdf():
    import improved_diffusion.losses as losses_mod  # BGDB/improved-diffusion/improved_diffusion/losses.py

    if hasattr(root_utils, "approx_standard_normal_cdf_patched"):
        losses_mod.approx_standard_normal_cdf = root_utils.approx_standard_normal_cdf_patched
        print("[Patch] losses.approx_standard_normal_cdf -> root_utils.approx_standard_normal_cdf_patched")
    else:
        print("[Patch] root_utils.approx_standard_normal_cdf_patched NOT found, please check utils.py")


def patch_deeplab_with_my_segmentation_model():
    from network import _deeplab as dl_deeplab      # BGDB/DeepLabV3/network/_deeplab.py
    import network.utils as net_utils               # BGDB/DeepLabV3/network/utils.py

    MySegBase = root_utils.MySimpleSegmentationModel

    def _patch_class_base(module, cls_name: str):
        if not hasattr(module, cls_name):
            return
        cls = getattr(module, cls_name)
        if MySegBase not in cls.__bases__:
            cls.__bases__ = (MySegBase,)
            print(f"[Patch] {module.__name__}.{cls_name}.__bases__ -> MySimpleSegmentationModel")

    for mod in (net_utils, dl_deeplab):
        _patch_class_base(mod, "DeepLabV3")
        _patch_class_base(mod, "DeepLabV3Plus")

def main():
    patch_diffusion_losses_cdf()
    patch_deeplab_with_my_segmentation_model()

    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19

    root_utils.save_opts_to_yaml(opts)

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset == 'voc' and not opts.crop_val:
        opts.val_batch_size = 1

    train_dst, val_dst = get_dataset(opts)
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=50,
        drop_last=True)  # drop_last=True to ignore single-image batches.
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=50)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # Set up model (all models are 'constructed at network.modeling)
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

    # Set up optimizer
    # optimizer = torch.optim.SGD(params=[
    #     {'params': model.backbone.parameters(), 'lr': 0.1 * opts.lr},
    #     {'params': model.classifier.parameters(), 'lr': opts.lr},
    # ], lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    # torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    if opts.lr_policy == 'poly':
        scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    elif opts.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)

    # Set up criterion
    # criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
        criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    def save_ckpt(path):
        """ save current model
        """
        model_to_save = model.module if hasattr(model, "module") else model

        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model_to_save.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)

    utils.mkdir('checkpoints')
    # Restore
    mse = nn.MSELoss() # different with original code
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        # https://github.com/VainF/DeepLabV3Plus-Pytorch/issues/8#issuecomment-605601402, @PytaichukBohdan
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        # model = nn.DataParallel(model)
        model.to(device)
        if opts.continue_training:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            cur_itrs = checkpoint["cur_itrs"]
            best_score = checkpoint['best_score']
            print("Training state restored from %s" % opts.ckpt)
        print("Model restored from %s" % opts.ckpt)
        del checkpoint  # free memory
    else:
        print("[!] Retrain")
        # model = nn.DataParallel(model)
        model.to(device)

    # ==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if opts.enable_vis else None  # sample idxs for visualization
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # denormalization for ori images

    if opts.test_only:
        model.eval()
        val_score, ret_samples = validate(
            opts=opts, model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
        print(metrics.to_str(val_score))
        return

    interval_loss = 0
    while True:  # cur_itrs < opts.total_itrs:
        # =====  Train  =====
        model.train()
        print(cur_itrs)
        cur_epochs += 1
        print(cur_epochs)
        for (images, labels) in train_loader:
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(images)
            
            ##############################################################################################
            shape_output = [outputs.shape[0], opts.num_classes, 256, 256]
            var_zero = torch.zeros(shape_output).cuda()
            seg_ori, loss_diff = model.forward_train(images.float())
            _, mean, variance = model.diffusion.p_sample_loop(model.model, shape_output)
            loss = criterion(seg_ori, labels) + \
                0.5 * (criterion(mean, labels) +  \
                mse(variance,var_zero)) + \
                0.5 * loss_diff
            ##############################################################################################

            loss.backward()
            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss / 10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0

            if (cur_itrs) % opts.val_interval == 0:
                save_ckpt('checkpoints/latest_%s_%s_os%d.pth' %
                          (opts.model, opts.dataset, opts.output_stride))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts, model=model, loader=val_loader, device=device, metrics=metrics,
                    ret_samples_ids=vis_sample_id)
                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    save_ckpt('checkpoints/best_%s_%s_os%d.pth' %
                              (opts.model, opts.dataset, opts.output_stride))
                print(best_score)
                if vis is not None:  # visualize validation score and samples
                    vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                    vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                    vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        vis.vis_image('Sample %d' % k, concat_img)
                model.train()
            scheduler.step()

            if cur_itrs >= opts.total_itrs:
                return


if __name__ == '__main__':
    main()
