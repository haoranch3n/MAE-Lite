# --------------------------------------------------------
# Copyright (c) 2022 Megvii, Inc. and its affiliates. All Rights Reserved.
# --------------------------------------------------------
import os
import torch
from torch import nn

from timm.data import Mixup
from timm.models import create_model
from timm.utils import ModelEmaV2
from loguru import logger

from mae_lite.exps.timm_imagenet_exp import Exp as BaseExp
from mae_lite.layers import build_lr_scheduler
from models_mae import *


class MAE(nn.Module):
    def __init__(self, args, model):
        super(MAE, self).__init__()
        self.model = model
        # mixup
        mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
        if mixup_active:
            mixup_fn = Mixup(
                mixup_alpha=args.mixup,
                cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax,
                prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob,
                mode=args.mixup_mode,
                label_smoothing=args.smoothing,
                num_classes=args.num_classes,
            )
        else:
            mixup_fn = None
        self.mixup_fn = mixup_fn
        self.mask_ratio = args.mask_ratio
        # ema
        if args.model_ema:
            self.ema_model = ModelEmaV2(
                self.model, decay=args.model_ema_decay, device="cpu" if args.model_ema_force_cpu else None
            )
        else:
            self.ema_model = None

    def forward(self, x, target=None, update_param=False):
        if self.training:
            images = x
            if self.mixup_fn is not None:
                images, _ = self.mixup_fn(images, target)
            model_output = self.model(images, self.mask_ratio)
            if len(model_output) > 1:
                loss = model_output[0]
            else:
                loss = self.model(images, self.mask_ratio)

            if self.ema_model is not None:
                self.ema_model.update(self.model)
            return loss, None
        else:
            raise NotImplementedError


class Exp(BaseExp):
    def __init__(self, batch_size, max_epoch=400):
        super(Exp, self).__init__(batch_size, max_epoch)
        # MAE
        self.norm_pix_loss = True
        self.mask_ratio = 0.75

        # dataset & model
        self.dataset = "Fundus"
        self.encoder_arch = "mae_vit_tiny_patch16"
        self.img_size = 224

        # optimizer
        self.opt = "adamw"
        self.opt_eps = 1e-8
        self.opt_betas = (0.9, 0.95)
        self.momentum = 0.9
        self.weight_decay = 0.05
        self.clip_grad = None
        # self.clip_mode = "norm"

        # schedule
        self.sched = "warmcos"
        self.basic_lr_per_img = 1.5e-4 / 256
        self.warmup_lr = 0.0
        self.min_lr = 0.0
        self.warmup_epochs = 40

        # augmentation & regularization
        self.no_aug = False
        self.scale = (0.08, 1.0)
        self.ratio = (3.0 / 4, 4.0 / 3.0)
        self.hflip = 0.5
        self.vflip = 0.0
        self.color_jitter = 0.4
        self.smoothing = 0.0

        # self.num_workers = 10
        self.weights_prefix = "model"
        # self.print_interval = 10
        # self.enable_tensorboard = True
        self.save_folder_prefix = ""
        self.exp_name = os.path.splitext(os.path.realpath(__file__).split("playground/")[-1])[0]
        print(self.exp_name)

    def get_model(self):
        if "model" not in self.__dict__:
            # model = create_model(self.encoder_arch, norm_pix_loss=self.norm_pix_loss)
            model = create_model(self.encoder_arch, norm_pix_loss=self.norm_pix_loss, pretrained=self.pretrained)
            if self.sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            self.model = MAE(self, model)
        return self.model

    def get_data_loader(self):
        super().get_data_loader()
        self.data_loader["eval"] = None
        return self.data_loader

    def get_lr_scheduler(self):
        if "lr" not in self.__dict__:
            self.lr = self.basic_lr_per_img * self.batch_size
        if "warmup_lr" not in self.__dict__:
            self.warmup_lr = self.warmup_lr_per_img * self.batch_size
        if "min_lr" not in self.__dict__:
            self.min_lr = self.min_lr_per_img * self.batch_size

        optimizer = self.get_optimizer()
        iters_per_epoch = len(self.get_data_loader()["train"])
        scheduler = build_lr_scheduler(
            self.sched,
            optimizer,
            self.lr,
            total_steps=iters_per_epoch * self.max_epoch,
            warmup_steps=iters_per_epoch * self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            end_lr=self.min_lr,
        )
        return scheduler
    
    # def set_current_state(self, current_step, ckpt_path=None):
    #     if current_step == 0:
    #         # load pretrain ckpt
    #         if ckpt_path is None:
    #             assert self.pretrain_exp_name is not None, "Please provide a valid 'pretrain_exp_name'!"
    #             ckpt_path = os.path.join(self.output_dir, self.pretrain_exp_name, "last_epoch_ckpt.pth.tar")
    #         logger.info("Load pretrained checkpoints from {}.".format(ckpt_path))
    #         msg = self.set_model_weights(ckpt_path, map_location="cpu")
    #         logger.info("Model params {} are not loaded".format(msg.missing_keys))
    #         logger.info("State-dict params {} are not used".format(msg.unexpected_keys))

    # def set_model_weights(self, ckpt_path, map_location="cpu"):
    #     if not os.path.isfile(ckpt_path):
    #         from torch.nn.modules.module import _IncompatibleKeys

    #         logger.info("No checkpoints found! Training from scratch!")
    #         return _IncompatibleKeys(missing_keys=None, unexpected_keys=None)
    #     ckpt = torch.load(ckpt_path, map_location="cpu")
    #     weights_prefix = self.weights_prefix
    #     if not weights_prefix:
    #         state_dict = {"model." + k: v for k, v in ckpt["model"].items()}
    #     else:
    #         if weights_prefix and not weights_prefix.endswith("."):
    #             weights_prefix += "."
    #         if all(key.startswith("module.") for key in ckpt["model"].keys()):
    #             weights_prefix = "module." + weights_prefix
    #         state_dict = {k.replace(weights_prefix, "model."): v for k, v in ckpt["model"].items()}
    #     msg = self.get_model().load_state_dict(state_dict, strict=False)
    #     return msg


if __name__ == "__main__":
    exp = Exp(2)
    print(exp.exp_name)
    loader = exp.get_data_loader()
    model = exp.get_model()
    print(model)
    opt = exp.get_optimizer()
    sched = exp.get_lr_scheduler()
    
