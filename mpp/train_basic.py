import argparse
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel
from einops import rearrange
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from dadaptation import DAdaptAdam, DAdaptAdan
from adan_pytorch import Adan
from collections import OrderedDict
import wandb
import pickle as pkl
import gc
from torchinfo import summary
from collections import defaultdict

from mpp.data_utils.well_dataset import get_dataloader
from mpp.models.avit import build_avit
from mpp.utils import logging_utils
from mpp.utils.YParams import YParams
from mpp.loss_fns import RVMSELoss, NMSELoss


def add_weight_decay(model, weight_decay=1e-5, inner_lr=1e-3, skip_list=()):
    """From Ross Wightman at:
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3

    Goes through the parameter list and if the squeeze dim is 1 or 0 (usually means bias or scale)
    then don't apply weight decay.
    """
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.squeeze().shape) <= 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {
            "params": no_decay,
            "weight_decay": 0.0,
        },
        {"params": decay, "weight_decay": weight_decay},
    ]


class Trainer:
    def __init__(self, params, global_rank, local_rank, device, sweep_id=None):
        self.device = device
        self.params = params
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.sweep_id = sweep_id
        self.log_to_screen = params.log_to_screen
        # Basic setup
        self.train_loss = nn.MSELoss()
        self.startEpoch = 0
        self.epoch = 0
        self.mp_type = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.half
        )

        self.iters = 0
        self.initialize_data(self.params)
        print(f"Initializing model on rank {self.global_rank}")
        self.initialize_model(self.params)
        self.initialize_optimizer(self.params)
        if params.resuming:
            print("Loading checkpoint %s" % params.checkpoint_path)
            self.restore_checkpoint(params.checkpoint_path)
        if params.resuming == False and params.pretrained:
            print("Starting from pretrained model at %s" % params.pretrained_ckpt_path)
            self.restore_checkpoint(params.pretrained_ckpt_path)
            self.iters = 0
            self.startEpoch = 0
        # Do scheduler after checking for resume so we don't warmup every time
        self.initialize_scheduler(self.params)

    def single_print(self, *text):
        if self.global_rank == 0 and self.log_to_screen:
            print(" ".join([str(t) for t in text]))

    def initialize_data(self, params):
        if params.tie_batches:
            in_rank = 0
        else:
            in_rank = self.global_rank
        if self.log_to_screen:
            print(f"Initializing data on rank {self.global_rank}")
        self.train_data_loader = get_dataloader(
            path=params.data_dir,
            num_channels=params.n_states,
            split="train",
            datasets=params.datasets,
            min_stride=params.min_stride,
            max_stride=params.max_stride,
            T_in=params.n_steps,
            T_out=1,
            seed=params.seed,
            batch_size=params.batch_size,
            num_workers=params.num_data_workers,
            prefetch_factor=2,
            is_distributed=dist.is_initialized(),
            shuffle=True,
            use_normalization=False,  # Model handles normalization
        )
        self.train_dataset = self.train_data_loader.dataset
        self.train_sampler = self.train_data_loader.sampler

        self.valid_data_loader = get_dataloader(
            path=params.data_dir,
            num_channels=params.n_states,
            split="valid",
            datasets=params.datasets,
            min_stride=params.min_stride,
            max_stride=params.max_stride,
            T_in=params.n_steps,
            T_out=1,
            seed=params.seed,
            batch_size=params.batch_size,
            num_workers=params.num_data_workers,
            prefetch_factor=2,
            is_distributed=dist.is_initialized(),
            shuffle=False,
            use_normalization=False,  # Model handles normalization
        )
        self.valid_dataset = self.valid_data_loader.dataset
        if dist.is_initialized():
            self.train_sampler.set_epoch(0)

    def initialize_model(self, params):
        if self.params.model_type == "avit":
            self.model = build_avit(params).to(device)

        if self.params.compile:
            print(
                "WARNING: BFLOAT NOT SUPPORTED IN SOME COMPILE OPS SO SWITCHING TO FLOAT16"
            )
            self.mp_type = torch.half
            self.model = torch.compile(self.model)

        if dist.is_initialized():
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                output_device=[self.local_rank],
                find_unused_parameters=True,
            )

        self.single_print(
            f"Model parameter count: {sum([p.numel() for p in self.model.parameters()])}"
        )

    def initialize_optimizer(self, params):
        parameters = add_weight_decay(
            self.model, self.params.weight_decay
        )  # Dont use weight decay on bias/scaling terms
        if params.optimizer == "adam":
            if self.params.learning_rate < 0:
                self.optimizer = DAdaptAdam(
                    parameters, lr=1.0, growth_rate=1.05, log_every=1, decouple=True
                )
            else:
                self.optimizer = optim.AdamW(parameters, lr=params.learning_rate)
        elif params.optimizer == "adan":
            if self.params.learning_rate < 0:
                self.optimizer = DAdaptAdan(
                    parameters, lr=1.0, growth_rate=1.05, log_every=100
                )
            else:
                self.optimizer = Adan(parameters, lr=params.learning_rate)
        elif params.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=params.learning_rate, momentum=0.9
            )
        else:
            raise ValueError(f"Optimizer {params.optimizer} not supported")
        self.gscaler = amp.GradScaler(
            enabled=(self.mp_type == torch.half and params.enable_amp)
        )

    def initialize_scheduler(self, params):
        if params.scheduler_epochs > 0:
            sched_epochs = params.scheduler_epochs
        else:
            sched_epochs = params.max_epochs
        if params.scheduler == "cosine":
            if self.params.learning_rate < 0:
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    last_epoch=(self.startEpoch * params.epoch_size) - 1,
                    T_max=sched_epochs * params.epoch_size,
                    eta_min=params.learning_rate / 100,
                )
            else:
                k = params.warmup_steps
                if (self.startEpoch * params.epoch_size) < k:
                    warmup = torch.optim.lr_scheduler.LinearLR(
                        self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=k
                    )
                    decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                        self.optimizer,
                        eta_min=params.learning_rate / 100,
                        T_max=sched_epochs * params.epoch_size - k,
                    )
                    self.scheduler = torch.optim.lr_scheduler.SequentialLR(
                        self.optimizer,
                        [warmup, decay],
                        [k],
                        last_epoch=(params.epoch_size * self.startEpoch) - 1,
                    )
        else:
            self.scheduler = None

    def save_checkpoint(self, checkpoint_path, model=None):
        """Save model and optimizer to checkpoint"""
        if not model:
            model = self.model

        torch.save(
            {
                "iters": self.epoch * self.params.epoch_size,
                "epoch": self.epoch,
                "model_state": model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )

    def restore_checkpoint(self, checkpoint_path):
        """Load model/opt from path"""
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cuda:{}".format(self.local_rank),
            weights_only=False,
        )
        if "model_state" in checkpoint:
            model_state = checkpoint["model_state"]
        else:
            model_state = checkpoint
        try:  # Try to load with DDP Wrapper
            self.model.load_state_dict(model_state)
        except:  # If that fails, either try to load into module or strip DDP prefix
            if hasattr(self.model, "module"):
                self.model.module.load_state_dict(model_state)
            else:
                new_state_dict = OrderedDict()
                for key, val in model_state.items():
                    # Failing means this came from DDP - strip the DDP prefix
                    name = key[7:]
                    new_state_dict[name] = val
                self.model.load_state_dict(new_state_dict)

        if self.params.resuming:  # restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
            self.iters = checkpoint["iters"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.startEpoch = checkpoint["epoch"]
            self.epoch = self.startEpoch
        else:
            self.iters = 0
        if self.params.pretrained:
            if self.params.freeze_middle:
                self.model.module.freeze_middle()
            elif self.params.freeze_processor:
                self.model.module.freeze_processor()
            else:
                self.model.module.unfreeze()
            # See how much we need to expand the projections
            exp_proj = 0
            # Iterate through the appended datasets and add on enough embeddings for all of them.
            for add_on in self.params.append_datasets:
                exp_proj += len(DSET_NAME_TO_OBJECT[add_on]._specifics()[2])
            self.model.module.expand_projections(exp_proj)
        checkpoint = None
        self.model = self.model.to(self.device)

    def train_one_epoch(self):
        self.model.train()
        self.epoch += 1
        tr_time = 0
        data_time = 0
        data_start = time.time()
        steps = 0
        self.single_print(
            "train_loader_size", len(self.train_data_loader), len(self.train_dataset)
        )

        nmse_loss = NMSELoss(dims=(2, 3))
        rvmse_loss = RVMSELoss(dims=(2, 3))
        mse_loss = nn.MSELoss()

        for batch_idx, data in enumerate(self.train_data_loader):
            logs = {}
            steps += 1
            inp, file_index, field_labels, bcs, tar = map(
                lambda x: x.to(self.device), data
            )
            inp = rearrange(inp, "b t c h w -> t b c h w")
            tar = tar.squeeze(1)  # (b, 1, c, h, w) -> (b, c, h, w)
            data_time += time.time() - data_start
            dtime = time.time() - data_start

            self.model.require_backward_grad_sync = (
                1 + batch_idx
            ) % self.params.accum_grad == 0
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.mp_type,
                enabled=self.params.enable_amp,
            ):
                model_start = time.time()
                output = self.model(inp, field_labels, bcs)

                # Model returns denormalized output, compare to raw target
                spatial_dims = (2, 3)  # H, W dimensions
                residuals = output - tar
                # Differentiate between log and accumulation losses
                tar_norm = 1e-7 + tar.pow(2).mean(spatial_dims, keepdim=True)
                raw_loss = (residuals).pow(2).mean(
                    spatial_dims, keepdim=True
                ) / tar_norm
                # Scale loss for accum
                loss = raw_loss.mean() / self.params.accum_grad
                forward_end = time.time()

                # Logging
                with torch.no_grad():
                    logs["train/l1"] = F.l1_loss(output, tar)
                    logs["train/nmse"] = nmse_loss(output, tar)
                    logs["train/rvmse"] = rvmse_loss(output, tar)
                    logs["train/mse"] = mse_loss(output, tar)

                # Scaler is no op when not using AMP
                self.gscaler.scale(loss).backward()
                backward_end = time.time()
                # Only take step once per accumulation cycle
                if self.model.require_backward_grad_sync:
                    self.gscaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                    self.gscaler.step(self.optimizer)
                    self.gscaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.scheduler is not None:
                        self.scheduler.step()
                tr_time += time.time() - model_start
                lr = self.optimizer.param_groups[0]["lr"]
                if self.log_to_screen and (batch_idx % self.params.log_interval == 0):
                    print(
                        f"Epoch: {self.epoch}, Batch: {batch_idx}, L1 Loss: {logs['train/l1'].item()}, NMSE Loss: {logs['train/nmse'].item()}, RVMSE Loss: {logs['train/rvmse'].item()}, MSE Loss: {logs['train/mse'].item()}, LR: {lr}, "
                    )
                data_start = time.time()

            # If distributed, do lots of logging things
            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach())
                    logs[key] = float(logs[key].item() / dist.get_world_size())

            logs["batches"] = steps
            logs["lr"] = lr

            if self.params.log_to_wandb:
                wandb.log(logs)
            if steps >= self.params.epoch_size:
                break

    def validate_one_epoch(self, full=False):
        """
        Validates - for each batch just use a small subset to make it easier.

        Note: need to split datasets for meaningful metrics, but TBD.
        """
        # Don't bother with full validation set between epochs
        self.model.eval()
        if full:
            cutoff = 999999999999
        else:
            cutoff = 1000
        self.single_print("STARTING VALIDATION!!!")

        rvmse_loss = RVMSELoss()
        nmse_loss = NMSELoss()
        mse_loss = nn.MSELoss()
        logs = {}

        with torch.inference_mode():
            count = 0
            for batch_idx, data in enumerate(self.valid_data_loader):
                # Only do a few batches of each dataset if not doing full validation
                if count > cutoff:
                    break
                count += 1
                inp, file_index, field_labels, bcs, tar = map(
                    lambda x: x.to(self.device), data
                )

                inp = rearrange(inp, "b t c h w -> t b c h w")
                tar = tar.squeeze(1)  # (b, 1, c, h, w) -> (b, c, h, w)

                output = self.model(inp, field_labels, bcs)

                # Model returns denormalized output, compare to raw target
                nmse = nmse_loss(output, tar)
                mse = mse_loss(output, tar)
                rvmse = rvmse_loss(output, tar)

                logs["valid/nmse"] = logs.get("valid/nmse", 0) + nmse
                logs["valid/rmse"] = logs.get("valid/rmse", 0) + mse.sqrt()
                logs["valid/rvmse"] = logs.get("valid/rvmse", 0) + rvmse
                logs["valid/mse"] = logs.get("valid/mse", 0) + mse

                if count % 100 == 0:
                    self.single_print(
                        f"Validation batch {batch_idx}. NMSE: {nmse.item()}. MSE: {mse.sqrt().item()}. RVMSE: {rvmse.item()}"
                    )

        self.single_print("DONE VALIDATING - NOW SYNCING")
        # Average logs
        for key in logs.keys():
            logs[key] = logs[key] / count
        if dist.is_initialized():
            for key in sorted(logs.keys()):
                dist.all_reduce(
                    logs[key].detach()
                )  # There was a bug with means when I implemented this - dont know if fixed
                logs[key] = float(logs[key].item() / dist.get_world_size())
        self.single_print("DONE SYNCING - NOW LOGGING")
        return logs

    def train(self):
        # This is set up this way based on old code to allow wandb sweeps
        if self.params.log_to_wandb:
            if self.sweep_id:
                wandb.init(dir=self.params.experiment_dir)
                hpo_config = wandb.config.as_dict()
                self.params.update_params(hpo_config)
                params = self.params
            else:
                wandb.init(
                    dir=self.params.experiment_dir,
                    config=self.params,
                    name=self.params.name,
                    group=self.params.group,
                    project=self.params.project,
                    entity=self.params.entity,
                    resume=True,
                )

        if self.sweep_id and dist.is_initialized():
            param_file = f"temp_hpo_config_{os.environ['SLURM_JOBID']}.pkl"
            if self.global_rank == 0:
                with open(param_file, "wb") as f:
                    pkl.dump(hpo_config, f)
            dist.barrier()  # Stop until the configs are written by hacky MPI sub
            if self.global_rank != 0:
                with open(param_file, "rb") as f:
                    hpo_config = pkl.load(f)
            dist.barrier()  # Stop until the configs are written by hacky MPI sub
            if self.global_rank == 0:
                os.remove(param_file)
            # If tuning batch size, need to go from global to local batch size
            if "batch_size" in hpo_config:
                hpo_config["batch_size"] = int(
                    hpo_config["batch_size"] // self.world_size
                )
            self.params.update_params(hpo_config)
            params = self.params
            self.initialize_data(
                self.params
            )  # This is the annoying redundant part - but the HPs need to be set from wandb
            self.initialize_model(self.params)
            self.initialize_optimizer(self.params)
            self.initialize_scheduler(self.params)
        if self.global_rank == 0:
            summary(self.model)
        if self.params.log_to_wandb:
            wandb.watch(self.model)
        self.single_print("Starting Training Loop...")
        # Actually train now, saving checkpoints, logging time, and logging to wandb
        best_valid_loss = 1.0e6
        for epoch in range(self.startEpoch, self.params.max_epochs):
            if dist.is_initialized():
                self.train_sampler.set_epoch(epoch)
            start = time.time()

            # with torch.autograd.detect_anomaly(check_nan=True):
            self.train_one_epoch()

            valid_start = time.time()
            # Only do full validation set on last epoch - don't waste time
            if epoch == self.params.max_epochs - 1:
                valid_logs = self.validate_one_epoch(True)
            else:
                valid_logs = self.validate_one_epoch(False)

            post_start = time.time()
            time_logs = {}
            time_logs["time/train_time"] = valid_start - start
            time_logs["time/valid_time"] = post_start - valid_start
            if self.params.log_to_wandb:
                wandb.log(time_logs)
                wandb.log(valid_logs)
            gc.collect()
            torch.cuda.empty_cache()

            if self.global_rank == 0:
                if self.params.save_checkpoint:
                    self.save_checkpoint(self.params.checkpoint_path)
                if epoch % self.params.checkpoint_save_interval == 0:
                    self.save_checkpoint(self.params.checkpoint_path + f"_epoch{epoch}")

                cur_time = time.time()
                self.single_print(
                    f"Time for train {valid_start - start}. For valid: {post_start - valid_start}. For postprocessing:{cur_time - post_start}"
                )
                self.single_print(
                    "Time taken for epoch {} is {} sec".format(
                        epoch + 1, time.time() - start
                    )
                )
                self.single_print(f"Valid MSE loss: {valid_logs['valid/mse']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", default="00", type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument(
        "--use_ddp", action="store_true", help="Use distributed data parallel"
    )
    parser.add_argument("--yaml_config", default="./config/multi_ds.yaml", type=str)
    parser.add_argument("--config", default="basic_config", type=str)
    args = parser.parse_args()
    params = YParams(Path(args.yaml_config).absolute(), args.config)
    params.use_ddp = args.use_ddp
    params["data_dir"] = args.data_dir
    params["results_dir"] = args.results_dir

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.use_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(
            local_rank
        )  # Torch docs recommend just using device, but I had weird memory issues without setting this.
    device = (
        torch.device(local_rank) if torch.cuda.is_available() else torch.device("cpu")
    )

    # Modify params
    params["batch_size"] = int(params.batch_size // world_size)
    params["startEpoch"] = 0

    expDir = Path(args.results_dir) / args.config / str(args.run_name)

    params["experiment_dir"] = str(expDir.absolute())
    params["checkpoint_path"] = str(expDir / "training_checkpoints/ckpt.tar")

    # Have rank 0 check for and/or make directory
    if global_rank == 0:
        if not expDir.is_dir():
            expDir.mkdir(parents=True, exist_ok=True)
            (expDir / "training_checkpoints").mkdir(parents=True, exist_ok=True)

    params["resuming"] = True if Path(params.checkpoint_path).is_file() else False

    # WANDB things
    params["name"] = str(args.run_name)

    params["log_to_wandb"] = (global_rank == 0) and params["log_to_wandb"]
    params["log_to_screen"] = (global_rank == 0) and params["log_to_screen"]
    torch.backends.cudnn.benchmark = False

    if global_rank == 0:
        hparams = ruamelDict()
        yaml = YAML()
        for key, value in params.params.items():
            hparams[str(key)] = str(value)
        with open(expDir / "hyperparams.yaml", "w") as hpfile:
            yaml.dump(hparams, hpfile)
    trainer = Trainer(params, global_rank, local_rank, device)
    trainer.train()
    if params.log_to_screen:
        print("DONE ---- rank %d" % global_rank)
