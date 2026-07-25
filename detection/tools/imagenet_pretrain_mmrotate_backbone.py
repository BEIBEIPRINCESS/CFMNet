import os
import argparse
import time
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mmcv import Config
from mmcv.runner import set_random_seed
from mmrotate.models import build_backbone

# -------------------------
# DALI (GPU decode + aug)
# -------------------------
try:
    from nvidia.dali import fn, types
    from nvidia.dali.pipeline import Pipeline
    from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
except Exception as e:
    fn = types = Pipeline = DALIGenericIterator = LastBatchPolicy = None
    _dali_import_error = e


def init_dist():
    """Init torch.distributed if launched by torchrun."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)  # local_rank maps to visible device index
        dist.init_process_group(backend="nccl", init_method="env://")
        return True, local_rank
    return False, 0


def is_main():
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


@torch.no_grad()
def topk_acc_gpu(logits, targets, topk=(1, 5)):
    """Return (correct1, correct5, n) on GPU to avoid per-iter sync."""
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)     # [B, maxk]
    pred = pred.t()                                 # [maxk, B]
    correct = pred.eq(targets.view(1, -1).expand_as(pred))  # bool

    correct1 = correct[:1].reshape(-1).sum()
    correct5 = correct[:5].reshape(-1).sum()
    n = targets.numel()
    return correct1, correct5, n


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def save_backbone_only(model, path):
    model_ = unwrap_model(model)
    ckpt = {"state_dict": model_.state_dict()}
    torch.save(ckpt, path)


def save_full_checkpoint(model, optim, sched, scaler, epoch, best_top1, path, args):
    model_ = unwrap_model(model)
    ckpt = {
        "epoch": epoch,
        "best_top1": best_top1,
        "model": model_.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": sched.state_dict() if sched is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def load_full_checkpoint(model, optim, sched, scaler, path, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model_ = unwrap_model(model)
    model_.load_state_dict(ckpt["model"], strict=False)

    if optim is not None and ckpt.get("optimizer", None) is not None:
        optim.load_state_dict(ckpt["optimizer"])

    if sched is not None and ckpt.get("scheduler", None) is not None:
        sched.load_state_dict(ckpt["scheduler"])

    if scaler is not None and ckpt.get("scaler", None) is not None:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_top1 = float(ckpt.get("best_top1", 0.0))
    return start_epoch, best_top1


def build_warmup_cosine_scheduler(optim, total_steps, warmup_steps, min_lr, base_lr):
    """
    Warmup (linear) + Cosine decay (per-iter).
    - warmup: lr from 0 -> base_lr
    - cosine: lr from base_lr -> min_lr
    """
    assert total_steps > 0
    warmup_steps = int(max(0, warmup_steps))
    min_lr = float(min_lr)
    base_lr = float(base_lr)
    min_ratio = min_lr / base_lr

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if total_steps <= warmup_steps:
            return 1.0

        t = float(step - warmup_steps)
        T = float(total_steps - warmup_steps)
        t = min(t, T)
        cosine = 0.5 * (1.0 + math.cos(math.pi * t / T))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)


# ============================================================
# ✅ DALI Pipelines (FIXED): use fn.xxx(jpegs, ...) style
# ============================================================
class ImageNetTrainPipe(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id,
                 shard_id, num_shards, seed=0):
        super().__init__(batch_size=batch_size, num_threads=num_threads,
                         device_id=device_id, seed=seed)

        self.data_dir = data_dir
        self.shard_id = shard_id
        self.num_shards = num_shards

        # params
        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def define_graph(self):
        jpegs, labels = fn.readers.file(
            file_root=os.path.join(self.data_dir, "train"),
            random_shuffle=True,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            name="Reader",
        )

        # ✅ IMPORTANT: pass jpegs as input (this fixes "received 0")
        images = fn.decoders.image_random_crop(
            jpegs,
            device="mixed",
            output_type=types.RGB,
            random_aspect_ratio=[3.0 / 4.0, 4.0 / 3.0],
            random_area=[0.08, 1.0],
            num_attempts=10,
        )
        images = fn.resize(
            images,
            device="gpu",
            resize_x=224,
            resize_y=224,
            interp_type=types.INTERP_LINEAR,
        )
        mirror = fn.random.coin_flip(probability=0.5)

        images = fn.crop_mirror_normalize(
            images,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=self.mean,
            std=self.std,
            mirror=mirror,
        )

        labels = labels.gpu()
        return images, labels


class ImageNetValPipe(Pipeline):
    def __init__(self, data_dir, batch_size, num_threads, device_id,
                 shard_id, num_shards, seed=0):
        super().__init__(batch_size=batch_size, num_threads=num_threads,
                         device_id=device_id, seed=seed)

        self.data_dir = data_dir
        self.shard_id = shard_id
        self.num_shards = num_shards

        self.mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        self.std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    def define_graph(self):
        jpegs, labels = fn.readers.file(
            file_root=os.path.join(self.data_dir, "val"),
            random_shuffle=False,
            shard_id=self.shard_id,
            num_shards=self.num_shards,
            name="Reader",
        )

        # ✅ IMPORTANT: pass jpegs as input
        images = fn.decoders.image(
            jpegs,
            device="mixed",
            output_type=types.RGB,
        )
        images = fn.resize(
            images,
            device="gpu",
            resize_shorter=256,
            interp_type=types.INTERP_LINEAR,
        )

        # center crop 224
        images = fn.crop(
            images,
            device="gpu",
            crop=(224, 224),
            crop_pos_x=0.5,
            crop_pos_y=0.5,
        )

        images = fn.crop_mirror_normalize(
            images,
            device="gpu",
            dtype=types.FLOAT,
            output_layout="CHW",
            mean=self.mean,
            std=self.std,
            mirror=0,
        )

        labels = labels.gpu()
        return images, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--batch-size", type=int, default=256)  # per-GPU
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--min-lr", type=float, default=1.0e-5)
    parser.add_argument("--wd", type=float, default=0.05)

    # DALI thread count (per process)
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="work_dirs/imagenet_pretrain")
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=1)
    parser.add_argument("--save-interval", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--resume", type=str, default="")

    args = parser.parse_args()

    # DALI availability check
    if Pipeline is None:
        raise RuntimeError(
            f"DALI import failed: {_dali_import_error}\n"
            f"Please install DALI, e.g. `pip install nvidia-dali-cuda11x` (choose your CUDA version)."
        )

    ddp, local_rank = init_dist()
    torch.backends.cudnn.benchmark = True
    set_random_seed(args.seed, deterministic=False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
    world = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1

    # ----------------------
    # Build DALI pipelines
    # ----------------------
    train_pipe = ImageNetTrainPipe(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_threads=max(1, args.workers),
        device_id=local_rank,
        shard_id=rank,
        num_shards=world,
        seed=args.seed + rank,
    )
    val_pipe = ImageNetValPipe(
        data_dir=args.data,
        batch_size=args.batch_size,
        num_threads=max(1, args.workers),
        device_id=local_rank,
        shard_id=rank,
        num_shards=world,
        seed=args.seed + 1234 + rank,
    )

    train_pipe.build()
    val_pipe.build()

    # steps per epoch (per-shard)
    train_samples = train_pipe.epoch_size("Reader")
    val_samples = val_pipe.epoch_size("Reader")

    steps_per_epoch = train_samples // args.batch_size  # DROP
    if steps_per_epoch <= 0:
        raise RuntimeError(f"train_samples too small? train_samples={train_samples}, batch={args.batch_size}")

    val_steps = math.ceil(val_samples / args.batch_size)  # PARTIAL

    # ✅ DALI iterator: pass LIST[pipeline]
    train_loader = DALIGenericIterator(
        pipelines=[train_pipe],
        output_map=["data", "label"],
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.DROP,
        prepare_first_batch=True,
    )
    val_loader = DALIGenericIterator(
        pipelines=[val_pipe],
        output_map=["data", "label"],
        reader_name="Reader",
        auto_reset=True,
        last_batch_policy=LastBatchPolicy.PARTIAL,
        prepare_first_batch=True,
    )

    # ----------------------
    # Model
    # ----------------------
    cfg = Config.fromfile(args.cfg)
    model = build_backbone(cfg.backbone).cuda()

    if ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    criterion = nn.CrossEntropyLoss().cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ----------------------
    # Scheduler (per-iter)
    # ----------------------
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    sched = build_warmup_cosine_scheduler(
        optim,
        total_steps=total_steps,
        warmup_steps=warmup_steps,
        min_lr=args.min_lr,
        base_lr=args.lr
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    # ----------------------
    # Resume
    # ----------------------
    start_epoch = 0
    best_top1 = 0.0
    global_step = 0

    if args.resume:
        start_epoch, best_top1 = load_full_checkpoint(model, optim, sched, scaler, args.resume, map_location="cpu")
        global_step = int(getattr(sched, "last_epoch", 0))
        if is_main():
            print(f"[resume] from={args.resume} start_epoch={start_epoch} best_top1={best_top1:.2f} "
                  f"global_step={global_step}", flush=True)

    barrier()

    # ----------------------
    # Train / Eval Loop
    # ----------------------
    end_time = time.time()

    for epoch in range(start_epoch, args.epochs):
        model.train()
        torch.cuda.reset_peak_memory_stats()

        # Iterate exactly steps_per_epoch
        for it in range(steps_per_epoch):
            data_time = time.time() - end_time

            batch = next(train_loader)          # list length 1
            imgs = batch[0]["data"]             # GPU float CHW
            labels = batch[0]["label"].squeeze(-1).long()  # GPU int64 [B]

            optim.zero_grad(set_to_none=True)

            t0 = time.time()
            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim)
            scaler.update()

            sched.step()
            global_step += 1

            torch.cuda.synchronize()
            step_time = time.time() - t0
            end_time = time.time()

            if is_main() and (it % args.log_interval == 0):
                lr_now = optim.param_groups[0]["lr"]
                scale_now = float(scaler.get_scale()) if args.amp else 1.0
                mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
                print(f"[train] epoch={epoch+1}/{args.epochs} it={it}/{steps_per_epoch} "
                      f"loss={loss.item():.4f} lr={lr_now:.3e} data={data_time:.3f}s step={step_time:.3f}s "
                      f"scale={scale_now:.1f} mem={mem_gb:.2f}GB", flush=True)

        # ----------------------
        # Eval
        # ----------------------
        do_eval = (args.eval_interval > 0) and ((epoch + 1) % args.eval_interval == 0 or (epoch + 1) == args.epochs)
        if do_eval:
            model.eval()

            correct1 = torch.zeros([], device="cuda", dtype=torch.long)
            correct5 = torch.zeros([], device="cuda", dtype=torch.long)
            total_n = torch.zeros([], device="cuda", dtype=torch.long)

            with torch.no_grad():
                for _ in range(val_steps):
                    batch = next(val_loader)
                    imgs = batch[0]["data"]
                    labels = batch[0]["label"].squeeze(-1).long()

                    with torch.cuda.amp.autocast(enabled=args.amp):
                        logits = model(imgs)

                    c1, c5, n = topk_acc_gpu(logits, labels, topk=(1, 5))
                    correct1 += c1
                    correct5 += c5
                    total_n += n

            if ddp:
                dist.all_reduce(correct1, op=dist.ReduceOp.SUM)
                dist.all_reduce(correct5, op=dist.ReduceOp.SUM)
                dist.all_reduce(total_n, op=dist.ReduceOp.SUM)

            top1 = (correct1.float() * 100.0 / total_n.float()).item()
            top5 = (correct5.float() * 100.0 / total_n.float()).item()

            if is_main():
                lr_now = optim.param_groups[0]["lr"]
                print(f"[eval ] epoch={epoch+1}/{args.epochs} top1={top1:.2f} top5={top5:.2f} lr={lr_now:.3e}",
                      flush=True)

                save_backbone_only(model, str(out_dir / "backbone_only_last.pth"))

                if top1 > best_top1:
                    best_top1 = top1
                    save_backbone_only(model, str(out_dir / "backbone_only_best.pth"))
                    print(f"[best ] updated best_top1={best_top1:.2f}", flush=True)

        # ----------------------
        # Save checkpoints
        # ----------------------
        do_save = (args.save_interval > 0) and ((epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs)
        if do_save and is_main():
            ckpt_path = out_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            save_full_checkpoint(model, optim, sched, scaler, epoch, best_top1, str(ckpt_path), args)
            save_full_checkpoint(model, optim, sched, scaler, epoch, best_top1, str(out_dir / "checkpoint_latest.pth"), args)
            print(f"[ckpt] saved: {ckpt_path.name}", flush=True)

        barrier()

    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
