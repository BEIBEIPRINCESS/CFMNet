"""Benchmark UNetFormer with the canonical CFMNet backbone."""

import argparse
import time

import torch

from geoseg.models.UNetFormer_cfmnet import UNetFormer_CFMNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-classes', type=int, default=7)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--warmup-seconds', type=float, default=10.0)
    parser.add_argument('--measure-seconds', type=float, default=60.0)
    return parser.parse_args()


def replace_batchnorm(module):
    for child_name, child in module.named_children():
        if hasattr(child, 'fuse'):
            setattr(module, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(module, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def benchmark(model, inputs, device, warmup_seconds, measure_seconds):
    with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=device.type == 'cuda'):
        warmup_start = time.perf_counter()
        while time.perf_counter() - warmup_start < warmup_seconds:
            model(inputs)

        timings = []
        while sum(timings) < measure_seconds:
            synchronize(device)
            start = time.perf_counter()
            model(inputs)
            synchronize(device)
            timings.append(time.perf_counter() - start)

    mean_latency = sum(timings) / len(timings)
    return inputs.shape[0] / mean_latency, mean_latency


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available.')
    if device.type == 'cpu':
        torch.set_num_threads(1)

    model = UNetFormer_CFMNet(num_classes=args.num_classes)
    replace_batchnorm(model)
    model.to(device).eval()
    inputs = torch.randn(
        args.batch_size, 3, args.resolution, args.resolution, device=device)

    throughput, latency = benchmark(
        model, inputs, device, args.warmup_seconds, args.measure_seconds)
    print(f'Device: {device}')
    print(f'Batch size: {args.batch_size}')
    print(f'Throughput: {throughput:.2f} images/s')
    print(f'Batch latency: {latency * 1000:.2f} ms')


if __name__ == '__main__':
    main()
