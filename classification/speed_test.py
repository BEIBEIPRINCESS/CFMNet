"""
Benchmark CFMNet inference throughput.
"""
import os
import time

import torch

import utils
from models.CFMNet import CFMNet

torch.autograd.set_grad_enabled(False)

T0 = 10
T1 = 60


def compute_throughput_cpu(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    start = time.time()
    while time.time() - start < T0:
        model(inputs)

    timing = []
    while sum(timing) < T1:
        start = time.time()
        model(inputs)
        timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    images_per_second = batch_size / timing.mean().item()
    print(name, device, images_per_second, 'images/s @ batch size', batch_size)
    write_result('throughput_cpu_results.txt', name, device, batch_size, images_per_second)


def compute_throughput_cuda(name, model, device, batch_size, resolution=224):
    inputs = torch.randn(batch_size, 3, resolution, resolution, device=device)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast():
        while time.time() - start < T0:
            model(inputs)

    timing = []
    torch.cuda.synchronize()
    with torch.cuda.amp.autocast():
        while sum(timing) < T1:
            start = time.time()
            model(inputs)
            torch.cuda.synchronize()
            timing.append(time.time() - start)
    timing = torch.as_tensor(timing, dtype=torch.float32)
    images_per_second = batch_size / timing.mean().item()
    print(name, device, images_per_second, 'images/s @ batch size', batch_size)
    write_result('throughput_gpu_results.txt', name, device, batch_size, images_per_second)


def write_result(filename, name, device, batch_size, images_per_second):
    os.makedirs('./outputs', exist_ok=True)
    with open(os.path.join('./outputs', filename), 'a') as f:
        f.write(
            f"[backbone_name: {name}]  device: {device}  "
            f"batch_size: {batch_size}  images/s: {images_per_second:.4f}\n"
        )


def main():
    benchmarks = [
        ('CFMNet', CFMNet, 256, 224),
    ]

    for device in ['cuda:0']:
        if 'cuda' in device and not torch.cuda.is_available():
            print('no cuda')
            continue

        if device == 'cpu':
            print('Using 1 cpu thread')
            torch.set_num_threads(1)
            compute_throughput = compute_throughput_cpu
        else:
            print(torch.cuda.get_device_name(torch.cuda.current_device()))
            compute_throughput = compute_throughput_cuda

        for name, builder, batch_size0, resolution in benchmarks:
            batch_size = 16 if device == 'cpu' else batch_size0
            if device != 'cpu':
                torch.cuda.empty_cache()
            model = builder(num_classes=1000)
            utils.replace_batchnorm(model)
            model.to(device)
            model.eval()
            compute_throughput(name, model, device, batch_size, resolution=resolution)


if __name__ == '__main__':
    main()
