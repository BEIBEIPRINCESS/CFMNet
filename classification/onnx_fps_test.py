import argparse
import os
import platform
import site
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


ORT_DTYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
}


def ensure_nvidia_libs_on_path():
    if os.environ.get("ONNX_FPS_LIB_PATH_READY") == "1":
        return

    if "--providers" in sys.argv:
        providers_index = sys.argv.index("--providers")
        requested_providers = sys.argv[providers_index + 1 :]
        if "cuda" not in requested_providers:
            return
    elif "--providers=cuda" not in sys.argv:
        requested_providers = ["cpu", "cuda"]

    search_roots = []
    for site_dir in site.getsitepackages():
        nvidia_dir = Path(site_dir) / "nvidia"
        if nvidia_dir.is_dir():
            search_roots.append(nvidia_dir)

    user_site = site.getusersitepackages()
    user_nvidia_dir = Path(user_site) / "nvidia"
    if user_nvidia_dir.is_dir():
        search_roots.append(user_nvidia_dir)

    lib_dirs = []
    for root in search_roots:
        lib_dirs.extend(path for path in root.glob("*/lib") if path.is_dir())

    if not lib_dirs:
        return

    existing = [
        path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path
    ]
    new_paths = [str(path) for path in lib_dirs if str(path) not in existing]
    if not new_paths:
        return

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(new_paths + existing)
    env["ONNX_FPS_LIB_PATH_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable] + sys.argv, env)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark ONNX Runtime FPS on current CPU and GPU."
    )
    parser.add_argument(
        "--onnx",
        required=True,
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Input batch size. Static ONNX batch is used when omitted.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Fallback image size for dynamic H/W inputs.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Fallback input height. Overrides --img-size.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Fallback input width. Overrides --img-size.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations before timing.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=200,
        help="Timed iterations.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help="CPU intra-op thread count. 0 lets ONNX Runtime decide.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="CUDA device id used by ONNX Runtime.",
    )
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["cpu", "cuda"],
        choices=["cpu", "cuda"],
        help="Providers to benchmark.",
    )
    parser.add_argument(
        "--no-io-binding",
        action="store_true",
        help="Disable CUDA I/O binding and use normal session.run instead.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional txt file to append benchmark results.",
    )
    return parser.parse_args()


def is_static_dim(dim):
    return isinstance(dim, int) and dim > 0


def resolve_input_shape(input_meta, args):
    raw_shape = list(input_meta.shape)
    if len(raw_shape) != 4:
        raise ValueError(
            f"Only NCHW image inputs are supported, got input shape: {raw_shape}"
        )

    batch = raw_shape[0] if is_static_dim(raw_shape[0]) else args.batch_size or 1
    channels = raw_shape[1] if is_static_dim(raw_shape[1]) else 3
    height = raw_shape[2] if is_static_dim(raw_shape[2]) else args.height or args.img_size
    width = raw_shape[3] if is_static_dim(raw_shape[3]) else args.width or args.img_size

    if args.batch_size is not None and is_static_dim(raw_shape[0]):
        if args.batch_size != raw_shape[0]:
            print(
                f"Warning: ONNX has static batch={raw_shape[0]}, "
                f"ignore --batch-size {args.batch_size}."
            )

    return [int(batch), int(channels), int(height), int(width)]


def create_session(onnx_path, provider_name, args):
    sess_options = ort.SessionOptions()
    if args.cpu_threads > 0:
        sess_options.intra_op_num_threads = args.cpu_threads

    if provider_name == "cpu":
        providers = ["CPUExecutionProvider"]
    elif provider_name == "cuda":
        providers = [
            (
                "CUDAExecutionProvider",
                {
                    "device_id": args.gpu_device_id,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
            "CPUExecutionProvider",
        ]
    else:
        raise ValueError(f"Unsupported provider: {provider_name}")

    return ort.InferenceSession(str(onnx_path), sess_options, providers=providers)


def run_normal(session, input_name, output_names, input_array, warmup, runs):
    feed = {input_name: input_array}
    for _ in range(warmup):
        session.run(output_names, feed)

    start = time.perf_counter()
    for _ in range(runs):
        session.run(output_names, feed)
    elapsed = time.perf_counter() - start
    return elapsed


def run_cuda_iobinding(session, input_name, output_names, input_array, args):
    input_value = ort.OrtValue.ortvalue_from_numpy(
        input_array, "cuda", args.gpu_device_id
    )

    io_binding = session.io_binding()
    io_binding.bind_ortvalue_input(input_name, input_value)
    for output_name in output_names:
        io_binding.bind_output(output_name, "cuda", args.gpu_device_id)

    for _ in range(args.warmup):
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()

    start = time.perf_counter()
    for _ in range(args.runs):
        session.run_with_iobinding(io_binding)
        io_binding.synchronize_outputs()
    elapsed = time.perf_counter() - start
    return elapsed


def benchmark_provider(onnx_path, provider_name, args):
    available = ort.get_available_providers()
    required_provider = (
        "CUDAExecutionProvider" if provider_name == "cuda" else "CPUExecutionProvider"
    )
    if required_provider not in available:
        return {
            "provider": provider_name,
            "available": False,
            "reason": f"{required_provider} is not available. Available: {available}",
        }

    session = create_session(onnx_path, provider_name, args)
    active_providers = session.get_providers()
    input_meta = session.get_inputs()[0]
    output_names = [output.name for output in session.get_outputs()]
    input_shape = resolve_input_shape(input_meta, args)
    dtype = ORT_DTYPE_TO_NUMPY.get(input_meta.type)
    if dtype is None:
        raise ValueError(f"Unsupported input dtype: {input_meta.type}")

    rng = np.random.default_rng(0)
    input_array = rng.standard_normal(input_shape).astype(dtype)
    batch_size = input_shape[0]

    used_iobinding = False
    if provider_name == "cuda" and not args.no_io_binding:
        try:
            elapsed = run_cuda_iobinding(
                session, input_meta.name, output_names, input_array, args
            )
            used_iobinding = True
        except Exception as exc:
            print(f"CUDA I/O binding failed, fallback to session.run: {exc}")
            elapsed = run_normal(
                session,
                input_meta.name,
                output_names,
                input_array,
                args.warmup,
                args.runs,
            )
    else:
        elapsed = run_normal(
            session,
            input_meta.name,
            output_names,
            input_array,
            args.warmup,
            args.runs,
        )

    latency_ms = elapsed / args.runs * 1000.0
    fps = batch_size * args.runs / elapsed
    return {
        "provider": provider_name,
        "available": True,
        "active_providers": active_providers,
        "input_name": input_meta.name,
        "input_shape": input_shape,
        "input_type": input_meta.type,
        "batch_size": batch_size,
        "warmup": args.warmup,
        "runs": args.runs,
        "elapsed": elapsed,
        "latency_ms": latency_ms,
        "fps": fps,
        "used_iobinding": used_iobinding,
    }


def format_result(result):
    if not result["available"]:
        return f"[{result['provider'].upper()}] skipped: {result['reason']}"

    suffix = " with CUDA I/O binding" if result["used_iobinding"] else ""
    return (
        f"[{result['provider'].upper()}] "
        f"FPS: {result['fps']:.2f} images/s | "
        f"Latency: {result['latency_ms']:.3f} ms/batch | "
        f"Batch: {result['batch_size']} | "
        f"Input: {result['input_shape']} | "
        f"Runs: {result['runs']}{suffix}"
    )


def main():
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")

    print(f"ONNX: {onnx_path}")
    print(f"ONNX Runtime: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    print(f"CPU: {platform.processor() or platform.machine()}")

    results = []
    for provider_name in args.providers:
        result = benchmark_provider(onnx_path, provider_name, args)
        results.append(result)
        print(format_result(result))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as f:
            f.write(f"ONNX: {onnx_path}\n")
            f.write(f"ONNX Runtime: {ort.__version__}\n")
            for result in results:
                f.write(format_result(result) + "\n")
            f.write("\n")
        print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    ensure_nvidia_libs_on_path()
    main()
