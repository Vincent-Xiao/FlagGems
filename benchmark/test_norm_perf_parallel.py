import concurrent.futures
import gc
import multiprocessing as mp
from typing import List, Tuple

import pytest
import torch

import flag_gems
from benchmark.attri_util import BenchMode, BenchmarkMetrics, BenchmarkResult
from benchmark.conftest import Config
from benchmark.performance_utils import GenericBenchmark2DOnly, SkipVersion


def rms_norm_input_fn(shape, dtype, device):
    _, N = shape
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn(N, dtype=dtype, device=device)
    yield inp, (N,), weight


def _run_shapes_on_gpu(
    indexed_shapes: List[Tuple[int, Tuple[int, ...]]],
    dtype_name: str,
    gpu_id: int,
    worker_mode: str,
    worker_warm_up: int,
    worker_repetition: int,
    worker_metrics: List[str],
) -> List[Tuple[int, BenchmarkMetrics]]:
    Config.mode = BenchMode(worker_mode)
    Config.warm_up = worker_warm_up
    Config.repetition = worker_repetition
    dtype = getattr(torch, dtype_name)

    bench = GenericBenchmark2DOnly(
        input_fn=rms_norm_input_fn,
        op_name="rms_norm",
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.to_bench_metrics = worker_metrics

    torch.cuda.set_device(gpu_id)
    local_results: List[Tuple[int, BenchmarkMetrics]] = []

    for shape_idx, shape in indexed_shapes:
        metric = BenchmarkMetrics()
        try:
            for input_item in rms_norm_input_fn(shape, dtype, f"cuda:{gpu_id}"):
                args, kwargs = bench.unpack_to_args_kwargs(input_item)
                metric.shape_detail = bench.record_shapes(*args, **kwargs)

                if "latency_base" in bench.to_bench_metrics:
                    metric.latency_base = bench.get_latency(bench.torch_op, *args, **kwargs)
                if "latency" in bench.to_bench_metrics:
                    with flag_gems.use_gems():
                        metric.latency = bench.get_latency(bench.torch_op, *args, **kwargs)
                if "speedup" in bench.to_bench_metrics:
                    metric.speedup = metric.latency_base / metric.latency
                if "gbps" in bench.to_bench_metrics:
                    metric.gbps_base = bench.get_gbps(args, latency=metric.latency_base)
                    metric.gbps = bench.get_gbps(args, latency=metric.latency)
                if "tflops" in bench.to_bench_metrics:
                    metric.tflops = (
                        bench.get_tflops(bench.torch_op, *args, **kwargs)
                        / metric.latency
                        / 1e12
                        * 1e3
                    )
        except Exception as e:
            metric.error_msg = str(e)
            raise
        finally:
            local_results.append((shape_idx, metric))
            gc.collect()

    return local_results


class ParallelRMSNormBenchmark(GenericBenchmark2DOnly):
    def run(self):
        if Config.query:
            super().run()
            return

        self.init_user_config()
        if not torch.cuda.is_available():
            pytest.skip("CUDA is required for parallel RMSNorm benchmark.")

        required_gpus = 8
        available_gpus = torch.cuda.device_count()
        if available_gpus < required_gpus:
            pytest.skip(
                f"Parallel RMSNorm benchmark requires at least {required_gpus} GPUs, found {available_gpus}."
            )

        worker_gpus = list(range(required_gpus))

        for dtype in self.to_bench_dtypes:
            shape_buckets: List[List[Tuple[int, Tuple[int, ...]]]] = [
                [] for _ in worker_gpus
            ]
            for idx, shape in enumerate(self.shapes):
                shape_buckets[idx % required_gpus].append((idx, shape))

            metrics: List[BenchmarkMetrics] = [BenchmarkMetrics() for _ in self.shapes]
            mp_ctx = mp.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=required_gpus, mp_context=mp_ctx
            ) as executor:
                futures = [
                    executor.submit(
                        _run_shapes_on_gpu,
                        bucket,
                        str(dtype).split(".")[-1],
                        gpu_id,
                        Config.mode.value,
                        Config.warm_up,
                        Config.repetition,
                        list(self.to_bench_metrics),
                    )
                    for gpu_id, bucket in zip(worker_gpus, shape_buckets)
                    if bucket
                ]

                for future in concurrent.futures.as_completed(futures):
                    for shape_idx, metric in future.result():
                        metrics[shape_idx] = metric

            result = BenchmarkResult(
                level=Config.bench_level.value,
                op_name=self.op_name,
                dtype=str(dtype),
                mode=Config.mode.value,
                result=metrics,
            )
            print(result)


@pytest.mark.rms_norm
@pytest.mark.skipif(
    SkipVersion("torch", "<2.4"),
    reason="The version prior to 2.4 does not include the rms_norm API in torch.",
)
def test_perf_rms_norm():
    bench = ParallelRMSNormBenchmark(
        input_fn=rms_norm_input_fn,
        op_name="rms_norm",
        torch_op=torch.nn.functional.rms_norm,
    )
    bench.run()
