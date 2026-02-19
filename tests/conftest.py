import json
import logging
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from time import perf_counter

import pytest
import torch

import flag_gems

device = flag_gems.device

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"test_detail_and_result_{timestamp}.json"


def pytest_addoption(parser):
    parser.addoption(
        "--ref",
        action="store",
        default=device,
        required=False,
        choices=[device, "cpu"],
        help="device to run reference tests on",
    )
    parser.addoption(
        (
            "--mode"
            if not (flag_gems.vendor_name == "kunlunxin" and torch.__version__ < "2.5")
            else "--fg_mode"
        ),  # TODO: fix pytest-* common --mode args,
        action="store",
        default="normal",
        required=False,
        choices=["normal", "quick"],
        help="run tests on normal or quick mode",
    )
    parser.addoption(
        "--record",
        action="store",
        default="none",
        required=False,
        choices=["none", "log"],
        help="tests function param recorded in log files or not",
    )
    parser.addoption(
        "--parallel",
        action="store",
        type=int,
        default=0,
        required=False,
        help=(
            "Enable multi-GPU parallel execution for accuracy tests. "
            "Example: --parallel 8 means using GPU 0~7 in parallel. "
            "Default 0 means serial execution."
        ),
    )


def _worker_rank_from_id(worker_id: str) -> int:
    if not worker_id:
        return 0
    if worker_id.startswith("gw"):
        return int(worker_id[2:])
    return 0


def _strip_parallel_args(args):
    stripped = []
    i = 0
    while i < len(args):
        if args[i] == "--parallel":
            i += 2
            continue
        stripped.append(args[i])
        i += 1
    return stripped


def _run_worker_pytest(worker_args, rank, world_size):
    env = os.environ.copy()
    env["FLAGGEMS_ACCURACY_PARALLEL_WORKER"] = "1"
    env["FLAGGEMS_PARALLEL_RANK"] = str(rank)
    env["FLAGGEMS_PARALLEL_WORLD_SIZE"] = str(world_size)
    env["CUDA_VISIBLE_DEVICES"] = str(rank)
    cmd = [sys.executable, "-m", "pytest", *worker_args, "--parallel", "0"]
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _collect_summary_counts(stdout_text):
    summary_match = re.search(r"=+\s*(.+?)\s*in\s*([0-9.]+)s\s*=+", stdout_text)
    if not summary_match:
        return {}, 0.0

    counters = {}
    for count, name in re.findall(
        r"(\d+)\s+(passed|failed|skipped|deselected|xfailed|xpassed|errors?|warnings?)",
        summary_match.group(1),
    ):
        if name.startswith("error"):
            key = "error"
        elif name.startswith("warning"):
            key = "warning"
        else:
            key = name
        counters[key] = counters.get(key, 0) + int(count)
    return counters, float(summary_match.group(2))


def _is_empty_shard_result(result):
    if result.returncode != 5:
        return False
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    return (
        "0 selected" in text
        or "no tests ran" in text
        or "deselected" in text
    )


def _extract_first_match(lines, prefix):
    for line in lines:
        if line.startswith(prefix):
            return line
    return None


def pytest_configure(config):
    global TO_CPU
    TO_CPU = config.getoption("--ref") == "cpu"

    global QUICK_MODE
    QUICK_MODE = config.getoption("--mode") == "quick"

    global RECORD_LOG
    RECORD_LOG = config.getoption("--record") == "log"

    global PARALLEL
    PARALLEL = int(config.getoption("--parallel") or 0)

    if PARALLEL > 1 and os.environ.get("FLAGGEMS_ACCURACY_PARALLEL_WORKER") != "1":
        if not torch.cuda.is_available():
            raise pytest.UsageError("--parallel N requires CUDA environment.")
        available_gpus = torch.cuda.device_count()
        if available_gpus < PARALLEL:
            raise pytest.UsageError(
                f"--parallel requires at least {PARALLEL} GPUs, found {available_gpus}."
            )

    if os.environ.get("FLAGGEMS_ACCURACY_PARALLEL_WORKER") == "1":
        rank = int(os.environ.get("FLAGGEMS_PARALLEL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    if RECORD_LOG:
        global RUNTEST_INFO, BUILTIN_MARKS, REGISTERED_MARKERS
        RUNTEST_INFO = {}
        BUILTIN_MARKS = {
            "parametrize",
            "skip",
            "skipif",
            "xfail",
            "usefixtures",
            "filterwarnings",
            "timeout",
            "tryfirst",
            "trylast",
        }
        REGISTERED_MARKERS = {
            marker.split(":")[0].strip() for marker in config.getini("markers")
        }
        cmd_args = [
            arg.replace(".py", "").replace("=", "_").replace("/", "_")
            for arg in config.invocation_params.args
        ]
        logging.basicConfig(
            filename="result_{}.log".format("_".join(cmd_args)).replace("_-", "-"),
            filemode="w",
            level=logging.INFO,
            format="[%(levelname)s] %(message)s",
        )


def pytest_cmdline_main(config):
    parallel = int(config.getoption("--parallel") or 0)
    if parallel <= 1:
        return None
    if os.environ.get("FLAGGEMS_ACCURACY_PARALLEL_WORKER") == "1":
        return None

    worker_args = _strip_parallel_args(list(config.invocation_params.args))
    start_time = perf_counter()
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = [
            executor.submit(_run_worker_pytest, worker_args, rank, parallel)
            for rank in range(parallel)
        ]
        worker_results = [future.result() for future in futures]

    final_code = 0
    merged = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "xpassed": 0,
        "error": 0,
    }
    max_warnings = 0
    first_worker_lines = worker_results[0].stdout.splitlines() if worker_results else []
    for rank, result in enumerate(worker_results):
        if result.returncode != 0 and not _is_empty_shard_result(result):
            final_code = result.returncode

        counters, _ = _collect_summary_counts(result.stdout)
        for k in merged:
            merged[k] += counters.get(k, 0)
        max_warnings = max(max_warnings, counters.get("warning", 0))

    if final_code == 0:
        elapsed = perf_counter() - start_time
        session_line = _extract_first_match(first_worker_lines, "=============================")
        platform_line = _extract_first_match(first_worker_lines, "platform ")
        rootdir_line = _extract_first_match(first_worker_lines, "rootdir:")
        config_line = _extract_first_match(first_worker_lines, "configfile:")

        total_items = (
            merged["passed"]
            + merged["failed"]
            + merged["skipped"]
            + merged["xfailed"]
            + merged["xpassed"]
            + merged["error"]
        )
        target_path = "tests"
        for arg in worker_args:
            if not arg.startswith("-"):
                target_path = arg.split("::")[0]
                break

        progress = ""
        if merged["passed"] > 0:
            progress += "." * merged["passed"]
        if merged["failed"] > 0:
            progress += "F" * merged["failed"]
        if merged["skipped"] > 0:
            progress += "s" * merged["skipped"]
        if merged["xfailed"] > 0:
            progress += "x" * merged["xfailed"]
        if merged["xpassed"] > 0:
            progress += "X" * merged["xpassed"]
        if merged["error"] > 0:
            progress += "E" * merged["error"]

        if session_line:
            print(session_line)
        if platform_line:
            print(platform_line)
        if rootdir_line:
            print(rootdir_line)
        if config_line:
            print(config_line)
        print(f"collected {total_items} items")
        print()
        print(f"{target_path} {progress}")
        print()

        summary_parts = []
        for key in ("passed", "failed", "skipped", "xfailed", "xpassed", "error"):
            if merged[key] > 0:
                label = "errors" if key == "error" and merged[key] > 1 else key
                summary_parts.append(f"{merged[key]} {label}")
        if max_warnings > 0:
            summary_parts.append(f"{max_warnings} warnings")
        print(
            "=" * 23
            + f" {', '.join(summary_parts)} in {elapsed:.2f}s "
            + "=" * 23
        )
    else:
        for rank, result in enumerate(worker_results):
            if result.stdout:
                print(f"\n[parallel-worker-{rank}]\n{result.stdout}", end="")
            if result.stderr:
                print(
                    f"\n[parallel-worker-{rank}-stderr]\n{result.stderr}",
                    end="",
                    file=sys.stderr,
                )
    return final_code


def pytest_collection_modifyitems(config, items):
    if os.environ.get("FLAGGEMS_ACCURACY_PARALLEL_WORKER") != "1":
        return
    rank = int(os.environ.get("FLAGGEMS_PARALLEL_RANK", "0"))
    world_size = int(os.environ.get("FLAGGEMS_PARALLEL_WORLD_SIZE", "1"))
    if world_size <= 1:
        return

    selected = [item for idx, item in enumerate(items) if idx % world_size == rank]
    deselected = [item for idx, item in enumerate(items) if idx % world_size != rank]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_runtest_teardown(item, nextitem):
    if not RECORD_LOG:
        return
    if hasattr(item, "callspec"):
        all_marks = list(item.iter_markers())
        op_marks = [
            mark.name
            for mark in all_marks
            if mark.name not in BUILTIN_MARKS and mark.name not in REGISTERED_MARKERS
        ]
        if len(op_marks) > 0:
            params = str(item.callspec.params)
            for op_mark in op_marks:
                if op_mark not in RUNTEST_INFO:
                    RUNTEST_INFO[op_mark] = [params]
                else:
                    RUNTEST_INFO[op_mark].append(params)
        else:
            func_name = item.function.__name__
            logging.warning("There is no mark at {}".format(func_name))


def pytest_sessionfinish(session, exitstatus):
    if RECORD_LOG:
        logging.info(json.dumps(RUNTEST_INFO, indent=2))


test_results = {}


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    test_results[item.nodeid] = {"params": None, "result": None, "opname": None}
    param_values = {}
    request = item._request
    if hasattr(request, "node") and hasattr(request.node, "callspec"):
        param_values = request.node.callspec.params

    test_results[item.nodeid]["params"] = param_values
    # get all mark
    all_marks = [mark.name for mark in item.iter_markers()]
    # exclude marks，such as parametrize、skipif and so on
    exclude_marks = {"parametrize", "skip", "skipif", "xfail", "usefixtures", "inplace"}
    operator_marks = [mark for mark in all_marks if mark not in exclude_marks]
    test_results[item.nodeid]["opname"] = operator_marks


def get_skipped_reason(report):
    if hasattr(report.longrepr, "reprcrash"):
        return report.longrepr.reprcrash.message
    elif isinstance(report.longrepr, tuple):
        return report.longrepr[2]
    else:
        return str(report.longrepr)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_logreport(report):
    if report.when == "setup":
        if report.outcome == "skipped":
            reason = get_skipped_reason(report)
            test_results[report.nodeid]["result"] = "skipped"
            test_results[report.nodeid]["skipped_reason"] = reason

    elif report.when == "call":
        test_results[report.nodeid]["result"] = report.outcome
        if report.outcome == "skipped":
            reason = get_skipped_reason(report)
            test_results[report.nodeid]["skipped_reason"] = reason
        else:
            test_results[report.nodeid]["skipped_reason"] = None


def pytest_terminal_summary(terminalreporter):
    if os.environ.get("FLAGGEMS_ACCURACY_PARALLEL_WORKER") == "1":
        return

    if os.path.exists(filename):
        with open(filename, "r") as json_file:
            existing_data = json.load(json_file)
        existing_data.update(test_results)
    else:
        existing_data = test_results

    with open("result.json", "w") as json_file:
        json.dump(existing_data, json_file, indent=4, default=str)
