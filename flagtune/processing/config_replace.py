#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_list(text: str, key: str) -> list[int]:
    pattern = re.compile(rf"^\s*{re.escape(key)}:\s*$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Cannot find key: {key}")

    values: list[int] = []
    lines = text[match.end() :].splitlines()
    for line in lines:
        if not line.strip():
            continue
        if re.match(r"^\s*[A-Za-z_][A-Za-z0-9_]*:\s*$", line):
            break
        item_match = re.match(r"^\s*-\s*(\d+)\s*$", line)
        if item_match:
            values.append(int(item_match.group(1)))
        elif values:
            break
    if not values:
        raise ValueError(f"No values found for key: {key}")
    return values


def list_literal(values: list[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def replace_matmul_get_configs(tune_yaml_path: Path, mm_py_path: Path) -> None:
    tune_text = tune_yaml_path.read_text(encoding="utf-8")
    mm_text = mm_py_path.read_text(encoding="utf-8")

    bm_values = parse_list(tune_text, "block_m")
    bn_values = parse_list(tune_text, "block_n")
    bk_values = parse_list(tune_text, "block_k")
    s_values = parse_list(tune_text, "stages")
    w_values = parse_list(tune_text, "warps")

    matmul_fn_pattern = re.compile(
        r"(def\s+matmul_get_configs\(pre_hook=matmul_tma_set_block_size_hook\):\n\s*return\s*\[\n(?:.|\n)*?\n\s*\])",
        re.MULTILINE,
    )
    matmul_match = matmul_fn_pattern.search(mm_text)
    if not matmul_match:
        raise ValueError("Cannot locate matmul_get_configs function block")

    new_block = """def matmul_get_configs(pre_hook=matmul_tma_set_block_size_hook):
    return [
        triton.Config(
            {{\"BLOCK_M\": BM, \"BLOCK_N\": BN, \"BLOCK_K\": BK}},
            num_stages=s,
            num_warps=w,
            pre_hook=pre_hook,
        )
        for BM in {bm}
        for BN in {bn}
        for BK in {bk}
        for s in {stages}
        for w in {warps}
    ]""".format(
        bm=list_literal(bm_values),
        bn=list_literal(bn_values),
        bk=list_literal(bk_values),
        stages=list_literal(s_values),
        warps=list_literal(w_values),
    )

    updated = mm_text[: matmul_match.start()] + new_block + mm_text[matmul_match.end() :]
    mm_py_path.write_text(updated, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replace matmul_get_configs by tune yaml")
    parser.add_argument("--tune-yaml", required=True, help="Path to tune yaml file")
    parser.add_argument("--target", required=True, help="Path to mm.py target file")
    args = parser.parse_args()

    replace_matmul_get_configs(Path(args.tune_yaml), Path(args.target))
    print("[PATCH] Updated matmul_get_configs from tune yaml")


if __name__ == "__main__":
    main()
