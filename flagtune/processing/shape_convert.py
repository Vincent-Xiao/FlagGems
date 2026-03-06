#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from collections import OrderedDict
from pathlib import Path

LINE_PATTERN = re.compile(
    r"^(?P<op>[^,]+),\s*\[shape info\]:\s*\[(?P<shape>[^\]]+)\](?:\([^\)]*\))?,\s*\[count\]:\s*(?P<count>\d+)\s*$"
)


def normalize_op_key(full_op: str) -> str:
    op = full_op.strip()
    marker = ".ops."
    if marker in op:
        op = op.split(marker, 1)[1]

    parts = op.split(".")
    if len(parts) >= 2:
        module_name = parts[0]
        func_name = parts[-1]
        if module_name == "mm" and func_name == "general_mm":
            return "mm"
        return module_name

    return op


def parse_shape(raw_shape: str) -> list[int]:
    dims: list[int] = []
    for item in raw_shape.split(","):
        token = item.strip()
        if token == "-":
            dims.append(1)
        else:
            dims.append(int(token))
    return dims


def convert_shapes(input_path: Path) -> OrderedDict[str, list[list[int]]]:
    grouped: OrderedDict[str, list[list[int]]] = OrderedDict()
    seen: dict[str, set[tuple[int, ...]]] = {}

    with input_path.open("r", encoding="utf-8") as file:
        for line_no, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = LINE_PATTERN.match(line)
            if not match:
                raise ValueError(f"Cannot parse line {line_no}: {raw_line.rstrip()}")

            op_key = normalize_op_key(match.group("op"))
            shape = parse_shape(match.group("shape"))
            shape_key = tuple(shape)

            if op_key not in grouped:
                grouped[op_key] = []
                seen[op_key] = set()

            if shape_key not in seen[op_key]:
                grouped[op_key].append(shape)
                seen[op_key].add(shape_key)

    return grouped


def filter_by_op(
    grouped_shapes: OrderedDict[str, list[list[int]]], op_name: str | None
) -> OrderedDict[str, list[list[int]]]:
    if not op_name:
        return grouped_shapes

    if op_name not in grouped_shapes:
        raise ValueError(f"Operator '{op_name}' not found in input file")

    filtered: OrderedDict[str, list[list[int]]] = OrderedDict()
    filtered[op_name] = grouped_shapes[op_name]
    return filtered


def dump_shapes_yaml(grouped_shapes: OrderedDict[str, list[list[int]]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for op_idx, (op_name, shapes) in enumerate(grouped_shapes.items()):
            file.write(f"{op_name}:\n")
            file.write("  shapes:\n")
            for shape in shapes:
                file.write("  - - ")
                file.write(f"{shape[0]}\n")
                for dim in shape[1:]:
                    file.write(f"    - {dim}\n")
            if op_name == "mm":
                file.write("  shape_desc: B, M, N, K\n")
            if op_idx != len(grouped_shapes) - 1:
                file.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert shape text config to YAML format")
    parser.add_argument(
        "--model",
        default="qwen3.5",
        help="Model name, used to derive input/output file names",
    )
    parser.add_argument(
        "--op",
        default=None,
        help="Optional operator name filter (e.g. mm)",
    )
    args = parser.parse_args()

    shape_config_dir = Path(__file__).resolve().parent.parent / "shape-config"
    output_dir = shape_config_dir / "convert"
    model_name = args.model
    input_path = shape_config_dir / f"{model_name}.txt"
    if args.op:
        output_path = output_dir / f"{model_name}_{args.op}.yaml"
    else:
        output_path = output_dir / f"{model_name}.yaml"

    grouped_shapes = convert_shapes(input_path)
    grouped_shapes = filter_by_op(grouped_shapes, args.op)
    dump_shapes_yaml(grouped_shapes, output_path)


if __name__ == "__main__":
    main()
