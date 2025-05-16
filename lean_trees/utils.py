import asyncio
import datetime
import functools
from collections import Counter
from enum import Enum
from typing import Set, TypeVar, Callable, Optional, AsyncIterator
import argparse
from pathlib import Path
import json
import gc
import os
import random
import re
import tempfile
import math

import numpy as np
import torch
from transformers import BatchEncoding
from scipy.stats import norm
from PrettyPrint import PrettyPrintTree

from lean_trees.file_span import FileSpan, FilePosition

def to_sync(func):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If there's no event loop in the current thread, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return loop.run_until_complete(func(*args, **kwargs))
    return wrapper

class AsyncToSyncIterator:
    """
    A synchronous iterator wrapper for an asynchronous iterator.
    
    This class allows asynchronous iterators to be used in synchronous contexts
    by converting async iteration to sync iteration.
    """
    def __init__(self, async_iter: AsyncIterator, loop: asyncio.AbstractEventLoop):
        self.async_iter = async_iter
        self.loop = loop
        
    def __iter__(self):
        return self
        
    def __next__(self):
        try:
            # Use run_until_complete to get the next item synchronously
            return self.loop.run_until_complete(self.async_iter.__anext__())
        except StopAsyncIteration:
            raise StopIteration

def to_sync_iterator(func):
    """
    Decorator to convert an async iterator function to a sync iterator function.
    
    This decorator takes an async function that returns an AsyncIterator and
    converts it to a synchronous function that returns a regular Iterator.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            # If there's no event loop in the current thread, create a new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        async_iter = func(*args, **kwargs)
        return AsyncToSyncIterator(async_iter, loop)
    
    return wrapper

def tree_to_str[TNode, TEdge](root: TNode, get_edge: Callable[[TNode], Optional[TEdge]]) -> str:
    max_steps_per_line = 4
    result = ""
    for layer in get_tree_layers(root, get_edge):
        for i in range(0, len(layer), max_steps_per_line):
            serialized = []
            for node in layer[i:i + max_steps_per_line]:
                edge = get_edge(node)
                if edge is not None:
                    serialized.append(str(edge.step))
                else:
                    serialized.append(f"Node {str(node.goal)}\n{node.state.value}")
            result += concat_horizontally(serialized)
        result += "\n"
    return boxed(result)


def get_tree_layers[TNode, TEdge](root: TNode, get_edge: Callable[[TNode], Optional[TEdge]]) -> list[list[TNode]]:
    layers = [[]]
    layer_remaining = [root]
    next_layer = []
    seen = set()
    while len(layer_remaining) > 0:
        node = layer_remaining.pop()
        if node in seen:
            continue
        seen.add(node)
        layers[-1].append(node)
        edge = get_edge(node)
        if edge is None:
            continue
        for child in edge.children:
            next_layer.append(child)

        if len(layer_remaining) == 0:
            layer_remaining = next_layer
            next_layer = []
            if len(layer_remaining) != 0:
                layers.append([])
    return layers


def gaussian_bins(mean: float, sigma: float, num_bins: int) -> list[float]:
    """
    Generates a distribution over bins based on a Gaussian distribution.
    """
    assert 0 <= mean <= 1

    edges = np.linspace(0, 1, num_bins + 1)
    cdf_values = norm.cdf(edges, loc=mean, scale=sigma)

    # Normalize the CDF to ensure it spans from 0 to 1 over [0, 1]
    cdf_normalized = (cdf_values - cdf_values[0]) / (cdf_values[-1] - cdf_values[0])

    bin_probs = np.diff(cdf_normalized)
    return list(bin_probs)


def format_distribution(bins: list[float], hist_height: int = 10, bin_labels: list[str] = None) -> str:
    bar_char = '❚'  # Heavy vertical bar character.

    num_bins = len(bins)
    max_bin = max(bins)
    result = ""

    if max_bin == 0:
        max_bin = 1  # To avoid division by zero; all bars will be zero height.

    scaled_bins = [(bin_value / max_bin) * hist_height for bin_value in bins]
    # Round up to ensure visibility of non-zero bins.
    bar_heights = [math.ceil(height) for height in scaled_bins]

    # Determine y-axis labels (from HIST_HEIGHT down to 1)
    for row in range(hist_height, 0, -1):
        label_value = (row / hist_height) * max_bin
        label = f"{label_value:>3.1f} |"
        row_str = label
        for height in bar_heights:
            if height >= row:
                row_str += f" {bar_char} "
            else:
                row_str += " " * 3
        result += row_str + "\n"

    x_axis = "    +" + "---" * num_bins
    result += x_axis + "\n"

    # x-axis labels.
    if not bin_labels:
        bin_labels = [f"{i}" for i in range(num_bins)]
    label_str = "     "
    for label in bin_labels:
        assert len(label) <= 2
        if len(label) == 1:
            label_str += f" {label} "
        else:
            label_str += f"{label} "
    result += label_str + "\n"
    return result


def concat_horizontally(strings: list[str]) -> str:
    strings = [s.replace("\t", " " * 4) for s in strings]
    all_lines = [s.split("\n") for s in strings]
    max_line_width = max(len(line) for lines in all_lines for line in lines)
    column_width = max_line_width + 5

    concat_lines = []
    for i in range(max(len(lines) for lines in all_lines)):
        concat_line = ""
        for lines in all_lines:
            if len(lines) <= i:
                concat_line += " " * column_width
            else:
                concat_line += lines[i] + " " * (column_width - len(lines[i]))
        concat_lines.append(concat_line)
    return "\n".join(concat_lines) + "\n"


def boxed(s: str) -> str:
    lines = s.split("\n")
    max_width = max(len(line) for line in lines)

    return "\n".join([
        "⎡" + "-" * max_width + "⎤",
        *["|" + line + " " * (max_width - len(line)) + "|" for line in lines],
        "⎣" + "-" * max_width + "⎦",
    ])


def pretty_print_tree[TypeNode](
        root: TypeNode,
        get_children: Callable[[TypeNode], list[TypeNode]],
        node_to_str: Callable[[TypeNode], str],
        edge_to_str: Callable[[TypeNode], str | None] | None = None,
        max_label_len=55,
        max_edge_label_len=None,
) -> str:
    def trimmed_edge_to_str(e: TypeNode) -> str | None:
        if edge_to_str is None:
            return None
        s = edge_to_str(e)
        if max_edge_label_len is None:
            return s
        if s is None:
            return s
        if len(s) > max_edge_label_len:
            dots = "..."
            return s[:max_edge_label_len - len(dots)] + dots
        return s

    pt = PrettyPrintTree(
        get_children=get_children,
        get_val=node_to_str,
        get_label=trimmed_edge_to_str,
        return_instead_of_print=True,
        # border=True,
        trim=max_label_len,
    )
    return pt(root)


def byte_size_to_human(size_bytes):
    if size_bytes == 0:
        return "0 B"
    power_names = ["B", "KB", "MB", "GB", "TB"]
    power_idx = int(math.floor(math.log(size_bytes, 1024)))
    power = math.pow(1024, power_idx)
    size = round(size_bytes / power, 2)
    return f"{size} {power_names[power_idx]}"


def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_paths(args: argparse.Namespace):
    models_home = os.getenv("ARCOSS_MODELS_HOME")
    data_home = os.getenv("ARCOSS_DATA_HOME")

    for attr_name, attr_value in vars(args).items():
        if isinstance(attr_value, Path):
            if attr_value.is_absolute():
                continue

            if models_home and "[MODELS]" in str(attr_value):
                attr_value = Path(str(attr_value).replace("[MODELS]", models_home))
            if data_home and not attr_value.is_absolute():
                attr_value = Path(data_home) / attr_value
            setattr(args, attr_name, attr_value)


def get_args_descriptor(
        args_ns: argparse.Namespace,
        *args,
        **kwargs,
):
    return get_dict_descriptor(vars(args_ns), *args, **kwargs)


def get_dict_descriptor(
        args: dict,
        param_blacklist: Set[str] | None = None,
        param_whitelist: Set[str] | None = None,
        extra_args: dict[str, object] | None = None,
        include_slurm_id=True,
        include_time=True,
) -> str:
    if include_time:
        descriptor = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    else:
        descriptor = ""

    if include_slurm_id and "SLURM_JOB_ID" in os.environ:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += f"id={os.environ['SLURM_JOB_ID']}"

    visible_args = {k: v for k, v in sorted(args.items())}
    if param_blacklist is not None:
        visible_args = {k: v for k, v in visible_args.items() if k not in param_blacklist}
    if param_whitelist is not None:
        visible_args = {k: v for k, v in visible_args.items() if k in param_whitelist}

    if extra_args is not None:
        visible_args = {**visible_args, **extra_args}

    def format_value(v: str) -> str:
        if isinstance(v, Path) or "/" in str(v):
            v = str(v)
            if v.endswith("/"):
                v = v[:-1]
            parts = [p for p in v.split("/") if len(p) != 0]
            return "_".join([v[:50] for v in parts[-2:]])
        if isinstance(v, str):
            return v.replace("<", "").replace(">", "")
        return str(v)

    if len(visible_args) > 0:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += ",".join((
            "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), format_value(v))
            for k, v in visible_args.items()
        ))

    assert len(descriptor) > 0

    return descriptor


def setup_output_dirs(args: argparse.Namespace, param_blacklist: Set[str]):
    descriptor = get_args_descriptor(args, param_blacklist=param_blacklist)

    if args.test:
        tmpdir = Path(tempfile.mkdtemp())
        args.log_dir = tmpdir / "tmp_train_logs" / descriptor
        args.models_dir = tmpdir / "tmp_models" / descriptor
    else:
        if not args.log_dir:
            args.log_dir = args.base_log_dir / descriptor
        if not args.models_dir:
            args.models_dir = args.base_models_dir / descriptor

    os.makedirs(args.log_dir, exist_ok=True)


def dump_args(args, logdir):
    path = os.path.join(logdir, "args.json")
    with open(path, "w") as f:
        data = {k: str(v) for k, v in args.__dict__.items()}
        json.dump(data, f, indent=4, sort_keys=True)
        f.write("\n")


def deep_shape(obj, seen=None, level=0, pretty=False):
    if seen is None:
        seen = set()
    if id(obj) in seen:
        return "<circular reference>"
    seen.add(id(obj))

    def join_parts(parts):
        if pretty:
            return "\n" + "  " * level + (",\n" + "  " * level).join(parts) + "\n" + "  " * (level - 1)
        return ", ".join(parts)

    if isinstance(obj, tuple):
        return "(" + join_parts([deep_shape(o, seen, level + 1, pretty) for o in obj]) + ")"
    if isinstance(obj, list):
        if all(isinstance(o, (int, float, str, bool, type(None))) for o in obj):
            type_counts = Counter(type(o).__name__ for o in obj)
            return f"[{', '.join(f'{k}-{v}' for k, v in type_counts.items())}]"
        return "[" + join_parts([deep_shape(o, seen, level + 1, pretty) for o in obj]) + "]"
    if isinstance(obj, dict):
        return "{" + join_parts([str(k) + ": " + deep_shape(v, seen, level + 1, pretty) for k, v in obj.items()]) + "}"
    if isinstance(obj, BatchEncoding):
        return "BatchEncoding-" + deep_shape(obj.data, seen, level, pretty)
    if isinstance(obj, np.ndarray):
        return "np-" + str(obj.shape)
    if isinstance(obj, torch.Tensor):
        return "pt-" + str(tuple(obj.shape))
    if isinstance(obj, str):
        return "str-" + str(len(obj))
    return str(obj)


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def strict_zip(a: list, b: list):
    if len(a) != len(b):
        raise Exception(f"List sizes differ ({len(a)} != {len(b)}).")
    return zip(a, b)


def same_up_to_ordering(arr1: list, arr2: list) -> bool:
    arr2 = [b for b in arr2]
    for a in arr1:
        for i in range(len(arr2)):
            if a is arr2[i]:
                del arr2[i]
                break
        else:
            return False
    return len(arr2) == 0


def print_torch_memory_summary():
    print(torch.cuda.memory_summary())


def print_torch_memory_allocated():
    print(f"Allocated: {byte_size_to_human(torch.cuda.memory_allocated())}\t"
          f"Max Allocated: {byte_size_to_human(torch.cuda.max_memory_allocated())}")


# Unix-only; must run in the main thread; only works in a single process interpreter.
def timeout(func, timeout_seconds: int):
    import signal

    def handler(signum, frame):
        raise TimeoutError()

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_seconds)
    try:
        return func()
    finally:
        signal.alarm(0)


class MetricsType:
    Number = 0
    Histogram = 1

    @staticmethod
    def write_to_tensorboard(writer, metric_name, metric, epoch_num):
        if isinstance(metric, tuple):
            metric_value, metric_type = metric
        else:
            metric_value, metric_type = metric, MetricsType.Number

        if metric_type == MetricsType.Number:
            writer.add_scalar(metric_name, metric_value, epoch_num)
        elif metric_type == MetricsType.Histogram:
            writer.add_histogram(metric_name, metric_value, epoch_num)
        else:
            raise Exception("Unknown metric type.")


def average_metrics(all_dicts: list[dict]):
    def get_value(x):
        if isinstance(x, tuple):
            x = x[0]
        if isinstance(x, torch.Tensor):
            return x.cpu()
        return x

    avg_metrics = {}
    for metric in all_dicts[0].keys():
        if isinstance(all_dicts[0][metric], tuple):
            type_ = all_dicts[0][metric][1]
        else:
            type_ = MetricsType.Number

        if type_ == MetricsType.Histogram:
            avg_metrics[metric] = (
                np.concatenate([get_value(m[metric]).flatten() for m in all_dicts]),
                MetricsType.Histogram,
            )
        else:
            avg_metrics[metric] = np.mean([get_value(m[metric]) for m in all_dicts])
    return avg_metrics


SomeValue = TypeVar("SomeValue")


class ValueOrError[SomeValue]:
    def __init__(self, value: SomeValue | None, error: str | None):
        assert (value is None) != (error is None)
        # TODO: rename to _value, _error + refactor any public usages
        self.value = value
        self.error = error

    # TODO: rename to from_success, from_error
    @staticmethod
    def success(value: SomeValue) -> "ValueOrError":
        return ValueOrError(value, None)

    @staticmethod
    def error(error: str) -> "ValueOrError":
        return ValueOrError(None, error)

    def is_success(self) -> bool:
        return self.value is not None

    # TODO: rename to `value`, make it a property
    def get_value(self) -> SomeValue:
        assert self.is_success()
        return self.value


class LogLevel(Enum):
    SUPPRESS = 0
    SUPPRESS_AND_STORE = 1
    INFO = 2
    DEBUG = 3


# TODO: replace with something built-in
class Logger:
    def __init__(self, log_level: LogLevel):
        self.log_level = log_level
        self._stored_messages = None
        if log_level == LogLevel.SUPPRESS_AND_STORE:
            self._stored_messages = []

    def info(self, msg: str):
        if self.log_level in [LogLevel.INFO, LogLevel.DEBUG]:
            print(msg)
        elif self.log_level == LogLevel.SUPPRESS_AND_STORE:
            self._stored_messages.append((msg, LogLevel.INFO))

    warning = info

    def debug(self, msg: str):
        if self.log_level == LogLevel.DEBUG:
            print(msg)
        elif self.log_level == LogLevel.SUPPRESS_AND_STORE:
            self._stored_messages.append((msg, LogLevel.DEBUG))

    def print_stored(self, log_level: LogLevel):
        assert log_level in [LogLevel.INFO, LogLevel.DEBUG]
        for msg, msg_level in self._stored_messages:
            if log_level.value >= msg_level.value:
                print(msg)
        self.delete_stored()

    def delete_stored(self):
        assert self._stored_messages is not None
        self._stored_messages = []


class NullLogger(Logger):
    def __init__(self):
        super().__init__(LogLevel.SUPPRESS)

    def info(self, msg: str):
        pass

    def debug(self, msg: str):
        pass

    def print_stored(self, log_level: LogLevel):
        pass

    def delete_stored(self):
        pass


# TODO: unit test
# TODO: fix this
def remove_comments(source: str) -> str:
    inside_comment = False
    result = []
    for line in source.splitlines():
        result_line = ""
        to_process = line
        while len(to_process) > 0:
            relevant_tokens = ["-/"] if inside_comment else ["--", "/-"]
            indices = [to_process.index(tok) for tok in relevant_tokens if tok in to_process]
            if not indices:
                if not inside_comment:
                    result_line += to_process
                break
            first_idx = min(indices)
            match to_process[first_idx:first_idx + 2]:
                case "-/":
                    inside_comment = False
                    to_process = to_process[first_idx + 2:]
                case "/-":
                    inside_comment = True
                    result_line += to_process[:first_idx]
                    to_process = to_process[first_idx + 2:]
                case "--":
                    result_line += to_process[:first_idx]
                    to_process = ""
        if inside_comment and not result_line:
            continue
        result.append(result_line)
    return "\n".join(result)


def remove_empty_lines(s: str) -> str:
    return "\n".join([l for l in s.splitlines() if l.strip()])


def is_just_comments(s: str) -> bool:
    return remove_empty_lines(remove_comments(s)).strip() == ""

def replace_with_sorries(theorem_str: str, sorries_mask: list[FileSpan]) -> str:
    return get_source_with_sorries(
        FileSpan.whole_file(theorem_str),
        sorries_mask,
        theorem_str,
    )

def get_source_with_sorries(
        span: FileSpan,
        sorries_mask: list[FileSpan] | None,
        file_content: str | None = None,
        file_path: Path | str | None = None,
) -> str:
    if file_content is None:
        with file_path.open("r", encoding="utf-8") as f:
            file_content = f.read()
    if not sorries_mask:
        return span.read_from_string(file_content)
    result = ""
    curr_position = span.start
    for mask_span in sorted(sorries_mask, key=lambda s: s.start):
        assert curr_position <= mask_span.start <= mask_span.finish <= span.finish
        result += FileSpan(curr_position, mask_span.start).read_from_string(file_content)
        result += "sorry"
        curr_position = mask_span.finish
    result += FileSpan(curr_position, span.finish).read_from_string(file_content)
    return result
