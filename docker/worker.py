"""Private container worker. It never receives hidden cases, labels or API keys."""
import contextlib
import io
import json
import math
import resource
import sys
import time
import traceback
import numpy as np

MAX_BYTES = 32 * 1024 * 1024
MAX_PRINT = 32768


def limited_import(name, *args, **kwargs):
    if name not in {"numpy", "math"}:
        raise ImportError("Only numpy and math may be imported")
    return {"numpy": np, "math": math}[name]


allowed = {name: getattr(__builtins__, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "float", "int", "isinstance", "len", "list", "max", "min", "print", "range", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError")}
allowed["__import__"] = limited_import
namespace = {"np": np, "math": math, "__builtins__": allowed}

for raw in sys.stdin:
    started = time.process_time()
    try:
        if len(raw) > MAX_BYTES:
            raise ValueError("Command too large")
        request = json.loads(raw)
        if request["op"] == "init":
            namespace["weights"] = {k: np.asarray(v, dtype=np.float32) for k, v in request["weights"].items()}
            result = {"ready": True}
        elif request["op"] == "exec":
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                exec(compile(request["code"], "<solver>", "exec"), namespace, namespace)
            result = {"output": output.getvalue()[:MAX_PRINT]}
        elif request["op"] == "weights":
            arrays = namespace.get("weights")
            if not isinstance(arrays, dict):
                raise ValueError("weights must remain a dictionary")
            if sum(np.asarray(v).size for v in arrays.values()) > 25000:
                raise ValueError("Too many tensor values")
            result = {"weights": {k: np.asarray(v, dtype=np.float32).tolist() for k, v in arrays.items()}}
        else:
            raise ValueError("Unknown operation")
        result["cpu_seconds"] = time.process_time() - started
    except BaseException as exc:
        result = {"error": f"{type(exc).__name__}: {str(exc)[:2000]}", "cpu_seconds": time.process_time() - started}
    try:
        print(json.dumps(result, allow_nan=False), flush=True)
    except ValueError:
        print(json.dumps({"error": "Non-finite tensors cannot be exported"}), flush=True)
