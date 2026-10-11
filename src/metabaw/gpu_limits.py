"""PyTorch/CuPy GPU caps; copied as sitecustomize for Python children.

Standard-library-only and Python 3.7 compatible (COMEBin's environment).
No CUDA initialization is performed in Python workers that do not use CUDA.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import sys
import threading


def install_torch_limit(torch, budget):
    cuda = torch.cuda
    setter = getattr(cuda, "set_per_process_memory_fraction", None)
    if setter is None:
        raise RuntimeError("PyTorch lacks set_per_process_memory_fraction; cannot enforce the GPU budget")
    original = cuda._lazy_init
    lock = threading.RLock()
    state = {"applying": False, "applied": False, "pid": os.getpid()}

    def initialize():
        nonlocal lock
        original()
        if state["pid"] != os.getpid():
            lock = threading.RLock()
            state.update(applying=False, applied=False, pid=os.getpid())
        with lock:
            if state["applied"] or state["applying"]:
                return
            state["applying"] = True
            try:
                total = cuda.get_device_properties(0).total_memory
                fraction = min(0.8, float(budget) / total)
                if fraction <= 0:
                    raise RuntimeError("Invalid GPU task share")
                setter(fraction, device=0)
                state["applied"] = True
                print("[GPU LIMIT] device=0 (assigned visible GPU); allocator budget=%s bytes; fraction=%.6f"
                      % (budget, fraction), flush=True)
            finally:
                state["applying"] = False

    cuda._lazy_init = initialize

    def bounded_setter(fraction, device=None):
        initialize()
        if device not in (None, 0, "cuda", "cuda:0") and str(device) not in ("cuda", "cuda:0", "0"):
            raise RuntimeError("A GPU task may only use its assigned visible device")
        cap = min(0.8, float(budget) / cuda.get_device_properties(0).total_memory)
        return setter(min(float(fraction), cap), device=0)

    cuda.set_per_process_memory_fraction = bounded_setter
    memory = getattr(cuda, "memory", None)
    if memory is not None:
        memory.set_per_process_memory_fraction = bounded_setter


def install_cupy_limit(cupy, budget):
    """MetaDecoder uses CuPy's default device memory pool, not PyTorch."""
    pool = cupy.get_default_memory_pool()
    if not hasattr(pool, "set_limit"):
        raise RuntimeError("CuPy lacks memory-pool limit control")
    with cupy.cuda.Device(0):
        pool.set_limit(size=int(budget))
    print("[GPU LIMIT] CuPy device=0 (assigned visible GPU); pool budget=%s bytes"
          % budget, flush=True)


class LimitedLoader:
    def __init__(self, loader, budget):
        self.loader, self.budget = loader, budget

    def create_module(self, spec):
        return self.loader.create_module(spec) if hasattr(self.loader, "create_module") else None

    def exec_module(self, module):
        self.loader.exec_module(module)
        try:
            if module.__name__ == "torch":
                install_torch_limit(module, self.budget)
            else:
                install_cupy_limit(module, self.budget)
        except Exception as exc:
            # MetaDecoder catches ordinary CuPy import errors and silently
            # selects CPU. Failure to enforce a selected GPU budget is fatal.
            print("[GPU LIMIT ERROR] %s: %s" % (module.__name__, exc), file=sys.stderr, flush=True)
            raise SystemExit(78)


class GpuLimitFinder(importlib.abc.MetaPathFinder):
    def __init__(self, budget):
        self.budget = budget

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ("torch", "cupy"):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = LimitedLoader(spec.loader, self.budget)
        return spec


def install():
    value = os.environ.get("METABAW_GPU_BUDGET_BYTES")
    if not value:
        return
    budget = int(os.environ.get("METABAW_GPU_ALLOCATOR_BYTES", value))
    if budget <= 0:
        raise ValueError("Non-positive MetaBAW GPU budget")
    os.environ["CUPY_GPU_MEMORY_LIMIT"] = str(budget)
    for name, installer in (("torch", install_torch_limit), ("cupy", install_cupy_limit)):
        if name in sys.modules:
            installer(sys.modules[name], budget)
    sys.meta_path.insert(0, GpuLimitFinder(budget))


if __name__ == "sitecustomize":
    try:
        install()
        # Preserve a pre-existing sitecustomize instead of shadowing user setup.
        directory = os.path.dirname(os.path.abspath(__file__))
        paths = [path for path in sys.path if os.path.abspath(path or os.curdir) != directory]
        spec = importlib.machinery.PathFinder.find_spec("sitecustomize", paths)
        if spec is not None and spec.loader is not None and hasattr(spec.loader, "get_code"):
            code = spec.loader.get_code("sitecustomize")
            if code is not None:
                exec(code, {"__name__": "sitecustomize", "__file__": spec.origin,
                            "__spec__": spec, "__builtins__": __builtins__})
    except BaseException as exc:
        print("[GPU LIMIT ERROR] Cannot install GPU resource control: %s" % exc,
              file=sys.stderr, flush=True)
        os._exit(78)
