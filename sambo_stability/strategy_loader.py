# strategy_loader.py
from __future__ import annotations
import importlib
import importlib.util
import sys, os
from pathlib import Path
from typing import Union

def load_strategy(strategy_path: Union[str, os.PathLike], strategy_class: str):
    """
    Load a Strategy class from either:
      - a module import path (e.g. "sambo_stability.NewStrat2"), or
      - a filesystem .py path (e.g. ROOT/'strategies/Strat_SMA200.py').

    Parameters
    ----------
    strategy_path : str | PathLike
        Module import path OR a .py file path.
    strategy_class : str
        Class name to import from that module/file.
    """
    # Normalize to string and Path
    path_obj = Path(strategy_path) if not isinstance(strategy_path, str) else Path(strategy_path)
    spath = str(strategy_path)

    # Decide: file path vs module path
    looks_like_file = (
        isinstance(strategy_path, os.PathLike)
        or path_obj.suffix == ".py"
        or any(sep in spath for sep in ("/", "\\"))
        or path_obj.exists()
    )

    if looks_like_file:
        p = path_obj.resolve()
        if not p.exists():
            raise FileNotFoundError(f"Strategy file not found: {p}")
        if p.suffix != ".py":
            raise ValueError(f"Expected a .py file, got: {p}")

        spec = importlib.util.spec_from_file_location(p.stem, str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create import spec for '{p}'.")
        mod = importlib.util.module_from_spec(spec)

        unique_name = f"{p.stem}_{abs(hash(str(p))) & 0xFFFF:x}"
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)
    else:
        # Treat as module import path
        try:
            mod = importlib.import_module(spath)
        except ModuleNotFoundError as e:
            raise ImportError(f"Could not import module '{spath}'.") from e

    try:
        StrategyClass = getattr(mod, strategy_class)
    except AttributeError as e:
        where = str(path_obj) if looks_like_file else spath
        raise AttributeError(f"Class '{strategy_class}' not found in '{where}'.") from e

    return StrategyClass
