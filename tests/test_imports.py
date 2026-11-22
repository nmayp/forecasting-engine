"""Every module of the package imports (heavy model libraries are imported lazily)."""
import importlib
import pkgutil

import forecasting_engine


def test_all_modules_import():
    bad = []
    for m in pkgutil.walk_packages(forecasting_engine.__path__, "forecasting_engine."):
        try:
            importlib.import_module(m.name)
        except Exception as e:  # noqa: BLE001
            bad.append((m.name, repr(e)[:120]))
    assert not bad, bad
