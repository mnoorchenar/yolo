# Model discovery & caching
from pathlib import Path


def get_best_model_path() -> str | None:
    """Return path of the most recently trained best.pt, or None."""
    runs = Path("runs/detect")
    if not runs.exists():
        runs = Path("runs")
    candidates = sorted(
        runs.rglob("best.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def list_trained_models() -> list[dict]:
    """List all best.pt weights found under runs/."""
    runs   = Path("runs")
    models = []
    for pt in runs.rglob("best.pt"):
        models.append({
            "name":    pt.parent.parent.name,
            "path":    str(pt),
            "size_mb": round(pt.stat().st_size / 1_000_000, 1),
        })
    models.sort(key=lambda m: m["name"])
    return models