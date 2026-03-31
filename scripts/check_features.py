from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_artifact(filename: str) -> Path:
    candidates = [
        BASE_DIR / "models" / filename,
        BASE_DIR / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Missing required artifact: {filename}. Looked in {candidates[0]} and {candidates[1]}"
    )


def main() -> None:
    features_path = _resolve_artifact("model_features.pkl")
    features = joblib.load(features_path)

    print(f"Loaded {len(features)} features from: {features_path}")
    for idx, feature in enumerate(features, start=1):
        print(f"{idx:02d}. {feature}")


if __name__ == "__main__":
    main()
