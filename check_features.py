try:
    from scripts.check_features import main
except ModuleNotFoundError as exc:
    missing = {"joblib", "pandas", "sklearn"}
    if exc.name in missing:
        print(
            "Missing Python dependencies. Activate your virtual environment first: "
            ".\\venv\\Scripts\\Activate.ps1"
        )
        raise SystemExit(1) from exc
    raise


if __name__ == "__main__":
    main()
