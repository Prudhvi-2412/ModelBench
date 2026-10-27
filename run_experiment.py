import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a saved Model Bench experiment config")
    parser.add_argument("--config", required=True, help="Path to a config under configs/experiments")
    parser.add_argument("--api-url", default=os.environ.get("MODELBENCH_API_URL", "http://127.0.0.1:8000"))
    args = parser.parse_args()
    config_path = Path(args.config)
    name = config_path.stem
    request = Request(
        f"{args.api_url.rstrip('/')}/api/v1/experiments/run",
        data=json.dumps({"config": name}).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **(
                {"Authorization": f"Bearer {os.environ['MODELBENCH_API_KEY']}"}
                if os.environ.get("MODELBENCH_API_KEY")
                else {}
            ),
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=3600) as response:
            print(json.dumps(json.loads(response.read()), indent=2))
    except HTTPError as exc:
        print(exc.read().decode("utf-8", errors="replace"))
        return 1
    except URLError as exc:
        print(f"Could not reach Model Bench API: {exc.reason}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
