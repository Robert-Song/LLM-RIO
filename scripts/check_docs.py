"""Check maintained documentation, configuration examples, and CLI help without GPUs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    documents += [
        path for path in (ROOT / "docs").glob("*.md") if path.name != "vllm-upgrade-plan.md"
    ]
    documents += list((ROOT / "docs/release").glob("*.md"))
    documents += [ROOT / "docs/historical/README.md"]
    errors = []
    for path in documents:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if link.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target = link.split("#", 1)[0]
            if not (path.parent / target).exists():
                errors.append(f"{path.relative_to(ROOT)}: missing {target}")
    # Start outside the checkout to exclude the operator's ignored beta config/.env.
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLMRIO_")}
    env["PYTHONPATH"] = str(ROOT / "src")
    with tempfile.TemporaryDirectory(prefix="rio-docs-") as workdir:
        configs = [ROOT / "config.example.toml", *(ROOT / "examples/config").glob("*.toml")]
        for config in configs:
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    "from llm_rio.config import Settings; "
                    "import sys; Settings(config_file=sys.argv[1])",
                    str(config),
                ],
                cwd=workdir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
        for args in [
            [],
            ["serve"],
            ["doctor"],
            ["capabilities"],
            ["status"],
            ["models"],
            ["models", "add"],
            ["models", "validate"],
            ["models", "trust-measurements"],
            ["models", "profile-state"],
            ["keys"],
            ["maintenance"],
        ]:
            subprocess.run(
                [sys.executable, "-m", "llm_rio", *args, "--help"],
                cwd=workdir,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
    for script in ("native_acceptance.py", "record_native_acceptance.py"):
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
    import json

    for example in (ROOT / "examples/acceptance").glob("*.json"):
        json.loads(example.read_text())
    if errors:
        raise SystemExit("\n".join(errors))
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/check_acceptance_plan.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(f"Documentation links, {len(configs)} configs and CLI help passed.")


if __name__ == "__main__":
    main()
