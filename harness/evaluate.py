import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness.config import load_settings


SPEC_PATH = Path("evaluation/spec.json")
MANIFEST_PATH = Path("evaluation/runs.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_manifest(manifest):
    temporary = MANIFEST_PATH.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(MANIFEST_PATH)


def main():
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    settings = load_settings()

    # Exclude credentials from evaluation metadata.
    configuration = settings.model_dump(
        mode="json",
        exclude={"groq_api_key"},
    )

    input_hashes = {}
    for task in spec["tasks"]:
        for path in sorted(Path(task["input"]).rglob("*")):
            if path.is_file():
                input_hashes[path.as_posix()] = digest(path)

    # Record which implementation produced the measured results.
    code_hashes = {
        path.as_posix(): digest(path)
        for path in sorted(Path("harness").glob("*.py"))
    }

    fingerprint = {
        "spec_sha256": digest(SPEC_PATH),
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "configuration": configuration,
    }

    if MANIFEST_PATH.exists():
        manifest = json.loads(
            MANIFEST_PATH.read_text(encoding="utf-8")
        )
        if manifest["fingerprint"] != fingerprint:
            raise SystemExit(
                "Evaluation inputs, code, or configuration changed. "
                "Keep the existing manifest; do not mix different "
                "versions into the same evaluation."
            )
    else:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "fingerprint": fingerprint,
            "runs": [],
        }
        save_manifest(manifest)

    completed = {
        (record["task_id"], record["mode"])
        for record in manifest["runs"]
    }

    runs_root = Path("runs")
    runs_root.mkdir(exist_ok=True)

    for index, task in enumerate(spec["tasks"]):
        # Alternate which architecture runs first to reduce a consistent
        # ordering advantage. This does not eliminate rate-limit effects.
        modes = (
            ["single_agent", "multi_agent"]
            if index % 2 == 0
            else ["multi_agent", "single_agent"]
        )

        for mode in modes:
            key = (task["id"], mode)
            if key in completed:
                print(f"Already recorded: {task['id']} / {mode}")
                continue

            module = (
                "harness.baseline"
                if mode == "single_agent"
                else "harness.orchestrator"
            )

            print(
                f"\n=== Evaluation: {task['id']} / {mode} ===",
                flush=True,
            )

            before = {
                path.name
                for path in runs_root.iterdir()
                if path.is_dir()
            }

            # Inherit the terminal so the existing approval prompt works.
            process = subprocess.run([
                sys.executable,
                "-m",
                module,
                "--input",
                task["input"],
                "--goal",
                task["goal"],
            ])

            after = {
                path.name
                for path in runs_root.iterdir()
                if path.is_dir()
            }

            created = sorted(after - before)
            if len(created) != 1:
                raise SystemExit(
                    "No unique run directory was created. Evaluation "
                    "stopped; existing recorded runs are preserved."
                )

            run_id = created[0]
            state_path = runs_root / run_id / "state.json"

            if state_path.exists():
                state = json.loads(
                    state_path.read_text(encoding="utf-8")
                )
                if state.get("goal") != task["goal"]:
                    raise SystemExit(
                        "Run goal does not match the evaluation task."
                    )
                status = state["status"]
            else:
                status = "crashed"

            manifest["runs"].append({
                "task_id": task["id"],
                "mode": mode,
                "run_id": run_id,
                "status": status,
                "process_exit_code": process.returncode,
                "recorded_at": datetime.now(timezone.utc).isoformat(),
            })
            save_manifest(manifest)
            completed.add(key)

            print(
                f"Recorded: {task['id']} / {mode} "
                f"-> {status} ({run_id})",
                flush=True,
            )

            # Failures and partial runs count as observed results.
            # Do not silently rerun them until a successful result appears.
            if status == "crashed":
                raise SystemExit(
                    "Run crashed before saving state. Recorded the crash "
                    "and stopped for diagnosis."
                )

    print("\nAll six evaluation runs recorded.")
    print("Manifest:", MANIFEST_PATH)
    print("Quality scores have not been assigned yet.")


if __name__ == "__main__":
    main()

