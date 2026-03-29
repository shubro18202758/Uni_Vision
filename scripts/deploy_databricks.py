#!/usr/bin/env python3
"""
Deploy Uni Vision notebooks to a Databricks workspace.

Usage:
  1. Configure the CLI:
       databricks configure --token
       (Enter your workspace URL and personal access token)

  2. Run this script:
       python scripts/deploy_databricks.py

  3. Or specify a custom workspace path:
       python scripts/deploy_databricks.py --workspace-path /Users/you@email.com/Uni_Vision
"""

import argparse
import subprocess
import sys
import os

NOTEBOOK_DIR = os.path.join(os.path.dirname(__file__), "..", "databricks_notebooks")
DEFAULT_WORKSPACE_PATH = "/Uni_Vision"

NOTEBOOKS = [
    "00_setup",
    "01_delta_lake_demo",
    "02_mlflow_tracking_demo",
    "03_spark_analytics_demo",
    "04_vector_search_demo",
    "05_full_pipeline_demo",
]


def run_cmd(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"  ❌ FAILED: {result.stderr.strip()}")
        return result
    return result


def check_cli() -> bool:
    """Check if Databricks CLI is installed and configured."""
    result = run_cmd(["databricks", "--version"], check=False)
    if result.returncode != 0:
        print("❌ Databricks CLI not found.")
        print("   Install:  pip install databricks-cli")
        print("   Configure: databricks configure --token")
        return False
    print(f"  ✅ {result.stdout.strip()}")

    # Check if configured
    result = run_cmd(["databricks", "workspace", "ls", "/"], check=False)
    if result.returncode != 0:
        print("❌ Databricks CLI not configured.")
        print("   Run: databricks configure --token")
        print("   You'll need your workspace URL and personal access token.")
        return False
    print("  ✅ CLI authenticated\n")
    return True


def create_workspace_dir(workspace_path: str) -> bool:
    """Create the workspace directory if it doesn't exist."""
    print(f"📁 Creating workspace directory: {workspace_path}")
    result = run_cmd(["databricks", "workspace", "mkdirs", workspace_path], check=False)
    if result.returncode != 0 and "already exists" not in result.stderr.lower():
        print(f"  ⚠️  Could not create directory (may already exist)")
    else:
        print(f"  ✅ Directory ready")
    return True


def deploy_notebooks(workspace_path: str) -> int:
    """Deploy all notebooks to the workspace."""
    success = 0
    failed = 0

    for nb_name in NOTEBOOKS:
        local_path = os.path.join(NOTEBOOK_DIR, f"{nb_name}.py")
        remote_path = f"{workspace_path}/{nb_name}"

        if not os.path.exists(local_path):
            print(f"  ⚠️  {nb_name}.py not found locally, skipping")
            failed += 1
            continue

        print(f"\n📤 Uploading: {nb_name}")
        result = run_cmd([
            "databricks", "workspace", "import",
            local_path, remote_path,
            "--language", "PYTHON",
            "--format", "SOURCE",
            "--overwrite",
        ], check=False)

        if result.returncode == 0:
            print(f"  ✅ {remote_path}")
            success += 1
        else:
            print(f"  ❌ Failed: {result.stderr.strip()}")
            failed += 1

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Deploy Uni Vision to Databricks")
    parser.add_argument(
        "--workspace-path", "-p",
        default=DEFAULT_WORKSPACE_PATH,
        help=f"Workspace path for notebooks (default: {DEFAULT_WORKSPACE_PATH})",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  🚀 Uni Vision — Databricks Deployment")
    print("=" * 60)

    # Step 1: Check CLI
    print("\n🔍 Checking Databricks CLI...")
    if not check_cli():
        sys.exit(1)

    # Step 2: Create workspace directory
    create_workspace_dir(args.workspace_path)

    # Step 3: Deploy notebooks
    print(f"\n📦 Deploying {len(NOTEBOOKS)} notebooks to {args.workspace_path}...")
    success, failed = deploy_notebooks(args.workspace_path)

    # Summary
    print("\n" + "=" * 60)
    print(f"  📊 Deployment Summary")
    print(f"     Uploaded:  {success}/{len(NOTEBOOKS)}")
    if failed:
        print(f"     Failed:    {failed}")
    print(f"     Location:  {args.workspace_path}")
    print("=" * 60)

    if success == len(NOTEBOOKS):
        print(f"""
  ✅ All notebooks deployed successfully!

  Open your Databricks workspace and navigate to:
    {args.workspace_path}/

  Run notebooks in this order:
    1. 00_setup            — Install dependencies
    2. 01_delta_lake_demo  — Delta Lake ACID storage
    3. 02_mlflow_tracking  — MLflow inference tracking
    4. 03_spark_analytics  — PySpark batch analytics
    5. 04_vector_search    — FAISS similarity search
    6. 05_full_pipeline    — End-to-end orchestration
""")
    else:
        print("\n  ⚠️  Some notebooks failed. Check errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
