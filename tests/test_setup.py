"""
Sanity check tests for Phase 1 project setup.
"""
from pathlib import Path
import pytest


def test_required_directories_exist():
    """Verify that all foundational project directories are present."""
    base_dir = Path(__file__).resolve().parent.parent
    expected_dirs = [
        "data/raw",
        "data/processed",
        "data/sample",
        "src/data",
        "src/intents",
        "src/retrieval",
        "src/generation",
        "src/escalation",
        "src/agent",
        "src/utils",
        "baselines",
        "evaluation",
        "golden_set",
        "configs",
        "tests",
        "notebooks",
        "results",
        "reports",
    ]
    for rel_dir in expected_dirs:
        dir_path = base_dir / rel_dir
        assert dir_path.is_dir(), f"Expected directory missing: {rel_dir}"


def test_required_files_exist():
    """Verify that all Phase 1 documentation and config files exist."""
    base_dir = Path(__file__).resolve().parent.parent
    expected_files = [
        "README.md",
        "requirements.txt",
        ".env.example",
        ".gitignore",
        "decision_log.md",
    ]
    for rel_file in expected_files:
        file_path = base_dir / rel_file
        assert file_path.is_file(), f"Expected file missing: {rel_file}"
        assert file_path.stat().st_size > 0, f"File is empty: {rel_file}"


def test_package_imports():
    """Verify that src and its subpackages can be imported."""
    import src
    import src.data
    import src.intents
    import src.retrieval
    import src.generation
    import src.escalation
    import src.agent
    import src.utils
    import baselines
    import evaluation

    assert src.__version__ == "0.1.0"
