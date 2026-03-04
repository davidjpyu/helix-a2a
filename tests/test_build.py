"""Smoke test: verify the package imports and workspace_size works."""
import sys


def test_import():
    import helix_a2a
    assert hasattr(helix_a2a, "workspace_size")
    assert hasattr(helix_a2a, "allocate_workspace")
    assert hasattr(helix_a2a, "init_workspace")
    assert hasattr(helix_a2a, "alltoall")


def test_workspace_size():
    import helix_a2a
    for cp_size in [2, 4, 8]:
        ws = helix_a2a.workspace_size(cp_size)
        assert isinstance(ws, int), f"Expected int, got {type(ws)}"
        assert ws > 0, f"workspace_size({cp_size}) = {ws}, expected > 0"
        print(f"  workspace_size({cp_size}) = {ws} bytes "
              f"({ws / (1024 * 1024):.1f} MiB)")


def test_version():
    import helix_a2a
    assert hasattr(helix_a2a, "__version__")
    assert helix_a2a.__version__ == "0.1.0"


if __name__ == "__main__":
    print("Running helix_a2a build smoke tests...")
    test_import()
    print("[PASS] import helix_a2a")
    test_workspace_size()
    print("[PASS] workspace_size returns positive integers")
    test_version()
    print("[PASS] version check")
    print("\nAll smoke tests passed!")
    sys.exit(0)
