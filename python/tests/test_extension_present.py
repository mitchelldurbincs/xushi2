"""Sanity check: the compiled C++ extension must be present.

This test fails hard (never skips) so that CI and Hermes workers immediately
notice when the repository was checked out or cloned without building the
native extension first.
"""

from __future__ import annotations


def test_xushi2_cpp_extension_is_importable():
    """xushi2_cpp must be importable and expose core symbols."""
    try:
        from xushi2 import xushi2_cpp as _cpp
    except ImportError as exc:
        raise ImportError(
            "xushi2_cpp extension not found. Build the C++ side with CMake first:\n"
            "    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release\n"
            "    cmake --build build -j\n"
            "(from the repository root, not the python/ subdir)."
        ) from exc

    # Verify the module is not just an empty stub.
    assert hasattr(_cpp, "Sim"), "xushi2_cpp imported but missing Sim"
    assert hasattr(_cpp, "TICK_HZ"), "xushi2_cpp imported but missing TICK_HZ"
