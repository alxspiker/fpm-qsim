#!/bin/bash
# build.sh — Build the fpm_cpp Python extension module
# =================================================================
#
# Usage:
#   ./build.sh
#
# Outputs:
#   fpm_cpp.cpython-<ver>-<arch>-linux-gnu.so  (the importable extension)
#
# Requirements:
#   - Python 3.9+ with pybind11 installed (pip install pybind11)
#   - g++ 11+ (or clang++ 11+) with OpenMP support
#   - Linux or macOS (for Windows, use MSYS2/MinGW or WSL)
#
set -euo pipefail

# Resolve script directory (so the script works from any CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Python
PYTHON="${PYTHON:-python3}"
echo "Using Python: $($PYTHON --version)"

# Get Python include path
PY_INC="$($PYTHON -c 'import sysconfig; print(sysconfig.get_path("include"))')"
echo "Python include: $PY_INC"

# Get pybind11 include path
PYBIND_INC="$($PYTHON -c 'import pybind11; print(pybind11.get_include())')"
echo "pybind11 include: $PYBIND_INC"

# Get extension suffix (e.g., .cpython-312-x86_64-linux-gnu.so)
EXT_SUFFIX="$($PYTHON -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')"
echo "Extension suffix: $EXT_SUFFIX"

# Detect compiler (prefer g++, fall back to clang++)
CC="${CXX:-}"
if [[ -z "$CC" ]]; then
    if command -v g++ &>/dev/null; then
        CC="g++"
    elif command -v clang++ &>/dev/null; then
        CC="clang++"
    else
        echo "ERROR: Neither g++ nor clang++ found in PATH." >&2
        exit 1
    fi
fi
echo "Using compiler: $CC"

# Detect CPU architecture for -march=native
ARCH_FLAGS=""
if [[ "$($CC -dumpmachine 2>/dev/null)" == *x86_64* ]]; then
    ARCH_FLAGS="-march=native"
elif [[ "$($CC -dumpmachine 2>/dev/null)" == *aarch64* ]] || [[ "$($CC -dumpmachine 2>/dev/null)" == *arm64* ]]; then
    ARCH_FLAGS="-march=native"
fi
echo "Architecture flags: $ARCH_FLAGS"

# Compile
echo ""
echo "Compiling fpm_cpp bindings..."
"$CC" -O3 $ARCH_FLAGS -ffast-math -fopenmp \
    -shared -std=c++17 -fPIC \
    -I"$PY_INC" -I"$PYBIND_INC" \
    fpm_cpp_bindings.cpp \
    -o "fpm_cpp${EXT_SUFFIX}" \
    -fopenmp

# Verify
echo ""
echo "Build successful:"
ls -la "fpm_cpp${EXT_SUFFIX}"

# Smoke test
echo ""
echo "Smoke test:"
$PYTHON -c "
import sys; sys.path.insert(0, '.')
import fpm_cpp
print(f'  version: {fpm_cpp.__version__}')
print(f'  build:   {fpm_cpp.build_info}')
print(f'  GAMMA_MAX = {fpm_cpp.GAMMA_MAX}')
print('  OK')
" || {
    echo "ERROR: Smoke test failed!" >&2
    exit 1
}

echo ""
echo "Done. The module is now importable from Python:"
echo "  import sys; sys.path.insert(0, '$SCRIPT_DIR')"
echo "  import fpm_cpp"
