from __future__ import annotations

import os
import sys

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


VERSION = "0.2.1"


def compiler_args() -> list[str]:
    if sys.platform == "win32":
        return ["/O2", "/std:c++17"]
    return ["-O3", "-std=c++17"]


def linker_args() -> list[str]:
    return []


ext_modules = [
    Pybind11Extension(
        "fpm_cpp",
        ["src/fpm_cpp_bindings.cpp"],
        include_dirs=["src"],
        cxx_std=17,
        define_macros=[("VERSION_INFO", f'"{VERSION}"')],
        extra_compile_args=compiler_args(),
        extra_link_args=linker_args(),
    )
]


setup(
    version=VERSION,
    packages=[],
    py_modules=[],
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
