from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext_modules = [
    Pybind11Extension(
        "boucwen_cpp",
        ["boucwen_cpp.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    name="boucwen_cpp",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
