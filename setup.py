import os
from pathlib import Path
from setuptools import find_packages

from skbuild import setup
import torch

TORCH_ROOT = torch.utils.cmake_prefix_path
TORCH_VERSION = [int(x) for x in torch.__version__.split(".")[:2]]
assert TORCH_VERSION >= [1, 8], "Requires PyTorch >= 1.8"

def get_version() -> str:
    init_py_path = Path(__file__).parent / "cuda_playground" / "__init__.py"
    with open(init_py_path) as f:
        init_py = f.readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][-1]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    suffix = os.getenv("VBT_VERSION_SUFFIX", "")
    version = version + suffix
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="cuda-playground",
    version=get_version(),
    author="Ming Yang (ymviv@qq.com)",
    url="https://github.com/vivym/cuda-playground",
    description="CUDA Playground",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
    ],
    zip_safe=False,
    cmake_args=[f"-DCMAKE_PREFIX_PATH={TORCH_ROOT}"]
)
