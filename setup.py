import os

from setuptools import setup

import torch.utils.cpp_extension as _cpp_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# NGC images often ship a CUDA toolkit newer than what PyTorch was compiled
# against (e.g. nvcc 13.1 with torch.version.cuda == "12.8").  The strict
# major-version check in torch's BuildExtension rejects this, but compiling
# SM 90/100 kernels with a newer toolkit against an older PyTorch is safe.
_cpp_ext._check_cuda_version = lambda *a, **kw: None

extra_nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_90,code=sm_90",
    "-gencode=arch=compute_90a,code=sm_90a",
    "-gencode=arch=compute_100,code=sm_100",
]

extra_cxx_flags = ["-O3"]

setup(
    name="helix_a2a",
    version="0.1.0",
    description="Standalone Helix All-to-All CUDA extension",
    packages=["helix_a2a"],
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
    ext_modules=[
        CUDAExtension(
            name="helix_a2a._C",
            sources=[
                "csrc/torch_binding.cpp",
                "csrc/helix_alltoall.cu",
            ],
            include_dirs=[os.path.join(os.path.dirname(__file__), "csrc")],
            extra_compile_args={
                "cxx": extra_cxx_flags,
                "nvcc": extra_nvcc_flags,
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
