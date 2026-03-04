import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

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
    packages=["helix_a2a"],
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
