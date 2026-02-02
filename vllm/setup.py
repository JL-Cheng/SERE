# SPDX-License-Identifier: Apache-2.0

from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import torch

def detect_cxx_standard():
    """Detect PyTorch C++ standard requirement."""
    # PyTorch 2.0+ requires C++17, but let's check what PyTorch actually expects
    try:
        # Try to get the actual CXX standard from PyTorch
        version = torch.__version__
        major, minor = map(int, version.split('.')[:2])
        
        # Modern PyTorch versions require C++17
        if major >= 2:
            return 'c++17'
        elif major == 1 and minor >= 12:
            return 'c++17'
        else:
            return 'c++17'  # Default to c++17 for safety
    except:
        return 'c++17'  # Default to c++17 if detection fails

# Detect C++ standard
cxx_std = detect_cxx_standard()
print(f"Detected PyTorch C++ standard: {cxx_std}")

# Build CUDA extensions only if CUDA is available
ext_modules = []
if torch.cuda.is_available():
    try:
        # SERE CUDA extension
        SERE_ext = CUDAExtension(
            name='SERE_vllm.rerouting_cuda_ops.rerouting_ops',
            sources=[
                'SERE_vllm/rerouting_cuda_ops/rerouting_ops.cpp',
                'SERE_vllm/rerouting_cuda_ops/rerouting_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3', f'-std={cxx_std}'],
                'nvcc': ['-O3', '--use_fast_math', f'-std={cxx_std}']
            },
            extra_link_args=['-Wl,-rpath,$ORIGIN']
        )
        
        ext_modules = [SERE_ext]
        print("Building with CUDA operation: SERE_ext")
    except Exception as e:
        print(f"Warning: Failed to setup CUDA extensions: {e}")
        print("Building without CUDA support")
else:
    print("CUDA not available, skipping CUDA extensions")

setup(name='SERE_vllm',
      version='0.1',
      packages=find_packages(),
      ext_modules=ext_modules,
      cmdclass={'build_ext': BuildExtension} if ext_modules else {},
      package_data={
          # Compiled CUDA operations
          'SERE_vllm.rerouting_cuda_ops': ['*.so', '*.py'],
      },
      include_package_data=True,
      entry_points={
          'vllm.general_plugins':
          ["register_SERE_vllm = SERE_vllm:register"]
      })