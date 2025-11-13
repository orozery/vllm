from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='mycuda',
    ext_modules=[
        CUDAExtension(
            name='mycuda',
            sources=['torch_bindings.cpp','cache_kernels.cu'],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3', '-Xptxas', '-O3', '--use_fast_math']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)