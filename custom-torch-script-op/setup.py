from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='warp_perspective',
    packages=['warp_perspective'],
    ext_modules=ext_modules = [
        CppExtension(
            'warp_perspective', ['op.cpp'],
            extra_compile_args=['-g']),
    ],
    cmdclass={'build_ext': BuildExtension})
