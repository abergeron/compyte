import os, sys

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext
from distutils.dep_util import newer
import numpy as np

class build_ext_nvcc(build_ext):
    user_options = build_ext.user_options
    user_options.extend([
            ('cuda-root=', None, "The cuda root directory")])
    
    def initialize_options(self):
        build_ext.initialize_options(self)
        self.cuda_root = None

    def finalize_options(self):
        build_ext.finalize_options(self)
        if self.cuda_root is None:
            self.cuda_root = os.getenv('CUDA_ROOT', None)
        if self.cuda_root is not None:
            self._nvcc_bin = os.path.join(self.cuda_root, 'bin', 'nvcc')
        else:
            self._nvcc_bin = 'nvcc'
    def cuda_process(self, source, include_args):
        target = source+'.cpp'
        if newer(source, target):
            self.spawn([self._nvcc_bin, '--cuda', source, '-o', target] + \
                           include_args)
        return target

    def cuda_extension(self, ext):
        includes = self.distribution.include_dirs+ext.include_dirs
        include_args = ['-I'+i for i in includes]
        new_sources = []
        anycuda = False
        for src in ext.sources:
            if src.endswith('.cu'):
                new_sources.append(self.cuda_process(src, include_args))
                anycuda = True
            else:
                new_sources.append(src)
        if anycuda:
            ext.sources = new_sources
            if self.cuda_root is not None:
                lib = os.path.join(self.cuda_root, 'lib')
                lib64 = os.path.join(self.cuda_root, 'lib64')
                if os.path.isdir(lib):
                    ext.library_dirs.append(lib)
                    ext.extra_link_args.append('-Xlinker')
                    ext.extra_link_args.append('-rpath')
                    ext.extra_link_args.append('-Xlinker')                    
                    ext.extra_link_args.append(lib)
                if os.path.isdir(lib64):
                    ext.library_dirs.append(lib64)
#                    ext.extra_link_args.append('-rpath')
#                    ext.extra_link_args.append(lib64)
            if 'cudart' not in ext.libraries:
                ext.libraries.append('cudart')
    
    def build_extensions(self):
        self.check_extensions_list(self.extensions)

        for ext in self.extensions:
            self.cuda_extension(ext)
            # uncomment this + inherit from the cython version of build_ext
            # work with cuda and cython sources
            #ext.sources = self.cython_sources(ext.sources, ext)
            self.build_extension(ext)

setup(name='compyte',
      cmdclass = {'build_ext': build_ext_nvcc},
      include_dirs = [np.get_include(), '.'],
      ext_modules=[Extension('pygpu_ndarray', 
                             sources = ['pygpu_language_cuda.cu',
                                        'pygpu_ndarray.cpp'],
                             libraries = ['cublas'])
                   ]
)
