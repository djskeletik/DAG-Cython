from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
from pathlib import Path

# Абсолютный путь к src
here = Path(__file__).parent
src_path = here / "src" / "library.pyx"

extensions = [
    Extension(
        name="graph_cython.library",
        sources=[str(src_path)],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="graph_cython",
    ext_modules=cythonize(extensions, compiler_directives={'language_level': "3"}),
)
