# setup.py
from setuptools import setup, find_packages, Extension
import numpy

ext_args = {
    'include_dirs': [numpy.get_include()],
    'define_macros': [
        ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
        ('inline', '__inline'),       # ← inline을 __inline으로 바꿔줌
    ],
    # MSVC에서 C11 기능을 쓰려면 시도해볼 수 있습니다 (VS2022 이상)
    'extra_compile_args': ['/std:c11'],
}

setup(
    name="imusim",
    version="0.2",
    author="Alex Young and Martin Ling",
    license="GPLv3",
    url="http://www.imusim.org/",
    install_requires=["simpy>=2.3,<3", "pyparsing", "numpy"],
    packages=find_packages(),
    ext_modules=[
        Extension("imusim.maths.quaternions",
                  ["imusim/maths/quaternions.c"],
                  **ext_args),
        Extension("imusim.maths.quat_splines",
                  ["imusim/maths/quat_splines.c"],
                  **ext_args),
        Extension("imusim.maths.vectors",
                  ["imusim/maths/vectors.c"],
                  **ext_args),
        Extension("imusim.maths.natural_neighbour",
                  [
                    "imusim/maths/natural_neighbour/utils.c",
                    "imusim/maths/natural_neighbour/delaunay.c",
                    "imusim/maths/natural_neighbour/natural.c",
                    "imusim/maths/natural_neighbour.c",
                  ],
                  **ext_args),
    ],
)