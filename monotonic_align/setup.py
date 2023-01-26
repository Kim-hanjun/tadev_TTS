from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os
import shutil

setup(ext_modules=cythonize("monotonic_align_search.pyx"), include_dirs=[numpy.get_include()])

found_full_path = None
found_file = None

"""
컴파일된 파일은 무조건 build 하위에 생성됩니다.
build 하위에 모든 경로를 탐색하며 .so파일을 찾습니다.
.so파일은 1개의 c파일을 컴파일하면 보통 1개만 생성됩니다. (현재의 경우도 그러함)
"""
for (path, dir, files) in os.walk("./build/"):
    for filename in files:
        ext = os.path.splitext(filename)[-1]
        if ext == ".so":
            found_full_path = os.path.join(path, filename)
            found_file = filename

# 찾은 경로를 이용하여 setup.py가 있는 root 디렉터리로 copy합니다. (사실 mv 시켜도 상관없음.)
shutil.copyfile(found_full_path, found_file)
