# Coursera-ML-AndrewNg-Torch

This project is a reimplementation of the exercises from the coursera-machine-learning course using torch7 + Lua. It serves as a practice project for using torch7 and welcomes discussions and learning.

## Dependencies

The project relies on some external libraries. If you encounter any issues during runtime, it's likely due to missing these libraries. Here is an introduction to the dependencies:

1. **matio:**
   - Download and install the matio shared library from [https://sourceforge.net/projects/matio/files/matio/](https://sourceforge.net/projects/matio/files/matio/). Extract it, then run `make; make check; make install; ldconfig` for installation.
   - Install the matio Lua interface library with `luarocks install matio` or visit [https://github.com/soumith/matio-ffi.torch](https://github.com/soumith/matio-ffi.torch) for manual installation.

2. **svm library:**
   - Refer to [https://github.com/koraykv/torch-svm](https://github.com/koraykv/torch-svm) for installation.

3. **lpeg library:**
   - Run `luarocks install lpeg` or download and install from [http://www.inf.puc-rio.br/~roberto/lpeg/lpeg-1.1.0.tar.gz](http://www.inf.puc-rio.br/~roberto/lpeg/lpeg-1.1.0.tar.gz).

## Running

Navigate to a specific exercise directory and use the `th` command to run. For example, `cd ex1-linear_regression; th ex1.lua`. Note: Most exercises have two parts, for instance, in ex1, there are ex1.lua and ex1_multi.lua, you can run them separately.
