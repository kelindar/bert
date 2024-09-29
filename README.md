# Go Wrapper for BERT C++ Library

This repository contains a Go wrapper for the [bert.cpp](https://github.com/skeskinen/bert.cpp) library, or an updated fork [embeddings.cpp](https://github.com/xyzhang626/embeddings.cpp). The library is semi-deprecated, since the functionality has been integrated in [llama.cpp](https://github.com/ggerganov/llama.cpp). However, `bert.cpp` is much smaller and easier to use, so it might be useful for some use cases where one does not need the full functionality of LLM and just wants to use BERT embeddings, for example when using to generate an embedding for a query in a web server or local lookups.

## Precompiled binaries

Precompiled binaries for Windows and Linux are available in the [dist](dist) directory. If the architecture/platform you are using is not available, you would need to compile the library yourself. I've additionally provided also a couple of models in the same directory. The model in question is [MiniLM](https://arxiv.org/abs/2002.10957) with 6 and 12 layers.

## Compile library

First, clone the repository and its submodules with the following commands. The `--recurse-submodules` flag is used to clone the `ggml` submodule, which is a header-only library for matrix operations.

```bash
git clone --recurse-submodules https://github.com/skeskinen/bert.cpp
cd bert.cpp
```

### Compile on Linux

Make sure you have a C/C++ compiler and CMake installed. For Ubuntu, you can install them with the following commands:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake
```

Then you can compile the library with the following commands:

```bash
cd bert.cpp
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++ -DCMAKE_C_COMPILER=gcc
make all
```

This should egnerate `libbert.so` that you can use.

### Compile on Windows

Make sure you have a C/C++ compiler and CMake installed. For Windows, a simple option is to use [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/) (make sure CLI tools are included) and [CMake](https://cmake.org/download/).

```bash
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=cl -DCMAKE_C_COMPILER=cl
```

If you are using Visual Studio, solution files are generated. You can open the solution file with Visual Studio and build the project from there. The `bin` directory would then contain `bert.dll` and `ggml.dll`.
