# Odin bindings to the [Llama.cpp](https://github.com/ggerganov/llama.cpp)

At the moment functionality in the [binding.odin](./binding.odin) is enough to run [simple example](./examples/simple/) based on [llama.cpp/examples/simple](https://github.com/ggerganov/llama.cpp/tree/b3735/examples/simple) example.

## Quick start

1. Pull [Llama.cpp](https://github.com/ggerganov/llama.cpp) and checkout to the **b3735** tag:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout b3735
```

2. [Optional] Build libraries and binaries. This allows you to test build with chosen build flags.

```bash
# assume you're in the llama.cpp dir
mkdir build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON ..
cmake --build . --config Release --parallel 8
```

3. Download model you want into the [models](./models/) folder:

```bash
# assume you're in the project root dir
make get-gemma2b
```

4. [Optional] Test downloaded model with `llama-cli` you've built on the step (2)

```bash
./llama.cpp/build/bin/llama-cli --model models/gemma-2b.Q2_K.gguf --prompt "Once upon a time" --predict 128
```

5. Build and run simple example

```bash
# assume you're in the project root dir
make example-simple
```

This will apply [Makefile.patch](Makefile.patch) to the [llama.cpp/Makefile](./llama.cpp/Makefile), copy [binding.cpp](binding.cpp) and [binding.h](binding.h) to the `llama.cpp/` folder, and build `binding.a` file.

Then, [simple project](./examples/simple/) will be compiled and linked with `binding.a`.

## Mini Demo

<video width="800" controls>
    <source src="./assets/demo.mp4" type="video/mp4">
</video>

## Note

1. This is a "thin" wrapper, no custom functionality is added.
2. All command line arguments are directly passed to the llama.cpp for parsing.
3. Opaque pointers has `_ptr` postfix.
4. Wrapped functions has `lpp_` prefix (short of llama.cpp). Those are usually form [common.h](https://github.com/ggerganov/llama.cpp/blob/b3735/common/common.h).
5. Functions without `lpp_` prefix are from [llama.h](https://github.com/ggerganov/llama.cpp/blob/b3735/include/llama.h)


## Where to get documentation for the functionality?

Here: https://github.com/ggerganov/llama.cpp/tree/b3735/

## What next?

To extend current functionality, please refere to the following:
- code
  - https://github.com/ggerganov/llama.cpp/tree/b3735/common
  - https://github.com/ggerganov/llama.cpp/blob/b3735/include/llama.h
- examples
  - https://github.com/ggerganov/llama.cpp/tree/b3735/examples