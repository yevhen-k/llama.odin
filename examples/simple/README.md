# Simple example

1. Clone llama.cpp repo
2. Apply Makefile.patch

```bash
cd ../../llama.cpp/

git apply ../examples/simple/Makefile.patch
```

Patch adds the following important lines:

```makefile
binding.o: binding.cpp binding.h \
	$(OBJ_ALL)
	$(CXX) -c $(CXXFLAGS) $(LDFLAGS) $<

binding.a: binding.o
	ar rcs $@ $< $(OBJ_ALL)

libbinding.so: binding.o
	gcc -shared -o $@ $< $(OBJ_ALL)
```

3. Copy `binding.h` and `binding.cpp` files to the `llama.cpp` folder

```bash
cp ../examples/simple/binding.* .
```

4. Build bindings with acceleration you need

```bash
make GGML_CUDA=1 GGML_USE_CUDA=1 GGML_USE_BLAS=1 libbinding.so -j 8
# OR
make GGML_CUDA=1 GGML_USE_CUDA=1 GGML_USE_BLAS=1 binding.a -j 8
```

5. Build and run Odin example

```bash
cd ../examples/simple
odin run . -out:simple -extra-linker-flags:"-v -L/opt/cuda/lib -lstdc++ -lcudart -lcuda -lcublas -lgomp" -- -m ../../models/gemma-2b.Q2_K.gguf --prompt "Once upon a time" --predict 128 --gpu-layers 10

ldd simple
```

6. Clear build artifacts

```bash
cd ../../llama.cpp/
make clean
rm -f binding.o binding.a libbinding.so binding.h binding.cpp
git checkout -- Makefile
```