.PHONY: llama.cpp get-gemma2b get-qwen2 llama-cli-gemma2b llama-cli-qwen2 example-simple

# ------------------ llama.cpp build ------------------ #

LLAMA_TAG = b3735

# https://github.com/phronmophobic/llama.clj
llama.cpp:
	git clone https://github.com/ggerganov/llama.cpp
	cd llama.cpp && \
		git checkout $(LLAMA_TAG) && \
		mkdir build &&\
		cd build && \
		cmake -DBUILD_SHARED_LIBS=ON -DGGML_CUDA=ON .. && \
		cmake --build . --config Release --parallel 8


# ------------------ download models ------------------ #

get-gemma2b:
	if test -e models/gemma-2b.Q2_K.gguf; \
	then echo "model downloaded..."; \
	else wget https://huggingface.co/MaziyarPanahi/gemma-2b-GGUF/resolve/main/gemma-2b.Q2_K.gguf?download=true -O models/gemma-2b.Q2_K.gguf ; \
	fi

get-qwen2:
	if test -e models/qwen2-0_5b-instruct-fp16.gguf; \
	then echo "model downloaded..."; \
	else wget https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-fp16.gguf?download=true -O models/qwen2-0_5b-instruct-fp16.gguf ; \
	fi

# ------------------ llama.cpp test ------------------ #

llama-cli-gemma2b: get-gemma2b
	./llama.cpp/build/bin/llama-cli --model models/gemma-2b.Q2_K.gguf --prompt "Once upon a time" --predict 128

llama-cli-qwen2: get-qwen2
	./llama.cpp/build/bin/llama-cli --model models/qwen2-0_5b-instruct-fp16.gguf --prompt 'Classify the following statement by sentiment as positive or negative: "I positive Odin programming language". Use JSON format for answer, for example {"sentiment": "positive"} or {"sentiment": "negative"}.  Do not think about the answer just give it out outright.' --predict 20

# ------------------ Odin bindings examples ------------------ #

example-simple:
	cd llama.cpp && git apply ../examples/simple/Makefile.patch && \
		cp ../examples/simple/binding.* . && \
		make GGML_CUDA=1 GGML_USE_CUDA=1 GGML_USE_BLAS=1 binding.a -j 8
	cd examples/simple && \
		odin run . -out:simple -extra-linker-flags:"-v -L/opt/cuda/lib -lstdc++ -lcudart -lcuda -lcublas -lgomp" -- -m ../../models/gemma-2b.Q2_K.gguf --prompt "Once upon a time" --predict 128 --gpu-layers 10
	cd llama.cpp && git checkout -- Makefile && rm -f binding.o binding.a libbinding.so binding.h binding.cpp