-include .env

REMOTE_DIR        ?= $(shell basename $(CURDIR))
EXTRA             ?= nvidia
CUDA_VERSION      ?= 12.8
GCC_VERSION       ?= 13
LLAMA_CPP_VERSION ?= 0.3.26

define remote-exec
ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && $(1)'"
endef

.PHONY: sync ssh install install-llama-cpp setup run serve-vllm serve-ollama guard-%

guard-%:
	$(if $(value $*),,$(error $* is not set. Define it in .env or pass $*=...))

sync: guard-HOST
	rsync -avz --exclude='.git' --filter=':- .gitignore' ./ $(HOST):$(REMOTE_DIR)/

fetch: 
	rsync -avz $(HOST):$(REMOTE_DIR)/mlflow.db $(HOST):$(REMOTE_DIR)/mlruns  ./

ssh: guard-HOST
	ssh $(HOST)

install: guard-HOST
	$(call remote-exec,uv sync)

setup: install install-llama-cpp

run: guard-HOST guard-CMD
	$(call remote-exec,$(CMD))

serve-vllm: guard-HOST guard-MODEL
	$(call remote-exec,uv run vllm serve $(MODEL) --host 0.0.0.0 --port 8000)

serve-ollama: guard-HOST guard-MODEL
	$(call remote-exec,ollama pull $(MODEL) && ollama serve)

install-llama-cpp: guard-HOST
	ssh $(HOST) "bash -l -c 'cd $(REMOTE_DIR) && \
		CUDA_HOME=/usr/local/cuda-$(CUDA_VERSION) \
		PATH=/usr/local/cuda-$(CUDA_VERSION)/bin:\$$HOME/.local/bin:\$$PATH \
		CMAKE_ARGS=\"-DGGML_CUDA=on -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-$(GCC_VERSION)\" \
		CC=/usr/bin/gcc-$(GCC_VERSION) CXX=/usr/bin/g++-$(GCC_VERSION) \
		uv pip install \"llama-cpp-python==$(LLAMA_CPP_VERSION)\" --no-binary llama-cpp-python'"
