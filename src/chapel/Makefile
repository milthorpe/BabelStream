#DEBUG=true
CHPL_FLAGS+=--detailed-errors --savec tmp --print-commands
CHPL_SOURCE=main.chpl Stream.chpl
ifeq ($(CHPL_LOCALE_MODEL),gpu)
# GPU version
ifndef CHPL_GPU
$(error CHPL_GPU not defined)
else
CHPL_FLAGS+=--report-gpu -suseGPU=true --ldflags -no-pie --gpu-block-size=1024
CHPL_SOURCE+=ChapelStream.h
ifeq ($(CHPL_GPU),nvidia)
CHPL_SOURCE+=ChapelStream_cu.o
LDFLAGS+=-L$(CUDA_PATH)/lib64 -lcudart
else
ifeq ($(CHPL_GPU),amd)
CHPL_SOURCE+=ChapelStream_hip.o
LDFLAGS+=-lamdhip64
else
$(error Unknown value for CHPL_GPU: $(CHPL_GPU))
endif
endif
ifndef CHPL_GPU_ARCH
$(error CHPL_GPU_ARCH not defined)
endif
endif
else
# CPU version
ifdef CHPL_GPU
$(error CHPL_GPU is set, but CHPL_LOCALE_MODEL!= gpu)
endif
endif
ifdef DEBUG
  CHPL_FLAGS+=-g --devel
  CUDACXX_FLAGS=-g
else
  CHPL_FLAGS+=--fast
  CUDACXX_FLAGS+=-O3
endif

# -------
EXE = chapel-stream
.PHONY: all $(BENCHMARK_EXE) clean

all: $(EXE)

run:	$(EXE)
	./$(EXE) -n 2 -z 1024

ChapelStream_cu.o:	ChapelStream.cu
ifndef CUDA_PATH
	$(error CUDA_PATH not defined)
endif
	nvcc $(CUDACXX_FLAGS) -c -allow-unsupported-compiler ChapelStream.cu -o $@

ChapelStream.hip:	ChapelStream.cu
ifndef ROCM_PATH
	$(error ROCM_PATH not defined)
endif
	hipify-perl ChapelStream.cu > ChapelStream.hip

ChapelStream_hip.o:	ChapelStream.hip
ifndef ROCM_PATH
	$(error ROCM_PATH not defined)
endif
	hipcc $(CUDACXX_FLAGS) -c ChapelStream.hip -o $@

$(EXE):	$(CHPL_SOURCE)
	chpl $(CHPL_FLAGS) $(CHPL_SOURCE) $(LDFLAGS) -o $@

.PHONY:	clean

clean:	
	rm -rf $(EXE) ChapelStream_cu.o ChapelStream_hip.o

