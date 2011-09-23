all: pygpu_ndarray.so

PYTHON_VERSION ?= 2.5
CUDA_ROOT ?= /opt/lisa/os/cuda
THEANO_ROOT ?= /u/bastienf/repos/Theano

CFLAGS=-g
#BINDIR=--compiler-bindir ${HOME}/.theano.nvcc-bindir

#NPY_PATH!=python -c "import numpy;print numpy.__path__"
#NPY_INCLUDE=-I${NPY_PATH}/core/include
THEANO_CUDA_INCLUDE=-I${THEANO_ROOT}/theano/sandbox/cuda
CUDA_INCLUDE=-I${CUDA_ROOT}/include
#PYTHON_INCLUDE=-I/usr/include/python${PYTHON_VERSION}
INCLUDES=${CUDA_INCLUDE} ${THEANO_CUDA_INCLUDE} ${PYTHON_INCLUDE}
CUDA_FLAGS=-Xlinker -rpath -Xlinker ${CUDA_ROOT}/lib64 -Xlinker -rpath -Xlinker ${CUDA_ROOT}/lib

pygpu_language_cuda.o: pygpu_language_cuda.cu pygpu_language.h
	nvcc -c ${CFLAGS} -m64 -Xcompiler -fPIC,-m64 ${CUDA_FLAGS} ${INCLUDES} ${BINDIR} -o $@ $<

pygpu_ndarray.so: pygpu_ndarray.cpp pygpu_ndarray.h pygpu_language_cuda.o pygpu_ndarray_object.h
	g++ -shared ${CFLAGS} -m64 -fPIC ${CUDA_FLAGS} ${INCLUDES} ${BINDIR} -o $@ pygpu_language_cuda.o $< -lpython${PYTHON_VERSION} -lcublas -lcudart

clean:
	rm -f pygpu_ndarray.so core.* *.o *~

cleantmp:
	rm -f core.* *.o *~

BigLearn_NipsWorkshop.pdf: BigLearn_NipsWorkshop.tex
	#latex BigLearn_NipsWorkshop.tex
	#dvipdf latex BigLearn_NipsWorkshop
	pdflatex BigLearn_NipsWorkshop
	pdflatex BigLearn_NipsWorkshop
	bibtexall
	pdflatex BigLearn_NipsWorkshop
