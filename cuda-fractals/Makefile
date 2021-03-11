CC_CUDA = nvcc
CC = gcc

FILES_CUDA = \
	src/fractal.cu \
	src/utils.cu \

FILES = src/main.c

name_cuda = cudafractals
name = main


ifeq ($(OS),Windows_NT)
	PLATFORM_OS = WINDOWS
else
	UNAMEOS = $(shell uname)
	ifeq ($(UNAMEOS),Linux)
		PLATFORM_OS = LINUX
	endif
	ifeq ($(UNAMEOS),FreeBSD)
		PLATFORM_OS = BSD
	endif
	ifeq ($(UNAMEOS),OpenBSD)
		PLATFORM_OS = BSD
	endif
	ifeq ($(UNAMEOS),NetBSD)
		PLATFORM_OS = BSD
	endif
	ifeq ($(UNAMEOS),DragonFly)
		PLATFORM_OS = BSD
	endif
	ifeq ($(UNAMEOS),Darwin)
		PLATFORM_OS = OSX
	endif
endif


build: cuda_shared
	gcc -DCUDA -o $(name) -L. -l$(name_cuda) $(FILES)

run: build
	./$(name)

debug: cuda_shared
	gcc -DCUDA -g -o $(name) -L. -l$(name_cuda) $(FILES)


cuda_shared:
ifeq ($(PLATFORM_OS),WINDOWS)
	nvcc -o lib$(name_cuda).dll --shared $(FILES_CUDA)
	del lib$(name_cuda).exp /s
	del lib$(name_cuda).lib /s
else
	nvcc -o lib$(name_cuda).so --shared $(FILES_CUDA)
	rm lib$(name_cuda).exp lib$(name_cuda).lib
endif


clean:
ifeq ($(PLATFORM_OS),WINDOWS)
	del *.o /s
	del *.exe /s
	del *.dll /s
	del *.out.* /s
	del *.so /s
	del *.a /s
	del *.exp /s
	del *.lib /s
else
	rm -fv *.o *.exe *.dll *.so *.out.* *.a *.exp *.lib
endif
