CC = gcc
CFLAGS_DEBUG = -g -pthread -lOpenCL 
CFLAGS_RELEASE = -O3 -pthread -lOpenCL 
CFLAGS_LIBRARY = -O3 -pthread -lOpenCL 

FILES = \
	$(wildcard src/*.c) \

PNAME   ?= openclfractals
LIBTYPE ?= STATIC


ifeq ($(OS),Windows_NT)
	PLATFORM_OS = WINDOWS
else
	PLATFORM_OS = LINUX
endif


ifeq ($(LIBTYPE),SHARED)
	ifeq ($(PLATFORM_OS),WINDOWS)
		LIBNAME = lib$(PNAME).dll
	endif
	ifeq ($(PLATFORM_OS),LINUX)
		LIBNAME = lib$(PNAME).so
	endif
    CFLAGS_LIBRARY += -o $(LIBNAME) -fPIC -shared -DSHARED
endif
ifeq ($(LIBTYPE),STATIC)
    CFLAGS_LIBRARY += -c -DSTATIC
	LIBNAME = lib$(PNAME).a
endif


ifeq ($(PLATFORM_OS),WINDOWS)
	DIRS_INCLUDE = -I./OpenCL-Headers
	DIRS_LIB = -L./lib
endif
ifeq ($(PLATFORM_OS),LINUX)
	DIRS_INCLUDE = -I/usr/local/include/
	DIRS_LIB = -L/usr/local/lib/
endif


build: convert-kernels
	$(CC) -o $(PNAME)_release $(FILES) $(DIRS_INCLUDE) $(DIRS_LIB) $(CFLAGS_RELEASE)

convert-kernels:
	python convert_kernels.py

library: convert-kernels
	$(CC) $(FILES) $(DIRS_INCLUDE) $(DIRS_LIB) $(CFLAGS_LIBRARY)
ifeq ($(LIBTYPE),STATIC)
	ar -rc $(LIBNAME) *.o
endif

# for linking to a project that already implemented stb_image
# -- static only --
library-no-stb: convert-kernels
	$(CC) $(FILES) $(DIRS_INCLUDE) $(DIRS_LIB) $(CFLAGS_LIBRARY) -DNOSTB
	ar -rc $(LIBNAME) *.o
	xcopy /y $(LIBNAME) ..

run: build
	./$(PNAME)_release

debug: convert-kernels
	$(CC) -o $(PNAME)_debug $(FILES) $(DIRS_INCLUDE) $(DIRS_LIB) $(CFLAGS_DEBUG)

profile: debug
	valgrind --tool=callgrind ./$(PNAME)_debug

gprof:
	$(CC) -o $(PNAME)_gprof $(FILES) $(CFLAGS_DEBUG) -pg -lm
ifeq ($(PLATFORM_OS),WINDOWS)
	del gmon.out /s
	./$(PNAME)_gprof.exe
	gprof $(PNAME)_gprof.exe gmon.out > a_debug-gprof.out
else
	rm -fv gmon.out
	./$(PNAME)_gprof
	gprof $(PNAME)_gprof gmon.out > a_debug-gprof.out
endif

memcheck: debug
	valgrind --leak-check=full ./$(PNAME)_debug

cache: build
	valgrind --tool=cachegrind ./$(PNAME)_release
	#cg_annotate cachegrind.out.{PID}

clean:
ifeq ($(PLATFORM_OS),WINDOWS)
	del *.o /s
	del *.exe /s
	del *.dll /s
	del *.out.* /s
	del *.so /s
	del *.a /s
	del *.gcda /s
	del *.gcno /s
else
	rm -fv *.o *.exe *.dll *.so *.out.* *.a *.gcda *.gcno
endif
