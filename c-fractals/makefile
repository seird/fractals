CC = gcc
CFLAGS_DEBUG = -g -Wall -Wextra -pthread -mavx2 #-shared
CFLAGS_RELEASE = -O3 -Wall -Wextra -pthread -ffast-math -mavx2 #-shared
CFLAGS_RELEASE_LIB = -O3 -Wall -Wextra -shared -DSHARED -pthread -ffast-math -mavx2 -fPIC
CFLAGS_RELEASE_LIB_STATIC = -O3 -Wall -Wextra -pthread -DSTATIC -ffast-math -mavx2 -fPIC -c



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



library:
	$(CC) $(CFLAGS_RELEASE_LIB) src/*.c -o libfractal.dll

static:
	$(CC) $(CFLAGS_RELEASE_LIB_STATIC) src/*.c
	ar -rc libfractal.a *.o

build: 
	$(CC) $(CFLAGS_RELEASE) src/*.c -o a_release.exe

run: build
	./a_release.exe

debug:
	$(CC) $(CFLAGS_DEBUG) src/*.c -o a_debug.exe -lm

profile: debug
	valgrind --tool=callgrind ./a_debug.exe

memcheck: debug
	valgrind --leak-check=full ./a_debug.exe

cache: build
	valgrind --tool=cachegrind ./a_release.exe
	#cg_annotate cachegrind.out.{PID}

test:
	$(CC) $(CFLAGS_RELEASE) -DTEST tests/*.c src/*.c -o test.exe
	./test.exe

bench:
	$(CC) $(CFLAGS_RELEASE) -DBENCHMARK benchmark/*.c src/*.c -o benchmark.exe -lm
	./benchmark.exe

install: library
ifneq ($(OS),Windows_NT)
ifeq ($(shell uname),Linux)
	cp libfractal.dll /usr/local/lib/libfractal.so
	cp src/fractal_color.h /usr/local/include/fractal_color.h
endif
endif

clean:
ifeq ($(PLATFORM_OS),WINDOWS)
	del *.o /s
	del *.exe /s
	del *.dll /s
	del *.out.* /s
	del *.so /s
	del *.a /s
else
	rm -fv *.o *.exe *.dll *.so *.out.* *.a
endif
