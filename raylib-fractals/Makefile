CC = gcc
CFLAGS_DEBUG = -g -Wall -Wextra -pthread -mavx2 -lfractal -lraylib -L".."
CFLAGS_RELEASE = -O3 -Wall -Wextra -pthread -ffast-math -mavx2 -lfractal -lraylib -L".."

FILES = src/main.c


ifeq ($(OS),Windows_NT)
	PLATFORM_OS = WINDOWS
	CFLAGS_DEBUG += -lgdi32 -lWinmm
	CFLAGS_RELEASE += -lgdi32 -lWinmm
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
	CFLAGS_DEBUG += -lm -ldl
	CFLAGS_RELEASE += -lm -ldl
endif


build: 
	$(CC) $(FILES) -o a_release.exe $(CFLAGS_RELEASE)

build-cuda:
	$(CC) $(FILES) -DCUDA -o a_release.exe -lcudafractals $(CFLAGS_RELEASE)

run: build
	./a_release.exe

run-cuda: build-cuda
	./a_release.exe

debug:
	$(CC) $(FILES) -o a_debug.exe $(CFLAGS_DEBUG)

memcheck: debug
	valgrind --leak-check=full ./a_debug.exe

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
