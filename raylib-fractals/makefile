CC = gcc
CFLAGS_DEBUG = -g -Wall -Wextra -pthread -mavx2 -lfractal -lraylib -lm #-shared
CFLAGS_RELEASE = -O3 -Wall -Wextra -pthread -ffast-math -mavx2 -lfractal -lraylib -lm  -L. #-shared

FILES = src/main.c


build: 
	$(CC) $(FILES) -o a_release.exe $(CFLAGS_RELEASE)

run: build
	./a_release.exe

debug:
	$(CC) $(FILES) -o a_debug.exe $(CFLAGS_DEBUG)

memcheck: debug
	valgrind --leak-check=full ./a_debug.exe

clean:
	rm --force *.exe
	rm --force *.out.*
