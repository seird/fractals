CC = gcc
CFLAGS_DEBUG = -g -Wall -pthread -mavx2 #-shared
CFLAGS_RELEASE = -O3 -Wall -pthread -mavx2 #-shared
CFLAGS_RELEASE_LIB = -O3 -Wall -shared -DSHARED -pthread -mavx2 

library:
	$(CC) $(CFLAGS_RELEASE_LIB) src/*.c -o libfractal.dll

build: 
	$(CC) $(CFLAGS_RELEASE) src/*.c -o a_release.exe

run: build
	./a_release.exe

debug:
	$(CC) $(CFLAGS_DEBUG) src/*.c -o a_debug.exe -lm

profile: debug
	valgrind --tool=callgrind ./a_debug.exe

memcheck: debug
	valgrind --leak-check=yes ./a_debug.exe

cache: build
	valgrind --tool=cachegrind ./a_release.exe
	#cg_annotate cachegrind.out.{PID}

test:
	$(CC) $(CFLAGS_RELEASE) -DTEST tests/*.c src/*.c -o test.exe
	./test.exe

bench:
	$(CC) $(CFLAGS_RELEASE) -DBENCHMARK benchmark/*.c src/*.c -o benchmark.exe
	./benchmark.exe

clean:
	rm --force *.exe
	rm --force *.out.*