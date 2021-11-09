CC = gcc
CFLAGS_DEBUG = -g -Wall -Wextra -pthread -mavx2 #-shared
CFLAGS_RELEASE = -O3 -Wall -Wextra -pthread -ffast-math -mavx2 #-shared
CFLAGS_RELEASE_LIB = -O3 -Wall -Wextra -shared -DSHARED -pthread -ffast-math -march=native -mavx2 -fPIC
CFLAGS_RELEASE_LIB_STATIC = -O3 -Wall -Wextra -pthread -DSTATIC -ffast-math -march=native -mavx2 -fPIC -c

FILES = \
	$(wildcard src/*.c) \

FILES_TEST = \
	$(FILES) \
	$(wildcard tests/*.c) \

FILES_BENCH = \
	$(FILES) \
	$(wildcard benchmark/*.c) \

OBJS = $(addsuffix .o, $(basename $(notdir $(FILES))))


ifeq ($(OS),Windows_NT)
	PLATFORM_OS = WINDOWS
	LIBEXT = dll
else
	PLATFORM_OS = LINUX
	LIBEXT = so
endif



library:
	$(CC) $(CFLAGS_RELEASE_LIB) $(FILES) -o libfractal.$(LIBEXT)

static:
	$(CC) $(CFLAGS_RELEASE_LIB_STATIC) $(FILES)
	ar -rc libfractal.a $(OBJS)

build: 
	$(CC) $(CFLAGS_RELEASE) $(FILES) -o a_release.exe

build-cuda: 
	$(CC) $(CFLAGS_RELEASE) -DCUDA -lcudafractals -L. $(FILES) -o a_release.exe

run: build
	./a_release.exe

run-cuda: build-cuda
	./a_release.exe

debug:
	$(CC) $(CFLAGS_DEBUG) $(FILES) -o a_debug.exe -lm

profile: debug
	valgrind --tool=callgrind ./a_debug.exe

memcheck: debug
	valgrind --leak-check=full ./a_debug.exe

cache: build
	valgrind --tool=cachegrind ./a_release.exe
	#cg_annotate cachegrind.out.{PID}

test:
	$(CC) $(CFLAGS_RELEASE) -fprofile-arcs -ftest-coverage -DTEST $(FILES_TEST) -o test.exe
	./test.exe
	gcovr -e "tests/*" -e "src/stb_image_write.h" --xml-pretty --exclude-unreachable-branches --print-summary -o coverage.xml

coverage_html:
	$(CC) $(CFLAGS_RELEASE) -fprofile-arcs -ftest-coverage -DTEST $(FILES_TEST) -o test.exe
	./test.exe
	gcovr -e "tests/*" -e "src/stb_image_write.h" --html --html-details --exclude-unreachable-branches --print-summary -o coverage.html

bench:
	$(CC) $(CFLAGS_RELEASE) -DBENCHMARK $(FILES_BENCH) -o benchmark.exe -lm
	./benchmark.exe

bench-cuda:
	$(CC) $(CFLAGS_RELEASE) -DBENCHMARK -DCUDA  -L. $(FILES_BENCH) -o benchmark.exe -lm -lcudafractals
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
	del *.xml /s
	del *.html /s
	del *.gcno /s
	del *.gcda /s
else
	rm -fv *.o *.exe *.dll *.so *.out.* *.a *.xml *.html *.gcno *.gcda
endif
