CC = gcc
CFLAGS_RELEASE = -O3 -Wall -Wextra -pthread -ffast-math -mavx2
LIBS = -lfractal -lcudafractals -L..

FILES_EXAMPLES = \
	$(wildcard *.c) \

PROGS = $(patsubst %.c,%,$(FILES_EXAMPLES))


ifeq ($(OS),Windows_NT)
	PLATFORM_OS = WINDOWS
else
	PLATFORM_OS = LINUX
endif


all: $(PROGS)

%: %.c
	$(CC) -o $@ $< $(FILES) $(CFLAGS_RELEASE) -DEXAMPLES $(LIBS)

clean:
ifeq ($(PLATFORM_OS),WINDOWS)
	del *.exe /s
else
	rm -f $(PROGS)
endif
