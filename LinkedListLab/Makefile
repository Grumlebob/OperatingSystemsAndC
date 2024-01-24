#
# Makefile for clab
#

CC = gcc

CFLAGS = -std=c99 -Og -g -Wall -Wextra -Wpedantic -Wconversion
CFLAGS += -Wstrict-prototypes -Wwrite-strings -Werror -Wvla

# Targets to compile
FILES = qtest

.PHONY: all
all: $(FILES)

# List header dependencies
qtest.o: harness.h queue.h report.h console.h
harness.o: harness.h
console.o: report.h console.h
report.o: report.h
queue.o: harness.h queue.h

# Compile object files
%.o: %.c
	$(CC) $(CFLAGS) -o $@ -c $<

# Compile qtest binaries
qtest: qtest.o report.o console.o harness.o queue.o
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: run
run: qtest
	./qtest

.PHONY: test
test: qtest driver.py
	chmod +x driver.py
	./driver.py

.PHONY: clean
clean:
	rm -rf $(FILES)
	rm -rf *.o *~ *.pyc *.dSYM
	(cd traces; rm -f *~)
