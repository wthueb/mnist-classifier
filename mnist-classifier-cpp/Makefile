CC := g++
CFLAGS := -g -Wall -std=c++14
LIB :=
INC :=

SRCDIR := src
BUILDDIR := build

TARGET := bin/mnist
 
SRCEXT := cpp
SOURCES := $(shell find $(SRCDIR) -type f -name *.$(SRCEXT))
OBJECTS := $(patsubst $(SRCDIR)/%,$(BUILDDIR)/%,$(SOURCES:.$(SRCEXT)=.o))

$(TARGET): $(OBJECTS)
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILDDIR)/%.o: $(SRCDIR)/%.$(SRCEXT)
	mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	rm -r $(BUILDDIR) $(TARGET)
