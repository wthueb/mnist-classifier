CC := g++
CFLAGS := -g -Wall -std=c++14 -Wextra -MD -MP
SRC_EXT := cpp

TARGET := bin/mnist
 
LIB :=
INC :=

SRC_DIR := src
BUILD_DIR := build

SRC := $(shell find $(SRC_DIR) -type f -name *.$(SRC_EXT))
OBJ := $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(SRC:.$(SRC_EXT)=.o))

$(TARGET): $(OBJ)
	$(CC) $^ -o $(TARGET) $(LIB)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.$(SRC_EXT)
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(INC) -c -o $@ $<

clean:
	rm -r $(BUILD_DIR) $(TARGET)

-include $(patsubst $(SRC_DIR)/%,$(BUILD_DIR)/%,$(SRC:.$(SRC_EXT)=.d))
