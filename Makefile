CC = gcc
CFLAGS = -std=c99 -Wall -Wextra -O2 -Iinclude 
LDFLAGS = -lm

SRC = src/topology.c src/silk.c src/lifecycle.c src/flow.c src/seed.c src/loss.c src/echo.c src/mutate.c src/main.c
OBJ = $(SRC:.c=.o)
BIN = build/radnet_skeleton
TRAIN_BIN = build/train

all: $(BIN) $(TRAIN_BIN)

$(BIN): $(OBJ)
	@mkdir -p build
	$(CC) $(OBJ) -o $(BIN) $(LDFLAGS)

OBJ_LIB = $(filter-out src/main.o, $(OBJ))
$(TRAIN_BIN): $(OBJ_LIB) src/train.cpp
	@mkdir -p build
	g++ -std=c++11 -Wall -Wextra -O2 -Iinclude src/train.cpp $(OBJ_LIB) -o $(TRAIN_BIN) $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

run: all
	./$(BIN)

clean:
	rm -f $(OBJ) $(BIN)

.PHONY: all run clean
