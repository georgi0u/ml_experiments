# the compiler: gcc for C program, define as g++ for C++
	CC = g++

# compiler flags:
	CFLAGS  = -Ofast -std=c++11 -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3

# The build target 
	TARGET = network

	all: $(TARGET)

	$(TARGET): $(TARGET).cpp
		$(CC) $(CFLAGS) -o $(TARGET) $(TARGET).cpp

	clean:
	$(RM) $(TARGET)
