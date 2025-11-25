
BUILD_TYPE ?= Release
## switch between Release and Debug

all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)
	cmake --build build

clean:
	rm -rf build

run:
	./build/cuda