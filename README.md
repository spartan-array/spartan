## Sparrow

Sparrow is a distributed array engine, built on top of a Piccolo-style
updateable key-value store.

### Installation

Dependencies:

* Python development headers
* SWIG
* MPI
* Protocol buffers
* Boost threading library

On Debian-based systems, dependencies can be installed via:

	sudo apt-get install build-essential libboost-dev libboost-thread-dev \
    libopenmpi-dev libprotobuf-dev openmpi-bin protobuf-compiler  
    
Fetch the source:

    git clone https://github.com/rjpower/sparrow
    
Build:

    cd sparrow
    autoreconf -fvi
    ./configure && make

### Running

Sparrow compiles to a Python extension (sparrow.py + _sparrow.so).  These
should be on your PYTHONPATH for the remaining examples.

export PYTHONPATH=$PYTHONPATH:/path/to/sparrow/build/
