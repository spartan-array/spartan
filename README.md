## Spartan

Spartan is a distributed array engine, built on top of a Piccolo-style
key-value store.

### Installation

Dependencies:

* Python development headers
* SWIG
* MPI
* Protocol buffers
* Boost threading library


Fetch the source:

    git clone https://github.com/rjpower/spartan
    

On Debian-based systems, dependencies can be installed via:

	sudo apt-get install build-essential libboost-dev libboost-thread-dev \
    libopenmpi-dev libprotobuf-dev openmpi-bin protobuf-compiler
    
Build:

    cd spartan
    autoreconf -fvi
    ./configure && make

### Running

Spartan compiles to a Python extension (spartan.py + _spartan.so).  These
should be on your PYTHONPATH for the remaining examples.

export PYTHONPATH=$PYTHONPATH:/path/to/spartan/build/
