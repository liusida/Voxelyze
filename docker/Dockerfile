FROM nvidia/cuda:10.1-devel-ubuntu16.04
RUN apt-get update; \
    apt-get install -y git wget tar libssl-dev software-properties-common
# ===
#install cmake
RUN wget https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2.tar.gz; \
    tar -xf cmake-3.16.2.tar.gz; \
    cd cmake-3.16.2; \
    ./bootstrap --parallel=10; \
    make -j 10; \
    make install; \
    export PATH=/usr/local/bin:$PATH
# cmake will be installed to /usr/local/bin/cmake
# ===
#install boost
RUN wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz; \
    tar -xf boost_1_66_0.tar.gz; \
    cd boost_1_66_0; \
    ./bootstrap.sh; \
    ./b2 --with-filesystem --with-thread install -j 10
# only build filesystem and thread to save time
# boost will be installed to /usr/local/include/boost and /usr/local/lib/boost
# ===
#install gcc-8
RUN add-apt-repository ppa:ubuntu-toolchain-r/test -y; \
    apt-get update; \
    apt-get install -y gcc-8 g++-8; \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10; \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10;
# change default g++ to g++-8
# ===
#install Voxelyze3
RUN git clone https://github.com/liusida/Voxelyze.git
RUN cd Voxelyze/; \
    git checkout dev-CUDA; \
    mkdir build; cd build; \
    cmake -DCMAKE_BUILD_TYPE=Release ..; \
    cmake --build . -j 10; cd ../; \
    mkdir taskPool; \
    mkdir taskPool/0_NewTasks; \
    mkdir taskPool/1_RunningTasks; \
    mkdir taskPool/2_FinishedTasks; \
    mkdir taskPool/CallTaskManager; \
    cp VXA_examples/* taskPool/0_NewTasks/ ; \
    touch taskPool/CallTaskManager/a
#VXA files are fed in, you can now run ./build/voxelyzeManager to see the simulation runs.
# ===