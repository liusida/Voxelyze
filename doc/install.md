# Install the development environment

Installing CUDA is a pain, but luckily NVidia provides docker images for those environments. Assuming you have docker installed, and let's install the development environement together :)

(But how to install Docker? [Official Document](https://docs.docker.com/v17.09/engine/installation/linux/docker-ce/ubuntu/) and [Easy Steps](https://computingforgeeks.com/how-to-install-docker-on-ubuntu/))

```bash
cd docker
docker build .
```

And wait for 30 mins, you will be all set, then you can start the virtual environment with:

```bash
docker run -it --gpus all voxelyze
```

remember, docker is designed to for clean deployment, so you may lost all your changes inside the container after shutting down the virtual environment. You may want to learn more about docker to save your changes inside docker environment. If you accidentally exit the docker container, use this to re-join (replace \<container-name> by what you see in `docker ps -a`):

```bash
docker ps -a
docker restart <container-name>
docker attach <container-name>
```

= Or, here are steps you can follow manually: = 

In host
```bash
docker pull nvidia/cuda:10.1-base-ubuntu16.04
docker run -it --gpus all nvidia/cuda:10.1-base-ubuntu16.04
```

In docker
```bash
apt-get update
apt-get install -y git wget tar libssl-dev software-properties-common

#install cmake
wget https://github.com/Kitware/CMake/releases/download/v3.16.2/cmake-3.16.2.tar.gz
tar -xf cmake-3.16.2.tar.gz
cd cmake-3.16.2
./bootstrap --parallel=10
make -j 10
make install
export PATH=/usr/local/bin:$PATH
# cmake will be installed to /usr/local/bin/cmake

#install boost
wget https://dl.bintray.com/boostorg/release/1.66.0/source/boost_1_66_0.tar.gz
tar -xf boost_1_66_0.tar.gz
cd boost_1_66_0
./bootstrap.sh
# only build filesystem and thread to save time
./b2 --with-filesystem --with-thread install -j 10
# boost will be installed to /usr/local/include/boost and /usr/local/lib/boost

#install gcc-8
add-apt-repository ppa:ubuntu-toolchain-r/test -y
apt-get update
apt-get install -y gcc-8 g++-8 
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 10
# change default g++ to g++-8

#install Voxelyze3
git clone https://github.com/liusida/Voxelyze.git
cd Voxelyze/
git checkout dev-CUDA
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j 10
cd ../
mkdir taskPool
mkdir taskPool/0_NewTasks
mkdir taskPool/1_RunningTasks
mkdir taskPool/2_FinishedTasks
mkdir taskPool/CallTaskManager
#To feed in VXA files:
cp VXA_examples/* taskPool/0_NewTasks/ ; touch taskPool/CallTaskManager/a
./build/voxelyzeManager
```

Hooray! We should see the simulations and report generated!
