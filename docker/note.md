This is a note of what command should be executed during install.

In host
```bash
docker pull nvidia/cuda:10.1-base-ubuntu16.04
```

In docker
```bash
apt update
apt install -y git cmake
git clone https://github.com/liusida/Voxelyze.git
cd Voxelyze/
git checkout dev-CUDA
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..