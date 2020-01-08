# Branch dev-CUDA

In this branch, I am going to implement another version of Voxelyze. Let's call it Voxelyze 3 for now.

So all simulational calculation in Voxelyze 3 will be running via CUDA.

If you would like to help or are simply interested, please contact me (Sida.Liu.1 at uvm.edu). (This project will mainly use C++, and here is [my brief plan](https://github.com/liusida/Voxelyze/blob/dev-CUDA/doc/Plan.txt).)

## First let me compare Voxelyze 1 and Voxelyze 2

![VX1 vs VX2](https://github.com/liusida/Voxelyze/blob/dev-CUDA/doc/VX1vsVX2.png?raw=true)

The distance is a little bit different, and people think it's fine. So Voxelyze 3 could have a little difference as well. Let's see.

## VXA, VXC File Format

There're many concepts and terminologies in the configure files, and here is [an incomplete note](https://github.com/liusida/Voxelyze/blob/dev-CUDA/doc/Format_of_VXA.txt) (I like VoxCAD, so I also point out where a certain parameter is used in VoxCAD).
