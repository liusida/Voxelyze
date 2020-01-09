#if !defined(VOXELYZE_3_H)
#define VOXELYZE_3_H
#include <iostream>
#include <cstring>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "VX3.h"
#include "Array3D.h"
#include "VX3_Voxel.h"

class Voxelyze3
{
public:
    Voxelyze3();
    ~Voxelyze3();
    CUDA_CALLABLE bool doTimeStep(double dt = -1.0f);
    void loadVXA(std::string VXAfilename);
    /* data */
    char vxa_path[256];

    double voxSize;
    double currentTime;
    double ambientTemp;
    double grav;
    bool floor, collisions;

    /* ignore items below for now */
    double boundingRadius;
    double watchDistance;
    bool collisionsStale;
    bool nearbyStale;
    /* data end */

    /* pointers */
    CArray3D<VX3_Voxel*> voxels; //main voxel array 3D lookup

    /* pointers end */
};

class CUDA_Voxelyze3 : public Voxelyze3
{
public:
    CUDA_Voxelyze3();
    ~CUDA_Voxelyze3();

    void setHost(const Voxelyze3 &);
    /* data */

};



#endif // VOXELYZE_3_H
