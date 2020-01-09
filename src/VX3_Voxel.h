#if !defined(VX3_VOXEL_H)
#define VX3_VOXEL_H

#include "VX3_Vec3D.h"
#include "VX3_Quat3D.h"

class VX3_Voxel
{
public:
    VX3_Voxel(/* args */);
    ~VX3_Voxel();

    /* data */
    int ix, iy, iz;
    Vec3D<double> pos;                  //current center position (meters) (GCS)
    Vec3D<double> linMom;				//current linear momentum (kg*m/s) (GCS)
	Quat3D<double> orient;				//current orientation (GCS)
	Vec3D<double> angMom;				//current angular momentum (kg*m^2/s) (GCS)
	double temp; //0 is no expansion
	Vec3D<double> pStrain; //cached poissons strain
	bool poissonsStrainInvalid; //flag for recomputing poissons strain.
	double previousDt; //remember the duration of the last timestep of this voxel
	Vec3D<double>* lastColWatchPosition;
    /* data end */
    
};

class CUDA_VX3_Voxel : public VX3_Voxel
{
public:
    CUDA_VX3_Voxel(/* args */);
    ~CUDA_VX3_Voxel();
};



#endif // VX3_VOXEL_H
