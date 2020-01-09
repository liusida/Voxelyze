#include "Voxelyze3.h"

Voxelyze3::Voxelyze3(/* args */)
{
}

Voxelyze3::~Voxelyze3()
{
}

void Voxelyze3::loadVXA(std::string VXAfilename) {
    boost::property_tree::ptree pt;
    boost::property_tree::read_xml(VXAfilename, pt);

    double DtFrac = pt.get<double>("VXA.Simulator.Integration.DtFrac", 0.9);
    printf("DtFrac %e\n", DtFrac);

    voxSize         = pt.get<double>("VXA.VXC.Lattice.Lattice_Dim", 0.001);
    ambientTemp     = pt.get<double>("VXA.Environment.Thermal.TempBase", 25);
    grav            = pt.get<double>("VXA.Environment.Gravity.GravAcc", -9.81);
    floor           = pt.get<int>   ("VXA.Environment.Gravity.FloorEnabled", 1);
    collisions      = pt.get<int>   ("VXA.Simulator.Collisions.SelfColEnabled", 0);

}

__host__ __device__ bool Voxelyze3::doTimeStep(double dt) {
    return false;
}

CUDA_Voxelyze3::CUDA_Voxelyze3(/* args */)
{
}

CUDA_Voxelyze3::~CUDA_Voxelyze3()
{
}

void CUDA_Voxelyze3::setHost(const Voxelyze3 &h_v3) {
    strcpy(vxa_path, h_v3.vxa_path);
}
