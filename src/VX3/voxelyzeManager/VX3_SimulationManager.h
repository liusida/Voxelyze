#if !defined(VX3_SIMULATION_MANAGER)
#define VX3_SIMULATION_MANAGER

#include <thread>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "VX3_TaskManager.h"

class VX3_SimulationManager
{
private:
    /* data */
public:
    VX3_SimulationManager() = default;
    
    //Overload operator to start thread
    void operator()(VX3_TaskManager* tm, fs::path batchFolder);
};

#endif // VX3_SIMULATION_MANAGER
