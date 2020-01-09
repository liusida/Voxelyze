#include "VX3_SimulationManager.h"
#include "VX3_VoxelyzeKernel.h"
#include "VX_Sim.h"


__global__ void CUDA_Simulation(VX3_VoxelyzeKernel *d_voxelyze_3, int num_tasks) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i<num_tasks) {
        VX3_VoxelyzeKernel *d_v3 = &d_voxelyze_3[i];
        if (i==1) {
            printf("debug.\n");
        }
        printf("--> simulation %d runs.\n", i);
        for (int j=0;j<1000000;j++) { //Maximum Steps 1000000
            if (d_v3->StopConditionMet()) break;
            // if (j%1000==0) {
            //     printf("----> [Task %d] doTimeStep %d, Current Time (in sec) %f \t", i, j, d_v3->currentTime);
            //     d_v3->updateCurrentCenterOfMass();
            //     printf("Current Location (in meter): %f %f %f\n", d_v3->currentCenterOfMass.x, d_v3->currentCenterOfMass.y, d_v3->currentCenterOfMass.z);
            // }
            if (!d_v3->doTimeStep()) {
                printf("--> simulation %d Diverged.\n", i);
                break;
            }
        }
        d_v3->updateCurrentCenterOfMass();
        printf("--> simulation %d ends.\n", i);
    }
}

void VX3_SimulationManager::operator()(VX3_TaskManager* tm, fs::path batchFolder) {
    //TODO: 
    //1. read every VXA files
    VX3_VoxelyzeKernel * d_voxelyze_3;
    std::vector<VX3_VoxelyzeKernel *> h_d_voxelyze_3;
    int batch_size = 0;
    for (auto &file : fs::directory_iterator( batchFolder) ) { batch_size++; }
    cudaMalloc((void**)&d_voxelyze_3, batch_size * sizeof(VX3_VoxelyzeKernel));

    int i = 0;
    for (auto &file : fs::directory_iterator( batchFolder ) ) {
        CVX_Environment MainEnv;
        CVX_Sim MainSim;
        CVX_Object MainObj;
        MainEnv.pObj = &MainObj; //connect environment to object
        MainSim.pEnv = &MainEnv; //connect Simulation to envirnment
        MainSim.LoadVXAFile(file.path().string());
        std::string err_string; //need to link this up to get info back...
        if (!MainSim.Import(NULL, NULL, &err_string)){
            std::cout<<err_string;
        }
        VX3_VoxelyzeKernel h_d_tmp(&MainSim.Vx);
        h_d_tmp.DtFrac = MainSim.DtFrac;
        h_d_tmp.StopConditionType = MainSim.StopConditionType;
        h_d_tmp.StopConditionValue = MainSim.StopConditionValue;
        h_d_tmp.TempEnabled = MainSim.pEnv->TempEnabled;
        h_d_tmp.VaryTempEnabled = MainSim.pEnv->VaryTempEnabled;
        h_d_tmp.TempBase = MainSim.pEnv->TempBase;
        h_d_tmp.TempAmplitude = MainSim.pEnv->TempAmplitude;
        h_d_tmp.TempPeriod = MainSim.pEnv->TempPeriod;
        h_d_tmp.currentTemperature = h_d_tmp.TempBase + h_d_tmp.TempAmplitude;
        
        cudaMemcpy(d_voxelyze_3 + i, &h_d_tmp, sizeof(VX3_VoxelyzeKernel), cudaMemcpyHostToDevice);
        i++;
    }

    //3. start CUDA Simulation
    int num_tasks = batch_size;
    int threadsPerBlock = 512;
    int numBlocks = (num_tasks + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks == 1)
        threadsPerBlock = num_tasks;
    CUDA_Simulation<<<numBlocks,threadsPerBlock>>>(d_voxelyze_3, num_tasks);

    //4. wait
    cudaDeviceSynchronize();
    
    //5. read result
    double final_z = 0.0;
    VX3_VoxelyzeKernel* result_voxelyze_kernel = (VX3_VoxelyzeKernel *)malloc(num_tasks * sizeof(VX3_VoxelyzeKernel));
    cudaMemcpy( result_voxelyze_kernel, d_voxelyze_3, num_tasks * sizeof(VX3_VoxelyzeKernel), cudaMemcpyDeviceToHost );
    //TODO: how to communicate with experiments? files? or other methods?
    printf("====[RESULTS for %s]====", batchFolder.filename());
    for (int i=0;i<num_tasks;i++) {
        auto ret = &result_voxelyze_kernel[i];
        printf("Task %d: position (in mm): %f %f %f, end time: %f\n", i,
        ret->currentCenterOfMass.x*1000, ret->currentCenterOfMass.y*1000, ret->currentCenterOfMass.z*1000,
        ret->currentTime
        );
    }
    //6. cleanup
    for (auto p:h_d_voxelyze_3) {
        p->cleanup();
    }
    cudaFree(d_voxelyze_3);
    tm->cleanBatchFolder(batchFolder);
    // delete result_voxelyze_kernel;

    return;
}