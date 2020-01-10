#include "VX3_TaskManager.h"
#include "VX3_SimulationManager.h"

#include <boost/thread.hpp>

bool VX3_TaskManager::checkForCalls() {
    fs::path p = PATH_CALLS;
    if (! fs::is_empty( p )) {
        for (auto file : fs::directory_iterator(p)) {
            fs::remove(file);
        }
        return true;
    }
    return false;
}

//create a batch folder and copy VXA files to that batch folder
fs::path VX3_TaskManager::batchVXAFiles() {
    fs::path p = PATH_NEW_TASK;
    fs::path running = PATH_RUNNING;
    fs::path batchFolder = running / u_format_now("Batch_%Y_%m_%d_%H_%M_%S");
    fs::create_directory(batchFolder);
    int count = 0;
    for (auto file : fs::directory_iterator(p)) {
        if (u_with_ext(file.path(), ".vxa")) {
            fs::path newfile = batchFolder / file.path().filename();
            fs::rename(file.path(), newfile);
            count ++;
        }
    }
    printf("New batch with %d VXA files.\n", count);
    return batchFolder;
}

void VX3_TaskManager::cleanBatchFolder(fs::path batchFolder) {
    fs::path finished = PATH_FINISHED;
    finished /= batchFolder.filename();
    fs::rename(batchFolder, finished);
    printf("One batch of simulations finished. (%s)\n", batchFolder.filename().c_str());
}

void VX3_TaskManager::makeTaskPool() {
    printf("Making the folders for tasks: %s\n", fs::canonical(fs::path(PATH_POOL)).c_str());
    fs::create_directory(PATH_POOL);
    fs::create_directory(PATH_CALLS);
    fs::create_directory(PATH_NEW_TASK);
    fs::create_directory(PATH_RUNNING);
    fs::create_directory(PATH_FINISHED);
    printf("\nPlease put your VXA files in \n %s\n and touch a new file in \n %s\n to tell the manager to start simulations.\n", PATH_NEW_TASK, PATH_CALLS);
}
void VX3_TaskManager::start() {

    makeTaskPool();

    while(1) {
        try {
            if (checkForCalls()) {
                printf("New call received.\n");
                fs::path batchFolder = batchVXAFiles();
                //Start Simulater and pass batchFolder to it
                boost::thread th1(VX3_SimulationManager(), this, batchFolder);
                printf("Continue watching for new calls.\n");
            } else {
                //printf("waiting for calls.\n");
            }
        } catch(const std::exception& e) {
            printf("ERROR: ignored.\n");
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}