#if !defined(VX3_TASKMANAGER_H)
#define VX3_TASKMANAGER_H

#include "VX3_Utils.h"

#include <chrono>
#include <thread>
#include <iostream>
#include <string>
#include <exception>
#include <cctype>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#define PATH_POOL "./taskPool/"
#define PATH_CALLS "./taskPool/CallTaskManager/"
#define PATH_NEW_TASK "./taskPool/0_NewTasks/"
#define PATH_RUNNING "./taskPool/1_RunningTasks/"
#define PATH_FINISHED "./taskPool/2_FinishedTasks/"

class VX3_TaskManager
{
private:
    /* data */
public:
    VX3_TaskManager()=default;

    void start(int how_many_runs);
    bool checkForCalls();
    fs::path batchVXAFiles();
    void cleanBatchFolder(fs::path batchFolder);
    void makeTaskPool();
};

#endif // VX3_TASKMANAGER_H
