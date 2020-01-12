#include <iostream>
#include "VX3_TaskManager.h"
#include "VX3.h"
int main(int argc, char *argv[])
{
	int how_many_runs = 0;
	if (argc>1) {
		how_many_runs = atoi(argv[1]);
		printf(COLORCODE_BOLD_RED "For debugging reason: ONLY RUN %d TIMES!\n" COLORCODE_RESET, how_many_runs);
	}
	VX3_TaskManager tm;
	
	tm.start(how_many_runs);
	//loop forever

}
