#include <iostream>
#include <math.h>
#include "mpi.h"

using namespace std;

int main(int argc,char *argv[])
{
	
 int myid,numprocs,namelen;
 char processor_name[MPI_MAX_PROCESSOR_NAME];
 MPI_Init(&argc,&argv);
 MPI_Comm_rank(MPI_COMM_WORLD,&myid);
 MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
 MPI_Get_processor_name(processor_name,&namelen);
 
 cout<<myid<<'\n';

 MPI_Finalize();
 return 0;
}