#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>

//Group: John Kuss, Robert Kuss

//Declare variables for values and accels to be allocated for device.
vector3* d_values;
vector3** d_accels;
double *d_mass;

//allocDeviceMemory
//Parameters: None
//Returns: None
//Helper function to cudaMalloc all neccessary device variables, and cudaMemcopy where applicable (?)
void allocDeviceMemory(){
	//vector3* d_values;
	//vector3** d_accels;
	cudaMalloc(&d_hVel, sizeof(vector3)*NUMENTITIES);
	cudaMalloc(&d_hPos, sizeof(vector3)*NUMENTITIES);
	cudaMalloc(&d_mass, sizeof(double)*NUMENTITIES);
	cudaMemcpy(&d_hVel, &hVel, NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(&d_hPos, &hPos, NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMemcpy(&d_mass, &mass, NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMalloc(&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	cudaMalloc(&d_accels, sizeof(vector3)*NUMENTITIES);
}

//freeDeviceMemory
//Parameters: None
//Returns: None
//Helper function to cudaFree all device variables.
void freeDeviceMemory(){
	cudaFree(&d_hVel);
	cudaFree(&d_hPos);
	cudaFree(&d_mass);
	cudaFree(&d_values);
	cudaFree(&d_accels);
}

__global__ void computePairwiseAccel(vector3* d_values, vector3** d_accels, vector3* d_hPos, double* d_mass) {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        int j=blockIdx.y*blockDim.y+threadIdx.y;
        int k;
        if (i==j) {
                FILL_VECTOR(d_accels[i][j],0,0,0);
        }
        else{
                vector3 distance;
                for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                double magnitude=sqrt(magnitude_sq);
                double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
                FILL_VECTOR(d_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
}

__global__ void accelSum(vector3 *accel_sum, vector3* d_hPos,  vector3* d_hVel, vector3** d_accels) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y+blockDim.y+threadIdx.y;
	int k;
	for (k=0;k<3;k++)
		atomicAdd(accel_sum[k],d_accels[i][j][k]);
	__synchthreads();
	for (k=0;k<3;k++){
		atomicAdd(d_hVel[i][k],accel_sum[k]*INTERVAL);
		atomicAdd(d_hPos[i][k],d_hVel[i][k]*INTERVAL);
	}
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	//printf("Start compute.\n");
	//make an acceleration matrix which is NUMENTITIES squared in size;
	int i;
	//vector3* values;
	//vector3** accels;
	for (i=0;i<NUMENTITIES;i++)
		d_accels[i]=&d_values[i*NUMENTITIES];
	//Kernel variables
	dim3 threadsPerBlock(NUMENTITIES,NUMENTITIES);
	dim3 numBlocks(threadsPerBlock.x/2,threadsPerBlock.y/2);
	//cudaMalloc(&d_values, sizeof(vector3)*NUMENTITIES*NUMENTITIES);
        //cudaMalloc(&d_accels, sizeof(vector3)*NUMENTITIES);
	//first compute the pairwise accelerations.  Effect is on the first argument.
	computePairwiseAccel<<<numBlocks,threadsPerBlock>>>(d_values, d_accels, d_hPos, d_mass);
	/*for (i=0;i<NUMENTITIES;i++){
		for (j=0;j<NUMENTITIES;j++){
			if (i==j) {
				FILL_VECTOR(accels[i][j],0,0,0);
			}
			else{
				vector3 distance;
				for (k=0;k<3;k++) distance[k]=hPos[i][k]-hPos[j][k];
				double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
				double magnitude=sqrt(magnitude_sq);
				double accelmag=-1*GRAV_CONSTANT*mass[j]/magnitude_sq;
				FILL_VECTOR(accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
			}
		}
	}*/
	//sum up the rows of our matrix to get effect on each entity, then update velocity and position.
	vector3 accel_sum={0,0,0};
	accelSum<<<numBlocks,threadsPerBlock>>>(&accel_sum, d_hPos, d_hVel, d_accels);
	/*for (i=0;i<NUMENTITIES;i++){
		vector3 accel_sum={0,0,0};
		for (j=0;j<NUMENTITIES;j++){
			for (k=0;k<3;k++)
				accel_sum[k]+=accels[i][j][k];
		}
		//compute the new velocity based on the acceleration and time interval
		//compute the new position based on the velocity and time interval
		for (k=0;k<3;k++){
			hVel[i][k]+=accel_sum[k]*INTERVAL;
			hPos[i][k]+=hVel[i][k]*INTERVAL;
		}
	}*/
	//cudaFree(accels);
	//cudaFree(values);
#ifdef DEBUG
	cudaMemcpy(&hVel, &d_hVel, NUMENTITIES, cudaMemcpyDeviceToHost);
        cudaMemcpy(&hPos, &d_hPos, NUMENTITIES, cudaMemcpyDeviceToHost);
#endif
}

/*__global__ void computePairwiseAccel(vector3* d_values, vector3** d_accels, vector3* d_hPos, float* d_mass) {
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int j=blockIdx.y*blockDim.y+threadIdx.y;
	int k;
	if (i==j) {
		FILL_VECTOR(d_accels[i][j],0,0,0);
	}
	else{
		vector3 distance;
		for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
		double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
		double magnitude=sqrt(magnitude_sq);
		double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
		FILL_VECTOR(d_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
	}
}
__global__ void accelSum() {
	
}*/
