#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <stdio.h>
#include <cuda_runtime.h>

//Group: John Kuss, Robert Kuss

//Declare variables for values and accels to be allocated for device.

vector3* d_values;
vector3** d_accels;
double *d_mass;
vector3* accel_sum;//={0,0,0};
vector3* d_accel_sum;//={0,0,0};

void cudaCheckError() {
	cudaError_t e=cudaGetLastError();
	if (e!=cudaSuccess) {
		printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));
		exit(0);
	}
}

//allocDeviceMemory
//Parameters: None
//Returns: None
//Helper function to cudaMalloc all neccessary device variables, and cudaMemcopy where applicable (?)
void allocDeviceMemory(){

	cudaMalloc((void**)&d_accels, (NUMENTITIES)*sizeof(vector3));
	cudaCheckError();
	vector3* temp[NUMENTITIES];
	for (int i = 0; i<NUMENTITIES; i++){
		cudaMalloc(&temp[i], sizeof(vector3)*NUMENTITIES);
	}
	cudaMemcpy(d_accels, temp, sizeof(vector3*)*NUMENTITIES, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_hPos, (NUMENTITIES)*sizeof(vector3));
	cudaCheckError();
	cudaMalloc((void**)&d_hVel, NUMENTITIES*sizeof(vector3));
	cudaCheckError();
	cudaMalloc((void**)&d_mass, (NUMENTITIES)*sizeof(double));
	cudaCheckError();
	cudaMemcpy(d_hPos, hPos, (NUMENTITIES)*sizeof(vector3), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMemcpy(d_hVel, hVel, (NUMENTITIES)*sizeof(vector3), cudaMemcpyHostToDevice);
	cudaCheckError();
	cudaMalloc((void**)&d_accel_sum,NUMENTITIES*sizeof(vector3));
}

//freeDeviceMemory
//Parameters: None
//Returns: None
//Helper function to cudaFree all device variables.
void freeDeviceMemory(){
	cudaFree(d_hVel);
	//cudaCheckError();
	cudaFree(d_hPos);
	//cudaCheckError();
	cudaFree(d_mass);
	//cudaCheckError();
	cudaFree(d_values);
	//cudaCheckError();
	cudaFree(d_accels);
	//cudaCheckError();
	cudaFree(d_accel_sum);
}

__global__ void computePairwiseAccel(vector3* d_values, vector3** d_accels, vector3* d_hPos, double* d_mass) {
        //printf("computePairwiseAccel call.\n");
	int i=blockIdx.x*blockDim.x+threadIdx.x;
        int j=blockIdx.y*blockDim.y+threadIdx.y;
	if (i>=NUMENTITIES || j>=NUMENTITIES) return;
        int k;
        if (i==j) {
                //FILL_VECTOR(d_accels[i][j],0,0,0);
		d_accels[i][j][0]=d_accels[i][j][1]=d_accels[i][j][2]=0.0;
        }
        else{
                vector3 distance;
                for (k=0;k<3;k++) distance[k]=d_hPos[i][k]-d_hPos[j][k];
                double magnitude_sq=distance[0]*distance[0]+distance[1]*distance[1]+distance[2]*distance[2];
                double magnitude=sqrt(magnitude_sq);
                double accelmag=-1*GRAV_CONSTANT*d_mass[j]/magnitude_sq;
		d_accels[i][j][0]=accelmag*distance[0]/magnitude;
		d_accels[i][j][1]=accelmag*distance[1]/magnitude;
		d_accels[i][j][2]=accelmag*distance[2]/magnitude;

                //FILL_VECTOR(d_accels[i][j],accelmag*distance[0]/magnitude,accelmag*distance[1]/magnitude,accelmag*distance[2]/magnitude);
        }
}

__global__ void accelSum(vector3 *d_accel_sum, vector3** d_accels) {
	//printf("accelSum call.\n");
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if (i>=NUMENTITIES) return;
	int j;
	int k;
	for (j=0;j<NUMENTITIES;j++){
		for (k=0;k<3;k++){
			d_accel_sum[i][k]+=d_accels[i][j][k];
		}
	}
}

__global__ void updateVelPos(vector3 *d_accel_sum, vector3* d_hPos, vector3* d_hVel) {
	//printf("updateVelPos call.\n");
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	int k;
	if (i>=NUMENTITIES) return;
	for (k=0;k<3;k++){
		d_hVel[i][k]+=d_accel_sum[i][k]*INTERVAL;
		d_hPos[i][k]+=d_hVel[i][k]*INTERVAL;
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
	vector3* values=(vector3*)malloc(sizeof(vector3)*NUMENTITIES*NUMENTITIES);
	vector3** accels=(vector3**)malloc(sizeof(vector3*)*NUMENTITIES);
	//printf("Test print 1.\n");
	for (i=0;i<NUMENTITIES;i++)
		accels[i]=&values[i*NUMENTITIES];
	//printf("Test print 2.\n");
	//cudaMemcpy(d_accels,accels,sizeof(vector3)*NUMENTITIES, cudaMemcpyHostToDevice);
	//printf("Test print 3.\n");
	//Kernel variables
	dim3 threadsPerBlock(16,16);
	int blocksPerDim=(NUMENTITIES+16-1)/16;
	dim3 numBlocks(blocksPerDim,blocksPerDim);
	//printf("%d %d %d %d %d\n",threadsPerBlock.x,threadsPerBlock.y,numBlocks.x,numBlocks.y,blocksPerDim);
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
	//vector3 accel_sum[3]={0,0,0}; //Declare cudamalloc
	accelSum<<<blocksPerDim,16>>>((vector3*)d_accel_sum, d_accels);
	//cudaCheckError();
	updateVelPos<<<blocksPerDim,16>>>((vector3*)d_accel_sum, d_hPos, d_hVel);
	//cudaCheckError();
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
	free(accels);
	free(values);
#ifdef DEBUG
	cudaMemcpy(hVel, d_hVel, NUMENTITIES, cudaMemcpyDeviceToHost);
//	cudaCheckError();
        cudaMemcpy(hPos, d_hPos, NUMENTITIES, cudaMemcpyDeviceToHost);
//	cudaCheckError();
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
}*/
