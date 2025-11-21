void compute();
void allocDeviceMemory();
void freeDeviceMemory();
void computePairwiseAccel(vector3* d_values, vector3** d_accels, vector3* d_hPos, float* d_mass);
