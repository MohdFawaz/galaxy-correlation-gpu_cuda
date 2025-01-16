/*
This is Moh'd Abu Quttain's solution for the problem,
This is the version code where threads atomicly add in a global memory of histogram
This code runs on the Dione GPU cluster shared by University of Turku and Abo Akademi, specifically V100 Nvidia
This code could be further optimized by using shared memory and float variables instead of double variables


Device Tesla V100-PCIE-16GB has compute capability 7.0
totalGlobalMemory = 16.95 GB
streaming multiprocessor = 80
maxThreadsPerMultiprocessor = 2048
sharedMemPerBlock = 49152 B
clockRate = 1.380 GHz
maxThreadsPerBlock = 1024
maxGridSize = 2147483647 x 65535 x 65535
maxThreadsDim = 1024 x 1024 x 64
*/

// libraries used
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <chrono>

// Constants
#define N_GALAXIES 100000
#define PI 3.14159265358979323846
#define ARCMIN_TO_RAD (PI/(180.0*60.0))
#define N_BINS 720
#define BIN_SIZE 0.25

//CUDA related constants
// num of blocks = 80 SM * 2048 / 256 (block size)
#define BLOCK_SIZE 256
#define NUM_BLOCKS 640  // For optimal occupancy on V100


// Error checking macro for code debugging in case an error occured
#define CHECK_CUDA_ERROR(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)


// Struct where all histograms are stored
struct Histograms {
    unsigned long long DD[N_BINS];
    unsigned long long DR[N_BINS];
    unsigned long long RR[N_BINS];
};


// Device function to calculate angular separation
__device__ double calculate_angular_separation(double ra1, double dec1, double ra2, double dec2) {
    // Convert to radians
    double ra1_rad = ra1 * ARCMIN_TO_RAD;
    double dec1_rad = dec1 * ARCMIN_TO_RAD;
    double ra2_rad = ra2 * ARCMIN_TO_RAD;
    double dec2_rad = dec2 * ARCMIN_TO_RAD;
    
    // Calculate dot product components
    double cos_theta = sin(dec1_rad) * sin(dec2_rad) + 
                      cos(dec1_rad) * cos(dec2_rad) * cos(ra1_rad - ra2_rad);
    
    // Numerical stability - clamp to [-1, 1]
    if (cos_theta > 1.0) cos_theta = 1.0;
    if (cos_theta < -1.0) cos_theta = -1.0;
    
    // Convert to degrees
    return acos(cos_theta) * 180.0 / PI;
}

// Device function to determine which histogram bin to manipulate
__device__ int get_bin(double angle) {
    int bin = (int)(angle / BIN_SIZE);
    return (bin >= N_BINS) ? N_BINS - 1 : bin;
}


// The Kernel for DD and RR calculations
__global__ void compute_auto_correlation(const double* ra, const double* dec, 
                                       unsigned long long* histogram, size_t n_galaxies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n_galaxies-1; i += stride) {
        for (int j = i + 1; j < n_galaxies; j++) {
            double angle = calculate_angular_separation(
                ra[i], dec[i],
                ra[j], dec[j]
            );
            int bin = get_bin(angle);
            atomicAdd(&histogram[bin], 2ULL);
        }
    }
}


// The Kernel for DR calculations
__global__ void compute_cross_correlation(const double* ra_real, const double* dec_real,
                                        const double* ra_random, const double* dec_random,
                                        unsigned long long* histogram, size_t n_galaxies) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for (int i = idx; i < n_galaxies; i += stride) {
        for (int j = 0; j < n_galaxies; j++) {
            double angle = calculate_angular_separation(
                ra_real[i], dec_real[i],
                ra_random[j], dec_random[j]
            );
            int bin = get_bin(angle);
            atomicAdd(&histogram[bin], 1ULL);
        }
    }
}


//The function to read txt files and ignore the first line where number of galaxies is stored
void read_galaxy_data(const char* filename, double* ra, double* dec) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        exit(1);
    }
    
    char buffer[256];
    fgets(buffer, sizeof(buffer), file);
    
    for (int i = 0; i < N_GALAXIES; i++) {
        if (fscanf(file, "%lf %lf", &ra[i], &dec[i]) != 2) {
            fprintf(stderr, "Error reading coordinates at line %d\n", i+2);
            exit(1);
        }
    }
    
    fclose(file);
}


//The main function of the code
int main(int argc, char** argv) {

    //time starts here to calculate the total time of both CPU and GPU work (use time in commands on dione to check the GPU timing alone)
    //This is the highest C++ resolution clock on the system(C++ 11 and later versions)
    auto start_time = std::chrono::high_resolution_clock::now();

    //To check if the 3 arguments ar passed such as this command: srun -p gpu --mem=1G -t 1:00:00 time ./output real_galaxies.txt random_galaxies.txt 
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <real_data_file> <random_data_file>\n", argv[0]);
        return 1;
    }
    
    // Allocate host memory (CPU)
    double *h_ra_real = (double*)malloc(N_GALAXIES * sizeof(double));
    double *h_dec_real = (double*)malloc(N_GALAXIES * sizeof(double));
    double *h_ra_random = (double*)malloc(N_GALAXIES * sizeof(double));
    double *h_dec_random = (double*)malloc(N_GALAXIES * sizeof(double));


    struct Histograms h_hist = {0}; //To declare and initialize histograms to 0
    
    // Read data of the 2 txt files
    read_galaxy_data(argv[1], h_ra_real, h_dec_real);
    read_galaxy_data(argv[2], h_ra_random, h_dec_random);
    
    // Allocate device memory (GPU /type V100)
    double *d_ra_real, *d_dec_real, *d_ra_random, *d_dec_random;
    unsigned long long *d_DD, *d_DR, *d_RR;
    

    CHECK_CUDA_ERROR(cudaMalloc(&d_ra_real, N_GALAXIES * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dec_real, N_GALAXIES * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_ra_random, N_GALAXIES * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_dec_random, N_GALAXIES * sizeof(double)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_DD, N_BINS * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_DR, N_BINS * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_RR, N_BINS * sizeof(unsigned long long)));
    

    // Copy data from host to device (CPU -> GPU)
    CHECK_CUDA_ERROR(cudaMemcpy(d_ra_real, h_ra_real, N_GALAXIES * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dec_real, h_dec_real, N_GALAXIES * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_ra_random, h_ra_random, N_GALAXIES * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_dec_random, h_dec_random, N_GALAXIES * sizeof(double), cudaMemcpyHostToDevice));
    

    // Initialize histograms to 0 (Global memory of the GPU)
    CHECK_CUDA_ERROR(cudaMemset(d_DD, 0, N_BINS * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMemset(d_DR, 0, N_BINS * sizeof(unsigned long long)));
    CHECK_CUDA_ERROR(cudaMemset(d_RR, 0, N_BINS * sizeof(unsigned long long)));

    // Launch kernels
    dim3 block(BLOCK_SIZE);
    dim3 grid(NUM_BLOCKS);

    compute_auto_correlation<<<grid, block>>>(d_ra_real, d_dec_real, d_DD, N_GALAXIES);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    compute_auto_correlation<<<grid, block>>>(d_ra_random, d_dec_random, d_RR, N_GALAXIES);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    compute_cross_correlation<<<grid, block>>>(d_ra_real, d_dec_real, 
                                             d_ra_random, d_dec_random,
                                             d_DR, N_GALAXIES);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Copy results back to host (from GPU to CPU)
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist.DD, d_DD, N_BINS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist.DR, d_DR, N_BINS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERROR(cudaMemcpy(h_hist.RR, d_RR, N_BINS * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    //we added 100,000 for DD and RR instead of calculating each galaxy with itself along the matrix diagonal to save computing power
    h_hist.DD[0]+=100000;
    h_hist.RR[0]+=100000;

    // Calculating and printing histogram sums
    unsigned long long dd_sum = 0, dr_sum = 0, rr_sum = 0;
    for (int i = 0; i < N_BINS; i++) {
        dd_sum += h_hist.DD[i];
        dr_sum += h_hist.DR[i];
        rr_sum += h_hist.RR[i];
    }

    printf("\nHistogram sums:\n");
    printf("DD: %llu\n", dd_sum);
    printf("DR: %llu\n", dr_sum);
    printf("RR: %llu\n", rr_sum);

    // Printing the  results
    printf("\nBin  Angle Range  DD        DR        RR        omega\n");
    printf("----------------------------------------------------\n");
    for (int i = 0; i < N_BINS; i++) {
      //  if (h_hist.DD[i] > 0 || h_hist.DR[i] > 0 || h_hist.RR[i] > 0) {           //uncomment this to view only the histogram bings with nonzero values
            double omega = 0.0;
            if (h_hist.RR[i] > 0) {
                omega = (double)(h_hist.DD[i] - 2.0 * h_hist.DR[i] + h_hist.RR[i]) / h_hist.RR[i];
            }
            printf("%3d  %4.2f-%-4.2f  %-8llu  %-8llu  %-8llu  %6.3f\n",
                   i, i*BIN_SIZE, (i+1)*BIN_SIZE,
                   h_hist.DD[i], h_hist.DR[i], h_hist.RR[i], omega);
        //}                                                                         //uncomment this to view only the histogram bings with nonzero values
    }

    // Calculate total time
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("\nTotal execution time: %.3f seconds\n", duration.count() / 1000.0);

    // cleaning up stage

    // freeing all alocated memories on the GPU
    cudaFree(d_ra_real);
    cudaFree(d_dec_real);
    cudaFree(d_ra_random);
    cudaFree(d_dec_random);
    cudaFree(d_DD);
    cudaFree(d_DR);
    cudaFree(d_RR);

    // freeing all allocated memory on the host
    free(h_ra_real);
    free(h_dec_real);
    free(h_ra_random);
    free(h_dec_random);

    return 0;
}