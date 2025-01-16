# galaxy-correlation-gpu_cuda
# 2-Point Angular Correlation Analysis with CUDA

Check commends on the code for implementation details

## Project Overview
A CUDA implementation for calculating 2-point angular correlation functions in cosmological datasets. This project processes galaxy position data to analyze the spatial distribution of galaxies and detect potential dark matter effects through statistical analysis.

## Features
- Parallel computation of angular separations between galaxy pairs
- Three correlation functions (DD, DR, RR) calculation
- Global memory implementation for large dataset handling
- Support for 100,000 galaxies with 720 histogram bins
- Runtime performance optimization through GPU parallelization

## Implementation Details

### Data Processing
- Input: Two sets of galaxy coordinates (real and random) in arc minutes
- Coordinate conversion from arc minutes to radians
- Three main correlation calculations:
  - DD (Data-Data): Real galaxy pairs
  - DR (Data-Random): Real-Random galaxy pairs
  - RR (Random-Random): Random galaxy pairs

### CUDA Architecture
- Block Size: 512 threads
- Grid Size: 640 blocks (optimized for V100 GPU)
- Global memory utilization for data storage
- Atomic operations for histogram updates
- Double precision floating-point calculations

### Key Components
1. **Angular Separation Calculation**
   - Device function implementation using spherical trigonometry
   - Numerical stability measures for cosine calculations
   - Conversion between coordinate systems

2. **Histogram Generation**
   - 720 bins covering 0-180 degrees (0.25° per bin)
   - Atomic operations for thread-safe updates
   - Separate histograms for DD, DR, and RR correlations

3. **Memory Management**
   - Efficient data transfer between host and device
   - Error checking for CUDA operations
   - Structured data handling with host/device arrays

## Performance Features
- Thread coalescing for memory access optimization
- Atomic operations for concurrent histogram updates
- Error checking macros for robust execution
- Timing measurements for performance analysis

## Results
- Computation of correlation statistics (ω(θ))
- Validation through histogram sum checks
- Performance benchmarking with timing data
- Scientific analysis through correlation function comparison

## Requirements
- CUDA-capable GPU (Tested on V100)
- CUDA Toolkit
- C++ Compiler with CUDA support
- Input data files in specified format

## Usage
```bash
nvcc -arch=sm_70 test_gpu5.cu -o correlation
./correlation real_data.txt random_data.txt
```

## Implementation Notes
- Uses global memory for straightforward data access
- Atomic operations ensure accurate histogram counts
- Double precision for mathematical accuracy
- Built-in error checking for CUDA operations

## Future Improvements
- Shared memory implementation for better performance
- Multi-GPU support for larger datasets
- Dynamic thread block sizing
- Additional optimization techniques
