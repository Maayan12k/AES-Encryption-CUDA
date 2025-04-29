# CUDA File Encryptor

This project began as an effort to encrypt large video files (10–15 GB) in parallel. As the implementation progressed, the focus shifted toward understanding the AES encryption algorithm in depth and achieving measurable performance improvements over traditional CPU-based encryption.

This repository contains a CUDA-based AES encryption system designed to efficiently process files in parallel. The encryption kernel operates on 128-bit blocks, utilizing the GPU's parallel processing capabilities to accelerate encryption.

## Performance

This implementation demonstrates a **~6000% speedup** over sequential CPU-based AES encryption for input files **larger than 5 MB**. For smaller files, GPU kernel launch overhead limits the performance benefit, but for large file sizes, GPU parallelism provides a significant throughput advantage.

## Features

- CUDA-accelerated AES encryption for efficient file processing  
- Block-based encryption (128-bit block size) suitable for large files  
- Padding support to ensure consistent encryption for files of arbitrary size  

## Requirements

- NVIDIA CUDA Toolkit  
- C++ compiler with CUDA support (e.g., `nvcc`)  

## Future Work

- Add support for AES-192 and AES-256 key sizes  
- Implement a mechanism similar to AES-CBC (Cipher Block Chaining) for enhanced security  
- Optimize memory usage through local/shared memory  
- Investigate additional CUDA-specific optimizations 
