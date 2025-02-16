# CUDA File Encryptor  

This project started as an idea to encrypt large files(10 - 15 GB) in parallel. However, as I explored the implementation, I pivoted toward understanding AES encryption in depth.  

This repository contains a CUDA-based AES encryption system designed to efficiently process files in parallel. The encryption kernel operates on 128-byte blocks, leveraging the GPU’s parallel processing power for high-speed encryption.  

## Features  
- **CUDA-accelerated AES encryption** for efficient file processing  
- **Block-based encryption** to handle large files  
- **Padded input handling** for consistency across different file sizes  

## Requirements  
- NVIDIA CUDA Toolkit  
- C++ Compiler with CUDA support  

## Future Work  
- Expand to full AES encryption  
- Improve memory optimizations  
- Explore CUDA optimizations 
