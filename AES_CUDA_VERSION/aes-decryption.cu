#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <sys/time.h>
#include <cmath>
#include <math.h>
#include <sys/stat.h>
#include <cstdio>

/**
    srun -A bchn-delta-gpu --time=00:20:00 --nodes=1 --tasks-per-node=16 --partition=gpuA100x4,gpuA40x4 --gpus=1 --mem=16g --pty /bin/bash
*/

#define BLOCK_SIZE 16
#define Nk 4  // The number of 32 bit words in a key.
#define Nr 10 // The number of rounds in AES Cipher.
#define getInvSBoxValue(num) (inv_sbox[(num)])
static const uint8_t Rcon[11] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b, 0x36};

static const uint8_t inv_sbox[256] = {
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d};

__constant__ uint8_t constant_inv_sbox[256];

/******************************************************************************************************* */
/* Helper Functions*/
/* START */

#define CHECK(call)                                                                  \
    {                                                                                \
        const cudaError_t cuda_ret = call;                                           \
        if (cuda_ret != cudaSuccess)                                                 \
        {                                                                            \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                            \
            printf("code: %d, reason:%s\n", cuda_ret, cudaGetErrorString(cuda_ret)); \
            exit(-1);                                                                \
        }                                                                            \
    }

double myCPUTimer()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec / 1.0e6);
}

// This function produces Nb(Nk+1) round keys. The round keys are used in each round to decrypt the states.
static void KeyExpansion(uint8_t *RoundKey, const uint8_t *Key)
{
    unsigned i, j, k;
    uint8_t tempa[4]; // Used for the column/row operations

    // The first round key is the key itself.
    for (i = 0; i < Nk; ++i)
    {
        RoundKey[(i * 4) + 0] = Key[(i * 4) + 0];
        RoundKey[(i * 4) + 1] = Key[(i * 4) + 1];
        RoundKey[(i * 4) + 2] = Key[(i * 4) + 2];
        RoundKey[(i * 4) + 3] = Key[(i * 4) + 3];
    }

    // All other round keys are found from the previous round keys.
    for (i = Nk; i < Nk * (Nr + 1); ++i)
    {

        k = (i - 1) * 4;
        tempa[0] = RoundKey[k + 0];
        tempa[1] = RoundKey[k + 1];
        tempa[2] = RoundKey[k + 2];
        tempa[3] = RoundKey[k + 3];

        if (i % Nk == 0)
        {
            // Rotate word
            const uint8_t u8tmp = tempa[0];
            tempa[0] = tempa[1];
            tempa[1] = tempa[2];
            tempa[2] = tempa[3];
            tempa[3] = u8tmp;

            // substitue bytes in word
            tempa[0] = getInvSBoxValue(tempa[0]);
            tempa[1] = getInvSBoxValue(tempa[1]);
            tempa[2] = getInvSBoxValue(tempa[2]);
            tempa[3] = getInvSBoxValue(tempa[3]);

            tempa[0] = tempa[0] ^ Rcon[i / Nk];
        }
        j = i * 4;
        k = (i - Nk) * 4;
        RoundKey[j + 0] = RoundKey[k + 0] ^ tempa[0];
        RoundKey[j + 1] = RoundKey[k + 1] ^ tempa[1];
        RoundKey[j + 2] = RoundKey[k + 2] ^ tempa[2];
        RoundKey[j + 3] = RoundKey[k + 3] ^ tempa[3];
    }
}

/* END */
/* Helper Functions*/
/******************************************************************************************************* */

/******************************************************************************************************* */
/* AES Functions*/
/* START */

__device__ void invSubBytes(uint8_t *index)
{

    for (int i = 0; i < 16; i++)
    {
        uint8_t byte = index[i];
        uint8_t first4Bits = (byte & 0xF0) >> 4;
        uint8_t last4Bits = byte & 0x0F;
        int inv_sbox_index = (first4Bits * 16) + last4Bits;

        index[i] = constant_inv_sbox[inv_sbox_index];
    }
}

__device__ void invShiftRows(uint8_t *index)
{
    uint8_t temp;

    temp = index[13];
    for (int i = 13; i >= 5; i -= 4)
        index[i] = index[i - 4];
    index[1] = temp;

    temp = index[2];
    index[2] = index[10];
    index[10] = temp;
    temp = index[6];
    index[6] = index[14];
    index[14] = temp;

    temp = index[3];
    for (int i = 3; i <= 11; i += 4)
        index[i] = index[i + 4];
    index[15] = temp;
}

__device__ uint8_t gmul(uint8_t a, uint8_t b)
{ // https://crypto.stackexchange.com/questions/71204/how-are-these-aes-mixcolumn-multiplication-tables-calculated
    uint8_t p = 0;
    while (b > 0)
    {
        p ^= a & -(b & 1);
        a = (a << 1) ^ (0x11b & -(a >> 7));
        b >>= 1;
    }
    return p;
}

__device__ void invMixColumns(uint8_t *index)
{
    uint8_t temp[16];

#pragma unroll
    for (int col = 0; col < 4; col++)
    {
        int i = col * 4;

        uint8_t s0 = index[i];
        uint8_t s1 = index[i + 1];
        uint8_t s2 = index[i + 2];
        uint8_t s3 = index[i + 3];

        temp[i + 0] = gmul(s0, 0x0e) ^ gmul(s1, 0x0b) ^ gmul(s2, 0x0d) ^ gmul(s3, 0x09);
        temp[i + 1] = gmul(s0, 0x09) ^ gmul(s1, 0x0e) ^ gmul(s2, 0x0b) ^ gmul(s3, 0x0d);
        temp[i + 2] = gmul(s0, 0x0d) ^ gmul(s1, 0x09) ^ gmul(s2, 0x0e) ^ gmul(s3, 0x0b);
        temp[i + 3] = gmul(s0, 0x0b) ^ gmul(s1, 0x0d) ^ gmul(s2, 0x09) ^ gmul(s3, 0x0e);
    }

#pragma unroll
    for (int i = 0; i < 16; i++)
    {
        index[i] = temp[i];
    }
}

__device__ void addRoundRey()
{
}

__global__ void decryptAes(uint8_t *in, uint8_t *out, unsigned int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = idx * 16; // Each thread processes 16 bytes

    if (offset >= n)
        return;

    // invSubBytes(in + offset);
    // for(int i = 0; i < 8; i++) //calling mix columns 4 times, returns the matrix to its original state, for testing purposes
    //     invMixColumns(in + offset);

    // Copy 16 bytes from input to output
    for (int i = 0; i < 16; i++)
    {
        if (offset + i < n)
        {
            out[offset + i] = in[offset + i];
        }
    }
}

/* AES Functions*/
/* END */
/******************************************************************************************************* */

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }

    struct stat file_stat;
    if (stat(argv[1], &file_stat) != 0)
    {
        perror("stat");
        return 1;
    }
    size_t file_size = file_stat.st_size;

    char *file_name = argv[1];
    printf("\nDecrypting file: \"%s\"\n", file_name);
    printf("\nFile Size: %d bytes\n", file_size);

    uint8_t *buffer = (uint8_t *)malloc(file_size);
    if (!buffer)
    {
        perror("Memory allocation failed");
        return 1;
    }

    FILE *input_file = fopen(argv[1], "rb");
    if (!input_file)
    {
        perror("Failed to open input file");
        free(buffer);
        return 1;
    }
    fread(buffer, 1, file_size, input_file);
    fclose(input_file);

    // for (int i = 0; i < padded_size; i++) {
    //     for (int bit = 7; bit >= 0; bit--) {
    //         printf("%d", (buffer[i] >> bit) & 1);  // Extract and print each bit
    //     }
    //     printf(" ");  // Separate bytes with a space
    // }
    // printf("\n");

    // AES key
    uint8_t key[BLOCK_SIZE] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

    printf("Decrypting...\n");
    uint8_t *inBuff, *outBuff;
    cudaMalloc((void **)&inBuff, sizeof(uint8_t) * file_size);
    cudaMalloc((void **)&outBuff, sizeof(uint8_t) * file_size);

    cudaMemcpy(inBuff, buffer, sizeof(uint8_t) * file_size, cudaMemcpyHostToDevice);

    cudaMemcpyToSymbol(constant_inv_sbox, inv_sbox, 256 * sizeof(uint8_t));

    int numThreads = file_size / 16;
    dim3 blockDim(16);
    dim3 gridDim((numThreads));

    decryptAes<<<gridDim, blockDim>>>(inBuff, outBuff, file_size);
    CHECK(cudaDeviceSynchronize());

    uint8_t *outFileBuff = (uint8_t *)calloc(file_size, sizeof(uint8_t));
    cudaMemcpy(outFileBuff, outBuff, sizeof(uint8_t) * file_size, cudaMemcpyDeviceToHost);

    FILE *output_file = fopen(argv[2], "wb");
    if (!output_file)
    {
        perror("Failed to open output file");
        free(outFileBuff);
        cudaFree(inBuff);
        cudaFree(outBuff);
        free(buffer);
        return 1;
    }

    uint8_t paddedValue = outFileBuff[file_size - 1];
    size_t original_size = file_size - paddedValue;

    fwrite(outFileBuff, sizeof(uint8_t), original_size, output_file);
    fclose(output_file);

    printf("Decryption complete. Output written to \"%s\"\n", argv[2]);

    // printf("IN  OUT \n");
    // for(int i = 0; i < 4; i++){
    //     printf("%02X  %02X\n",buffer[i], outFileBuff[i]);
    // }

    cudaFree(inBuff);
    cudaFree(outBuff);
    free(outFileBuff);
    free(buffer);
    return 0;
}
