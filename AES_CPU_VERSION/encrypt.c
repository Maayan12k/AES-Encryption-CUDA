#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/stat.h>
#include "aes.h"

#define BLOCK_SIZE 16

void generate_random_iv(uint8_t *iv, size_t length)
{
    for (size_t i = 0; i < length; i++)
    {
        iv[i] = rand() % 256;
    }
}

void pad_data(uint8_t *data, size_t file_size, size_t padded_size)
{
    uint8_t padding_value = padded_size - file_size;
    memset(data + file_size, padding_value, padding_value);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <input file> <output file>\n", argv[0]);
        return 1;
    }

    // Get file size using stat()
    struct stat file_stat;
    if (stat(argv[1], &file_stat) != 0)
    {
        perror("stat");
        return 1;
    }
    size_t file_size = file_stat.st_size;
    size_t padded_size = (file_size % BLOCK_SIZE == 0) ? file_size : ((file_size / BLOCK_SIZE + 1) * BLOCK_SIZE);

    // Allocate memory for padded input
    uint8_t *buffer = (uint8_t *)malloc(padded_size);
    if (!buffer)
    {
        perror("Memory allocation failed");
        return 1;
    }

    // Read file into buffer
    FILE *input_file = fopen(argv[1], "rb");
    if (!input_file)
    {
        perror("Failed to open input file");
        free(buffer);
        return 1;
    }
    fread(buffer, 1, file_size, input_file);
    fclose(input_file);

    // Pad data to be a multiple of 16 bytes using PKCS#7
    pad_data(buffer, file_size, padded_size);

    // Generate a random IV
    uint8_t iv[BLOCK_SIZE];
    srand(time(NULL));
    generate_random_iv(iv, BLOCK_SIZE);

    printf("Random IV: ");
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        printf("%02x ", iv[i]);
    }
    printf("\n");

    // AES key
    uint8_t key[BLOCK_SIZE] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

    // Initialize AES context and encrypt
    struct AES_ctx ctx;
    AES_init_ctx_iv(&ctx, key, iv);
    printf("Encrypting...\n");
    clock_t start, end;
    double elapsed_time;
    start = clock();
    AES_CBC_encrypt_buffer(&ctx, buffer, padded_size);
    end = clock();
    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Encryption time: %.6f seconds\n", elapsed_time);

    // Write IV and encrypted data to output file
    FILE *output_file = fopen(argv[2], "wb");
    if (!output_file)
    {
        perror("Failed to open output file");
        free(buffer);
        return 1;
    }
    fwrite(iv, 1, BLOCK_SIZE, output_file);
    fwrite(buffer, 1, padded_size, output_file);
    fclose(output_file);

    printf("Encryption successful! Encrypted file saved as: %s\n", argv[2]);

    free(buffer);
    return 0;
}