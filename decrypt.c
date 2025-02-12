#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "aes.h"

#define BLOCK_SIZE 16

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        printf("Usage: %s <encrypted file> <output file>\n", argv[0]);
        return 1;
    }

    // Get file size using stat()
    struct stat file_stat;
    if (stat(argv[1], &file_stat) != 0)
    {
        perror("Failed to get file size");
        return 1;
    }
    size_t file_size = file_stat.st_size;

    // Check if file size is valid (should be at least 16 bytes for IV)
    if (file_size < BLOCK_SIZE)
    {
        printf("Invalid encrypted file.\n");
        return 1;
    }

    // Open the encrypted file
    FILE *input_file = fopen(argv[1], "rb");
    if (!input_file)
    {
        perror("Failed to open encrypted file");
        return 1;
    }

    // Read IV
    uint8_t iv[BLOCK_SIZE];
    fread(iv, 1, BLOCK_SIZE, input_file);

    // Calculate encrypted data size
    size_t encrypted_size = file_size - BLOCK_SIZE;

    // Allocate memory for encrypted data
    uint8_t *encrypted_data = (uint8_t *)malloc(encrypted_size);
    if (!encrypted_data)
    {
        perror("Memory allocation failed");
        fclose(input_file);
        return 1;
    }

    // Read encrypted data
    fread(encrypted_data, 1, encrypted_size, input_file);
    fclose(input_file);

    // AES key (same as encryption key)
    uint8_t key[BLOCK_SIZE] = {
        0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6,
        0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c};

    // Initialize AES context and decrypt
    struct AES_ctx ctx;
    AES_init_ctx_iv(&ctx, key, iv);
    AES_CBC_decrypt_buffer(&ctx, encrypted_data, encrypted_size);

    // Remove padding (PKCS#7)
    uint8_t padding_value = encrypted_data[encrypted_size - 1];
    size_t unpadded_size = encrypted_size - padding_value;

    // Write decrypted data to output file
    FILE *output_file = fopen(argv[2], "wb");
    if (!output_file)
    {
        perror("Failed to open output file");
        free(encrypted_data);
        return 1;
    }

    fwrite(encrypted_data, 1, unpadded_size, output_file);
    fclose(output_file);

    printf("Decryption successful! Decrypted file saved as: %s\n", argv[2]);

    free(encrypted_data);
    return 0;
}