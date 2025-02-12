CC = gcc
CFLAGS = -Wall -Wextra
AES_SRC = aes.c
DECRYPT_SRC = decrypt.c $(AES_SRC)
ENCRYPT_SRC = encrypt.c $(AES_SRC)
DECRYPT_BIN = decrypt
ENCRYPT_BIN = encrypt

# Default target
all: $(ENCRYPT_BIN) $(DECRYPT_BIN)

# Encryption target
$(ENCRYPT_BIN): $(ENCRYPT_SRC)
	$(CC) $(CFLAGS) -o $(ENCRYPT_BIN) $(ENCRYPT_SRC)

# Decryption target
$(DECRYPT_BIN): $(DECRYPT_SRC)
	$(CC) $(CFLAGS) -o $(DECRYPT_BIN) $(DECRYPT_SRC)

# Clean up build files
clean:
	rm -f $(ENCRYPT_BIN) $(DECRYPT_BIN)

.PHONY: all clean
