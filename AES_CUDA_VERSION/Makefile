CC = nvcc
CFLAGS = -o
LIBS = 

all: encryption decryption

encryption: aes-encryption.cu
	$(CC) $(CFLAGS) encrypt aes-encryption.cu

decryption: aes-decryption.cu
	$(CC) $(CFLAGS) decrypt aes-decryption.cu

clean:
	rm -f encrypt decrypt
