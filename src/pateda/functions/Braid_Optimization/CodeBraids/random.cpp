#include "random.h"
#include <stdlib.h>

void Random_Shuffle(void* base, size_t nel, size_t size, Random r)
{
	unsigned char *byteBase = (unsigned char*)base;
	int i;
	for(i = nel-1; i>1; i--)
	{
		int j = r.nextInt(i);
		//XOR-swap each byte
		int k;
		for(k=0; k<size; k++)
		{
			byteBase[i*size+k] ^= byteBase[j*size+k];
			byteBase[j*size+k] ^= byteBase[i*size+k];
			byteBase[i*size+k] ^= byteBase[j*size+k];
		}
	}
}
