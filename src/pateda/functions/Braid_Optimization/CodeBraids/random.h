#ifndef RANDOM_H
#define RANDOM_H

#include <stdlib.h>

typedef struct
{
	void (*seed)(int seed);
	float (*next)();
	int (*nextInt)(int max);
} Random;

void Random_Shuffle(void* base, size_t nel, size_t size, Random r);

#endif
