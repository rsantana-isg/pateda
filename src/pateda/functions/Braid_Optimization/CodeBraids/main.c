#include "braid.h"
#include "braidGA.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

void _seed(int s)
{
	srand(s);
}

int _nextInt(int max)
{
	return rand()%max;
}

float _next()
{
	return (float)rand()/RAND_MAX;
}

Random r = {&_seed, &_next, &_nextInt};

//#define LAMBDA 0.1
double LAMBDA = 0;
double lambdas[] = {0, 0.01, 0.05, 0.1};
int lambdaCount = 4;
double fitness(Braid b, Quaternion target)
{
	
	double distance = Quaternion_Distance(target, Braid_GetComposite(b));
	double length = Braid_GetLength(b);
	if(distance > 0.0000001)
		return (1.0-LAMBDA)/(1+distance) + LAMBDA/length;
	else
		return -INFINITY;
}

#ifdef FAKE_QUATERNION
Quaternion Target = (Quaternion){U,Z,Z,Z, Z,U,Z,Z, Z,Z,Z,U, Z,Z,U,Z};
#else
Quaternion Target = (Quaternion){0,0,0,1};
#endif

int comparer(Braid *b1, Braid *b2)
{

        //Quaternion_Print(Target);
        //Quaternion_Print(Braid_GetComposite(*b1));
        //Quaternion_Print(Braid_GetComposite(*b2));
        double distance1 = Quaternion_Distance(Target, Braid_GetComposite(*b1));
      
	double distance2 = Quaternion_Distance(Target, Braid_GetComposite(*b2));

        //printf("%f %f \n", distance1,distance2);
	return (distance1 < distance2 ? 1 : -1);
}



void main()
{

	
 #ifdef FAKE_QUATERNIONS
//Fake quaternion test code
/*
	double l;

	Quaternion test = (Quaternion){I,Z,Z,U, Z,I,Z,Z, U,Z,Z,I, Z,Z,I,Z};

	for(l=1.0; l>0; l-=0.01)
	{
                printf("%f \n", l);
		Quaternion test2 = (Quaternion){I,Z,Z,U, Z,I,Z,(Complex){l,0}, U,(Complex){0,l},Z,I, Z,Z,I,Z};
		double len = Quaternion_Distance(test, test2);
		printf("%f %f\n", l, len);
	}
		return;
*/
	// #endif
	srand(time(NULL));

       
	BraidGAEngineParams params = (BraidGAEngineParams)
	{
		.populationSize = 80,
		.maxLength = 100,
		.generations = 400,
		.mutationRate = 0.000,
		.killFraction = 0.1,
		.target = Target,
		.random = r,
		.idBraider = IdBraider_Create(0, r),
		.fitness = &fitness
	};
	int runs = 10;

	
	int i;
	for(i=0; i<lambdaCount; i++)
	{
	        
		LAMBDA = lambdas[i];

		int run;
		double totalError = 0;
		double totalError2 = 0;
		for(run=0; run<runs; run++)
		{

                       
			BraidGAEngine engine = BraidGAEngine_Create(params);
                        
	                //printf(" Lambda %f run %d \n", LAMBDA,run);
			int generation;
			for(generation=0; generation<params.generations; generation++)
			{			
				BraidGAEngine_NextGeneration(engine);
			}
                      
			BraidGAEngine_Sort(engine, &comparer);
  	               	Braid best = BraidGAEngine_InitIteration(engine);
             	      	double error = Quaternion_Distance(Target, Braid_GetComposite(best));
			double logError = log(error);
                        totalError += logError;
			totalError2 += (logError*logError);
          	        BraidGAEngine_Destroy(engine);
		}
		double avgLogError = totalError / runs;
		double avgLogError2 = totalError2 / runs;
		printf("%f %f %f %d \n", LAMBDA, avgLogError, avgLogError2, runs);
	}

 #endif
}
