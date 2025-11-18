#ifndef __GA_H 
#define __GA_H 
#include <math.h>  
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream> 
#include <fstream> 
#include "auxfunc.h"


class GA { 
public: 

void GenUniformCXInd(Popul*,int, unsigned int*, unsigned int*);   
void GenOnePointCXInd(Popul*,int,unsigned int*,unsigned int*);   
void MutatePop(Popul*,int, int);
void GenCrossPop(int,Popul*, Popul*,int);
}; 

#endif 
