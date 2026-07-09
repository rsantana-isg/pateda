#include <math.h>  
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream> 
#include <fstream> 
#include "auxfunc.h"

class NK { 
public: 
int **lattice; 
double **Inter;
int* vector_r;
int* aux_perm;

int n; 
int k; 
int dim;
 
NK(int,int); 
NK(char*);  
~NK(); 
void Createlattice();
void SaveInstance(char*);
double evalfunc(unsigned int*);  
double evalfunc(int*); 
void RandomInstance();
}; 
