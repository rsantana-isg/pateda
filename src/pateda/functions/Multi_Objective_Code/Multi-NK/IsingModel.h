#include <math.h>  
#include <stdio.h> 
#include <string.h> 
#include <stdlib.h> 
#include <iostream> 
#include <fstream> 
#include "auxfunc.h"

class Ising { 
public: 
int **lattice; 
 double **Inter;
int NumberVars; 
int dim,width,neigh; 

double groundstate; 
 
Ising(int,int,int,int); 
Ising(char*);  //twodim model 
~Ising(); 
void Createlattice();
void InitLattice();
void SaveInstance(char*);
void SaveInstanceforChecking(char*);
double evalfunc(unsigned int*);  
double evalfunc(int*); 
void RandomSpins();
}; 
