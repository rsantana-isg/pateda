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
 int* EvalAuxVar_Left;
 int* EvalAuxVar_Right;
 int* P_Moves;
 int* Is_Promising;
 unsigned int* init_x;
 unsigned int* tabu_moves;

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
double greedy_evalfunc(unsigned int*); 
double random_evalfunc(unsigned int*,int); 
double HC_evalfunc(unsigned int*); 
 double Tabu_evalfunc(unsigned int* x,int,int);   
 double SA_evalfunc(unsigned int*,int); 
 double Random_SA_evalfunc(unsigned int*); 
 double Best_SA_evalfunc(unsigned int*); 
void RandomSpins();
}; 
