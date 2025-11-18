#ifndef __CNF_H 
#define __CNF_H 
 
#include <iostream> 
#include <fstream>
#include <stdio.h> 
#include <stdlib.h>

 
#define LINE_LENGTH 2000 
// This class keeps the clauses read from a file  
// in .cnf form 
// It can also evaluate the satisfiability of a 
// a given class 
 
class CNF { 
public: 
int **clauses; 
int cantclauses; 
int NumberVars; 
int dimclause; 
int **adjmatrix; 
double Tot_Weight; 
double *clause_weights; 
int Satisfied; 
int MinUnit; 
int MaxUnit; 
 unsigned* nsastindex;
 
CNF(int,int,int); 
CNF(FILE*,int); 
~CNF(); 
void Addclause(int,int*); 
void FillMatrix(); 
double SatClauses(unsigned* ); 
double SatClausesClean(unsigned* );
double SatClausesChange(unsigned* );
double SatClauses(int* ); 
void SetVariables(int,int,int); 
void AdaptWeights(double,unsigned*); 
void UpdateWeights(double*); 
void EqualUpdateWeights(double*); 
int FindConstraints(int); 
void AssignUnitations(); 
 
 
}; 
 
// This class generate an instance of 3-SAT  
// problem, variables are divided in clusters 
// and clauses are formed using variables in and 
// out the clusters 
// The instance can be saved in .cnf form 
 
class CNF_Generator : public CNF { 
public: 
 
int cantclusters; 
int Vars_per_Cluster; 
int Clauses_per_cluster; 
int Clauses_between_clusters; 
int card_sol; 
int internal_clauses; 
int **Clusters; 
unsigned* solution; 
 
 
CNF_Generator(int,int,int,int,int,int); 
~CNF_Generator(); 
 
void Generate_Clusters(); 
void Generate_Solution(); 
void Generate_OneExternal_Clause(int*); 
void Generate_OneInternal_Clause(int*,int); 
int All_Clusters_Different(int*);  
void Generate_External_Clauses(); 
void Generate_Internal_Clauses(); 
void Generate_Clauses(); 
int EvalClause(int*); 
void Create(); 
void SaveInstance(char*);  
}; 
 
class ISING { 
public:  
  std::fstream FILE;
int nrows;
int ncolumns;    
int** neighbors;
float** weights;
int NumberVars; 
double Max;
 

ISING(char*,int); 
~ISING(); 

 
 
}; 

#endif



