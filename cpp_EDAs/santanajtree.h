#ifndef __JTREE__ 
#define __JTREE__ 
#include "marg.h" 
#include "runif.h" 
#include "pop.h" 
#include "cliques.h" 
#include "chord.h" 
 
//---------------------------------------- 
#define __TAMANO_SUS__    100 
//---------------------------------------- 
class jtclique{ 
  int size, s_size, b_size, NoTables; 
  int *vars; 
  int *cards; 
  int *svars; 
  int *bvars; 
  marg **TableCount; 
  SUS  **GenSUS; 
  marg *overl; 
  int  *SValues; 
  int  *BValues; 
public: 
  jtclique(int *vars, int size); // not overlapped clique  
  jtclique(int *vars, int size, int *svars, int s_size); 
  ~jtclique(); 
  void clear();  // clear all the tables 
  void compute(pop &Pop); 
  void generate(int *vector); 
  void print();   
  void shortPrint(); 
  void StructPrint(); 
  int* getcards(){ return cards;}; 
  int* getvars(){ return  vars;}; 
  int* getsvars(){ return  svars;}; 
  int getsize(){ return size;}; 
  int gets_size() { return s_size;}; 
  
private: 
  int  not_in_svars(int); 
// ++++++++++++++++++++++++++++++++ 
// para experimento de Most Probable Configuration 
public: 
double ReturnProbOfConfiguration(int *vector); 
}; 
 
 
class jtree{ 
 
int *ordering; 
 
int NoCliques; 
jtclique **cliqs; 
 
int nvars; 
int __cliq; 
 
public: 
jtree(int vars, int nocliques); 
// jtree(char *file); 
~jtree(); 
void add(c C); 
void add(c C, c O); 
void add(int *vars, int size); 
void add(int *vars, int size, int *Over, int over_size); 
inline int getNoVars(){ return nvars;}; 
inline int getNoCliques(){ return NoCliques;}; 
inline jtclique* getCliq(int i){ return cliqs[i];}; 
int getindex(int i) { return ordering[i];}; 
void clear();  // clear all the tables 
void compute(pop &Pop); 
void generate(int *vector); 
void generate(pop &Pop); 
void print(); 
void shortPrint(); 
//void MakeJt( cl &O ); 
void StructPrint(); 
// ++++++++++++++++++++++++++++++++ 
// para experimento de Most Probable Configuration 
double ReturnProbOfConfiguration(int *vector); 
}; 
 
#endif 
 
 
 
 
 
 
