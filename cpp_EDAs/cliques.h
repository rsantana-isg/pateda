#ifndef __CLIQUE__ 
#define __CLIQUE__ 
 
 
#include "marg.h" 
#include "runif.h" 
 
//---------------------------------------- 
 
class c{ 
public: 
  int *vars; 
  int size; 
//....................... 
  int _nextEdge; 
  int _nextSubClique; 
  int _nextNode; 
  int NEdges; 
public: 
  c(); 
  c(int v1); 
  c(int s, int* vararray); 
  c(int v1, int v2); 
  c(int v1, int v2, int v3); 
  c(int v1, int v2, int v3, int v4); 
  c(int v1, int v2, int v3, int v4, int v5); 
  c(int v1, int v2, int v3, int v4, int v5, int v6); 
  c(int v1, int v2, int v3, int v4, int v5, int v6, int v7); 
//................................. 
  c( c &a ); 
//................................. 
  ~c(); 
  }; 
 
//============================================ 
/* 
const int sizeBuffcl=1000; 
class cl : public DLList<int>{ 
c **cliqs; 
int punt; 
public: 
  cl(); 
  ~cl(); 
  void insert(c a); 
  void print(); 
  c *clique(int); 
  void order(cl &); 
}; 
*/ 
#endif 
