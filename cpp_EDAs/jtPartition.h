#ifndef __PARTITION__ 
 
#define __PARTITION__ 
 
 
#include "JtreeTable.h" 

/* La clase JtPartition contiene una lista en la cual se almacenan las 
   k mejores configuraciones obtenidas por el algoritmo de Nilssen */ 
 
class JtPartition{ 
 
public: 
  
 
 
int CantConf;      // Cant de Configuraciones existentes en la lista 
 
int MaxKConf;      // Max. de Configuraciones (K) 
 
int NumCliq;       // Cant. de Cliques  
 
int NumVars;       // Cant. de Variables 
  
int *ordering;     // Indice representando el orden de  los elementos en la lista 
 
int BestTree;      // Mejor Conf( Arbol ) actual. 
 
int first, last, firstindex;  // Se utilizan para el trabajo con la lista ( ver funcion Cycle ) 
 
 
 
JtreeTable **KbestTrees;  // Conj. de configuraciones ( Arboles) 
   
 JtPartition(int MaxConf, int CantVar, int CantCliq); 
 
 inline JtreeTable* GetBestTree(){return KbestTrees[firstindex];}; 
 
~JtPartition(); 
 
 void Add(JtreeTable* tree, float val);  // Adiciona una conf. a la lista 
 
 int FindFather(int from, int son) ;    // Halla la conf. anterior a son en la lista ordenada 
 
 void SetPop(int Elit, Popul* aPop, int howmany);   // Llena una poblacion a partir de la lista 
 
 void Cycle();                           // Calcula las k configuraciones 
 
 inline int  GetCantConf() {return CantConf;}; //Devuelve el numero de configuraciones actual 
}; 
 
#endif 
 
 
 
 
 
 
 
 
