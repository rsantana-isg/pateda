#ifndef __AUXFUNC_H 
#define __AUXFUNC_H 

#include <iostream> 
#include <fstream>  
#include "Popul.h" 
#include "cdflib.h"


 void BinConvert(int , int , unsigned int*); 
int ConvertNum(int,int,int*);
 void NumConvert(int, int , int , int*); 
 double myrand(); 
 int randomint(int); 
 void SetIndexOneVar(int*,Popul*,int); 
 void SetIndexIsochain(int*,Popul*); 
 void SetIndexNormal(int*,Popul*); 
 void SetIndex(int,int*,Popul*,int); 
 void InitPerm(int,int*); 
 void RandomPerm(int,int,int*); 
 double FindChiVal(double,int,double);  
 void swap(double*, int, int, int*);
 void quicksort(double*, int,int,int*);
 #endif  
