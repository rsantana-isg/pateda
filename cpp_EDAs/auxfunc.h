#ifndef __AUXFUNC_H 
#define __AUXFUNC_H 

#include <iostream> 
#include <fstream>  
#include "Popul.h" 
#include "cdflib.h"


 void BinConvert(int , int , unsigned int*); 
 void BinConvert(int , int ,  int*); 
long int ConvertNum(int,int,int*);
long int ConvertNum(int,int,unsigned int*);
 void NumConvert(int, int , int , int*); 
 void NumConvert(int, int , int , unsigned int*); 
 double myrand(); 
 int randomint(int); 
 void SetIndexOneVar(int*,Popul*,int); 
 void SetIndexIsochain(int*,Popul*); 
 void SetIndexNormal(int*,Popul*); 
 void SetIndex(int,int*,Popul*,int); 
 void InitPerm(int,int*); 
 void RandomPerm(int,int,int*);
  void RandomPerm(int,int,unsigned long*); 
 double FindChiVal(double,int,double);  
 void nextperm (int, unsigned int*, unsigned int*);
 void Next(int,unsigned int*);
 void swap(int,int,unsigned int*);
 void SUS(int, double*, int,int*);
 #endif  
