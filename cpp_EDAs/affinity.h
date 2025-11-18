#ifndef __AFF_H   
#define __AFF_H
   
#include <stdio.h>
#include <stdlib.h>
#include <iostream> 
#include <fstream> 

/*#include <values.h>*/

#ifndef MINDOUBLE
#define MINDOUBLE 2.2250e-308
#endif
#ifndef MAXDOUBLE
#define MAXDOUBLE 1.7976e308
#endif

   
 class AffPropagation{   
	 public:   
  
   double* Matrix;
   unsigned long Abs_n;
 
   AffPropagation(unsigned long);
   ~AffPropagation(){};
   int affinity_prop(double, int, int, unsigned long, unsigned long, unsigned long**, double*, double*, unsigned long*, unsigned long*);
   int affinity_prop_clust(double, int, int, unsigned long, unsigned long, unsigned long**, double*, double*, unsigned long*, unsigned long*);
   int constr_affinity_prop(double, int, int,  unsigned long, unsigned long, unsigned long**, double*, double*, unsigned long*, int*, int, unsigned long*, unsigned long**, unsigned long*, int,double);
   unsigned long CallAffinity(double, int, int, unsigned long, unsigned long*, int, unsigned long**,int,double,unsigned long*,int*);
   unsigned long FindConnComponents(unsigned long, unsigned long*, unsigned long**, int, double);
};

#endif

