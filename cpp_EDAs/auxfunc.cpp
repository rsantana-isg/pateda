#include <time.h> 
#include <math.h> 
#include <stdlib.h> 
#include <stdio.h> 
#include "auxfunc.h" 
 
double myrand() 
{ double aux1,aux2; 
   aux1 = RAND_MAX; 
   aux2 = rand();         /* Valores entre 1 y Nstates */ 
   aux1 = aux2/aux1; 
   return aux1; 
} 
 
int randomint(int max) 
{  
  int aux; 
 
  do 
     aux = myrand()*max; 
  while (aux==max && max>1); 
  return aux; 
} 
 
void BinConvert(int number, int length, unsigned vect[] ) 
{                   /* Esta funcion convierte un numero entero number en binario vect */ 
  int i,j,todiv;    /* El numero entero tiene extension length */ 
  div_t div_result; 
   
  for(i=length;i>0;i--) 
  { 
    todiv = 1; 
	for(j=1;j<i;j++) todiv*=2; 
	div_result = div(number, todiv); 
	vect[length-i] = div_result.quot; // Se le quita un 1 a length-i 
	number = div_result.rem; 
  } 
   
    
}    
 

void BinConvert(int number, int length, int vect[] ) 
{                   /* Esta funcion convierte un numero entero number en binario vect */ 
  int i,j,todiv;    /* El numero entero tiene extension length */ 
  div_t div_result; 
   
  for(i=length;i>0;i--) 
  { 
    todiv = 1; 
	for(j=1;j<i;j++) todiv*=2; 
	div_result = div(number, todiv); 
	vect[length-i] = div_result.quot; // Se le quita un 1 a length-i 
	number = div_result.rem; 
  } 
   
    
}    

/*
int ConvertNum(int length, int base, int* vect ) 
{                   
  int i,todiv,number;   
   
  number = 0;
  todiv =1;
  for(i=1;i<=length;i++) 
  { 
      number += todiv*vect[length-i];
      todiv*=base;
  } 
   
return number;    
}   

*/

long int ConvertNum(int length, int base, int* vect ) 
{                   
  long int i,todiv,number;   
   
  number = 0;
  todiv =1;
  for(i=1;i<=length;i++) 
  { 
      number += todiv*vect[length-i];
      todiv*=base;
  } 
   
return number;    
}   

long int ConvertNum(int length, int base, unsigned int* vect ) 
{                   
  long int i,todiv,number;   
   
  number = 0;
  todiv =1;
  for(i=1;i<=length;i++) 
  { 
      number += todiv*vect[length-i];
      todiv*=base;
  } 
   
return number;    
} 

void NumConvert(int number, int length, int base, int* vect ) 
{                   /* Esta funcion convierte un numero entero number en base vect */ 
  int i,j,todiv;    /* El numero entero tiene extension length */ 
  div_t div_result; 
   
  for(i=length;i>0;i--) 
  { 
    todiv = 1; 
	for(j=1;j<i;j++) todiv*=base; 
	div_result = div(number, todiv); 
        //cout<<j<<" "<<div_result.quot<<endl; 
	vect[length-i] = int(div_result.quot); 
	number = div_result.rem; 
  } 
  //for(int l=0; l<length; l++) cout<<vect[l]<<" ";
return;    
}    
 

void NumConvert(int number, int length, int base, unsigned int* vect ) 
{                   /* Esta funcion convierte un numero entero number en base vect */ 
  int i,j,todiv;    /* El numero entero tiene extension length */ 
  div_t div_result; 
   
  for(i=length;i>0;i--) 
  { 
    todiv = 1; 
	for(j=1;j<i;j++) todiv*=base; 
	div_result = div(number, todiv); 
        //cout<<j<<" "<<div_result.quot<<endl; 
	vect[length-i] = (unsigned int) (div_result.quot); 
	number = div_result.rem; 
  } 
  //for(int l=0; l<length; l++) cout<<vect[l]<<" ";
return;    
}    
 

void SetIndexOneVar(int* AllIndex,Popul* pop, int var) 
{ 
	int i; 
	// Here it is assummed that there are two trees (ntrees=2) 
		for(i=0; i<pop->psize;i++)  AllIndex[i] = pop->P[i][var];	 
}	 
 
void SetIndexIsochain(int* AllIndex,Popul* pop) 
{ 
	int i,auxval; 
	auxval = pop->psize; 
	// Here it is assummed that there are four trees (ntrees=4) 
	// to solve a function whose maximum is triggered by 
	// its three last variables 
		for(i=0; i<pop->psize;i++)  AllIndex[i] = (pop->P[i][auxval] + pop->P[i][auxval-1] + pop->P[i][auxval-2]);	 
}	 
 
void SetIndexNormal(int* AllIndex,Popul* pop) 
{ 
  int i; 
  for(i=0; i<pop->psize;i++) AllIndex[i] = 1; 
} 
 
void SetIndex(int VisibVariable, int* AllIndex,Popul* pop, int var) 
{ 
  // AllIndex = new int[pop->psize]; 
	switch(VisibVariable) 
    { 
     case 0: SetIndexNormal(AllIndex,pop); break; 
     case 1: SetIndexOneVar(AllIndex,pop,var);break;  
     case 2: SetIndexIsochain(AllIndex, pop); break; 
  } 
} 
 
void i_user_defined(int a) 
{ 
	 
} 
void e_user_defined(int* buff, double fv, int b) 
{ 
} 
 
int s_user_defined(double a, double b, int y,  Popul* p) 
{ 
	return 0; 
} 
 
double user_defined_function(int* a, int b ) 
{ 
	return 0; 
} 
 
void InitPerm(int dim,int* auxsample) 
{ 
	int i; 
	  
	 for(i=0;i<dim;i++) auxsample[i]=i; 
} 
 
void RandomPerm(int dim,int perms, int* auxsample) 
  { 
     int i;	  
	 int auxpos1,auxpos2,aux; 
	  
	 for(i=0;i<perms;i++) 
		 { 
		     auxpos1=randomint(dim); 
		     auxpos2=randomint(dim); 
		     	  
			 aux=auxsample[auxpos1]; 
			 auxsample[auxpos1]=auxsample[auxpos2]; 
			 auxsample[auxpos2]=aux; 
		  } 
 
} 

void RandomPerm(int dim,int perms, unsigned long* auxsample) 
  { 
     int i;	  
	 int auxpos1,auxpos2,aux; 
	  
	 for(i=0;i<perms;i++) 
		 { 
		     auxpos1=randomint(dim); 
		     auxpos2=randomint(dim); 
		     	  
			 aux=auxsample[auxpos1]; 
			 auxsample[auxpos1]=auxsample[auxpos2]; 
			 auxsample[auxpos2]=aux; 
		  } 
 
} 

double Calculate_Best_Prior(int N, int NumberVars, int dim, double penalty) 
{ 
  double prior; 
 
   prior  = (penalty*N)/(NumberVars*pow(2,dim-1)); 
   return prior;  
 
} 

double FindChiVal(double thresh,int which, double df) //Type of chi square test and degrees of freedom  
{   
  double q,x,bound;  
  int status;  
  q= 1-thresh;  
  cdfchi(&which,&thresh,&q,&x,&df,&status,&bound);  
  return x;  
}  




void swap(int i, int j,unsigned int* nextperm) {

	int temp;
        //cout<<i<<" "<<j<<" "<<nextperm[i]<<" "<<nextperm[j]<<endl;
	temp = nextperm[i];
	nextperm[i] = nextperm[j];
	nextperm[j] = temp;
}

void Next(int n, unsigned int* nextperm) {

	int k,j,r,s;

	k = n-2; 
	while (k>-1 &&  (nextperm[k] > nextperm[k+1]))
         {
           k--;      
	 }
	if (k == -1) return;
	else {
		j = n-1;
		while (j>-1 && (nextperm[k] > nextperm[j])) j--;
               	swap(j,k,nextperm);
		r = n-1; s = k+1;
		while (s<n && r>-1 && r>s) {
			swap(r,s,nextperm);
			r--; s++;
		}
	
	}
	return;
}

void nextperm (int n, unsigned int* initperm, unsigned int* nextperm) {

	int i;

	if (n<=0) exit(1); 

	for (i=0; i<n; ++i) 
          {
           nextperm[i] = initperm[i];
           
          }
       
	Next(n,nextperm);

}


// Implements stochastic universal sampling, generating nsamples from the 
// vector of probabilities probvals
void SUS(int nprobvals, double* probvals, int nsamples, int* samples)
 {
   int i,j,a,nvals;
   a = 0;
  
   for(i=0;i<nprobvals;i++)
     {
       nvals = (int) nsamples*probvals[i];
       //cout<<i<<" "<<nprobvals<<" "<<nsamples<<" "<<probvals[i]<<" "<<nvals<<endl;
       j = 0;
       while(j<nvals && a<nsamples) 
        {
	  samples[a] = i;
          j++;
          a++;
        }
     }
    while (a<nsamples) samples[a++] = randomint(nprobvals);
}
