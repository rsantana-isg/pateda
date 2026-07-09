#include "NKModel.h"
#include "auxfunc.h"

NK::NK(int nvars, int kk) 
{
    n = nvars;
    k = kk;
    dim = pow(2,k+1);
    Createlattice();   
}


NK::NK(char* filename)   
{  
	FILE *stream;  
	int i,j;  
  
        stream = fopen(filename, "r+" );  
	fscanf(stream,"%d \n",&n);     
	fscanf(stream,"%d \n",&k);  

        Createlattice();   

        for (i=0; i<n; i++)  	 
          for (j=0; j<k+1; j++) 
            fscanf(stream,"%d ", &lattice[i][j]);	  
	
        for (i=0; i<n; i++)  	 
          for (j=0; j<dim; j++) 
	     fscanf(stream,"%f ", &Inter[i][j]);		 
	
	  
	fclose(stream);  
    
}   

void NK::Createlattice()
{
 int i;   
 lattice = new int*[n];
 Inter =  new double*[n];
 vector_r = new int[k+1];
 aux_perm = new int[n];
 
 for(i=0;i<n;i++) 
  {
   aux_perm[i] = i;
   lattice[i] = new int[k+1]; 
   Inter[i] = new double[dim];
  }
}

void NK::RandomInstance()
{
    int i,j,count;
    for (i=0; i<n; i++)  
     {
       lattice[i][0] = i;  	 
       RandomPerm(n,2*n,aux_perm);
       count = 0;
       for (j=1; j<k+1; j++) 
         {          
	   while(aux_perm[count] == i) count++;
           lattice[i][j] = aux_perm[count];
           count++;	   
	 }	
       for (j=0; j<dim; j++) Inter[i][j] = myrand();
	 
    }  
}


void NK::SaveInstance(char* filename)   
{  
	FILE *stream;  
	int i,j;  
  
    stream = fopen(filename, "w+" );  
	fprintf(stream,"%d \n",n);     
	fprintf(stream,"%d \n",k);  
    for (i=0; i<n; i++)  
      for (j=0; j<k+1; j++) 	    
        fprintf(stream,"%d\n ",lattice[i][j]);
    for (i=0; i<n; i++)  
      for (j=0; j<dim; j++) 	    
        fprintf(stream,"%f\n ",Inter[i][j]);
 	  
    fclose(stream);  
}   



double NK::evalfunc(unsigned int* x)   
{  
int i,j,pos;  
double sum;
 sum = 0;  
 for (i=0; i<n; i++)  
   { 
    for (j=0; j<k+1; j++) vector_r[j] = x[lattice[i][j]];
    pos = ConvertNum(k+1,2,vector_r);       
    sum = sum + Inter[i][pos];
   }
 return sum/n;
}   

double NK::evalfunc(int* x)   
{  
int i,j,pos;  
double sum;
 sum = 0;  
 for (i=0; i<n; i++)  
   { 
    for (j=0; j<k+1; j++) vector_r[j] = x[lattice[i][j]];
    pos = ConvertNum(k+1,2,vector_r);       
    sum = sum + Inter[i][pos];
   }
 return sum/n;
}   


NK::~NK()
{
 int i;
 for(i=0;i<n;i++) 
  {
   delete[] lattice[i]; 
   delete[] Inter[i];
  }
    delete[] lattice;
    delete[] Inter;
    delete[] vector_r;
    delete[] aux_perm;
}
