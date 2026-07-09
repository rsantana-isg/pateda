#include "IsingModel.h"
#include "auxfunc.h"

Ising::Ising(int nvars, int wid, int d, int n) 
{
    NumberVars = nvars;
    width = wid;
    dim = d;
    neigh = n;    
    Createlattice();   
}


Ising::Ising(char* filename)   
{  
	FILE *stream;  
	int i,j,auxint;  
  
    stream = fopen(filename, "r+" );  
	fscanf(stream,"%d \n",&NumberVars);     
	fscanf(stream,"%d \n",&dim);  
	fscanf(stream,"%d \n",&neigh);  
        fscanf(stream,"%d \n",&width);  
   
    Createlattice();   

    for (i=0; i<NumberVars; i++)  
	{ 
     fscanf(stream,"%d ",&lattice[i][0]); 
     // cout<<lattice[i][0]<<" ";
	 if(lattice[i][0]>0)  
	 { 
	   for (j=1; j<lattice[i][0]+1; j++) 
              {
                fscanf(stream,"%d ", &lattice[i][j]);	  
		// cout<<lattice[i][j]<<" ";
              }

	   for (j=0; j<lattice[i][0]; j++) 
              {
               	 
                fscanf(stream,"%d ", &auxint);
		Inter[i][j] =auxint;
                //cout<<int(Inter[i][j])<<" ";
              }
	   //cout<<endl;
 
	 }		 
	}  
	  
	fclose(stream);  
}   

void Ising::Createlattice()
{
 int i,neighbors_x;   
 lattice = new int*[NumberVars];
 Inter =  new double*[NumberVars];
 neighbors_x = int(pow(2,neigh)*dim);

 for(i=0;i<NumberVars;i++) 
  {
   lattice[i] = new int[neighbors_x+1]; 
   Inter[i] = new double[neighbors_x];
  }
}

void Ising::InitLattice()
{
    int i,j,k,auxv,auxn;
    int* vector_r;
    vector_r = new int[dim] ;

    
    for(i=0; i<NumberVars; i++)
    {
        lattice[i][0]=0;
	NumConvert(i,dim,width,&*vector_r);
	/*
	cout<<i<<" "<<dim<<" "<<width<<" "<<neigh<<" --> ";
	    for(int l=0; l<dim; l++) cout<<vector_r[l]<<" ";
                  cout<<endl;
	*/
        for(j=0; j<dim; j++)
           for(k=1; k<=neigh; k++)
	   {
	     auxv =  vector_r[j];
            
             if(auxv-k <0)  vector_r[j] = width -k;
             else vector_r[j]--;
              auxn=  ConvertNum(dim,width,vector_r);
              //cout<<"auxn "<<auxn<<endl;
             lattice[i][0]++;
             lattice[i][lattice[i][0]]=auxn;

             if(auxv+k > width-1)  vector_r[j] = k - 1;
             else vector_r[j] = auxv+1;
             auxn = ConvertNum(dim,width,vector_r);
             //cout<<"auxn "<<auxn<<endl;
             lattice[i][0]++;
             lattice[i][lattice[i][0]]=auxn;

             vector_r[j] = auxv;           
           }
    }  
    delete[] vector_r;
 RandomSpins();
}

void Ising::RandomSpins()
{
    int i,j,k,auxn;
    for (i=0; i<NumberVars; i++)  
	{ 
     	 if(lattice[i][0]>0)  
	  for (j=1; j<lattice[i][0]+1; j++) 
	    {
		auxn =  lattice[i][j];
	        if(i<auxn)  Inter[i][j-1] = (1-2*(myrand()>0.5));
		k = 1;	 
		while(lattice[auxn][k] != i) k++;
                Inter[auxn][k-1] = Inter[i][j-1];
	    }		 
	}  
}


void Ising::SaveInstance(char* filename)   
{  
	FILE *stream;  
	int i,j;  
  
    stream = fopen(filename, "w+" );  
	fprintf(stream,"%d \n",NumberVars);     
	fprintf(stream,"%d \n",dim);  
	fprintf(stream,"%d \n",neigh);  
        fprintf(stream,"%d \n",width); 

    for (i=0; i<NumberVars; i++)  
	{ 
        fprintf(stream,"%d ",lattice[i][0]); 
	 if(lattice[i][0]>0)  
	 { 
	   for (j=1; j<lattice[i][0]+1; j++) fprintf(stream,"%d ", lattice[i][j]);	  
	   for (j=0; j<lattice[i][0]; j++)  fprintf(stream,"%d ", int(Inter[i][j]));	 
	 }	
	  fprintf(stream,"\n"); 
	}  
	  
	fclose(stream);  
}   




void Ising::SaveInstanceforChecking(char* filename)   
{  
	FILE *stream;  
	int i,j;  
  
  stream = fopen(filename, "w+" ); 
 fprintf(stream,"type: pm \n");
 fprintf(stream,"name: ");
 fprintf(stream, filename);
 fprintf(stream," \n");
 fprintf(stream,"size: %d \n",width);  
 fprintf(stream," \n");
  	 
    for (i=0; i<NumberVars; i++)  
	{ 
     	 if(lattice[i][0]>0)  
	 { 
          
	   for (j=1; j<lattice[i][0]+1; j++) 
             {
               
                if(i<lattice[i][j])
		{
                  fprintf(stream,"%d ", i+1);
                  fprintf(stream,"%d ", lattice[i][j]+1);	  
	          fprintf(stream,"%d ", int(Inter[i][j-1]));
                  fprintf(stream," \n");
                }
                 
              }
	 }  
	 
        }
	fclose(stream);  
}   
 


double Ising::evalfunc(unsigned int* x)   
{  
int i,j,auxeq;  
double sum;
 
 sum = 0;  
 
    for (i=0; i<NumberVars; i++)  
	{   
	 if(lattice[i][0]>0)  
	 { 
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {
           if(i<lattice[i][j])
	    {
             auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
	     sum += (auxeq*Inter[i][j-1]);	 
            }
	  }
	 }		 
	}  
//    cout<<sum<<endl;
//if(x[0]==0) for (i=0; i<NumberVars; i++)  x[i]=1-x[i];  
  return sum;
}   

double Ising::evalfunc(int* x)   
{  
int i,j,auxeq;  
double sum;
 
 sum = 0;  
 
    for (i=0; i<NumberVars; i++)  
	{   
	 if(lattice[i][0]>0)  
	 { 
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {
           if(i<lattice[i][j])
	    {
             auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
	     sum += (auxeq*Inter[i][j-1]);	 
            }
	  }
	 }		 
	}  
//    cout<<sum<<endl;
    // if(x[0]==0)	for (i=0; i<NumberVars; i++)  x[i]=1-x[i];
  return sum;
}   



Ising::~Ising()
{
 int i;
 for(i=0;i<NumberVars;i++) 
  {
   delete[] lattice[i]; 
   delete[] Inter[i];
  }

    delete[] lattice;
    delete[] Inter;
}
