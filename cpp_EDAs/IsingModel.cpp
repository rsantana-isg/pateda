#include <iostream>
#include <fstream>

#include "IsingModel.h"
#include "auxfunc.h"

using namespace std;

Ising::Ising(int nvars, int wid, int d, int n) 
{
    NumberVars = nvars;
    width = wid;
    dim = d;
    neigh = n;        
    EvalAuxVar_Left = new int[NumberVars];
    EvalAuxVar_Right = new int[NumberVars];
    P_Moves = new int[NumberVars];
    Is_Promising = new int[NumberVars];
    init_x = new unsigned int[NumberVars];
    tabu_moves = new unsigned int[NumberVars];
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

        EvalAuxVar_Left = new int[NumberVars];
        EvalAuxVar_Right = new int[NumberVars];
        P_Moves = new int[NumberVars];
        Is_Promising = new int[NumberVars];
        init_x = new unsigned int[NumberVars];
        tabu_moves = new unsigned int[NumberVars];

    //cout<<NumberVars<<" "<<dim<<" "<<neigh<<" "<<width<<endl;         
    Createlattice();  
  
    for (i=0; i<NumberVars; i++)  
	{ 
           fscanf(stream,"%d ",&lattice[i][0]); 
           //cout<<lattice[i][0]<<" ";
	   for (j=1; j<width+1; j++) 
              {
                fscanf(stream,"%d ", &lattice[i][j]);	  
		//cout<<lattice[i][j]<<" ";
              }

	   for (j=0; j<width; j++) 
              {               	 
                fscanf(stream,"%d ", &auxint);
		Inter[i][j] =auxint;
		// cout<<int(Inter[i][j])<<" ";
              }
	   // cout<<endl; 
	 		 
	}  
	  
	fclose(stream);  
}   

void Ising::Createlattice()
{
 int i,neighbors_x;   
 lattice = new int*[NumberVars];
 Inter =  new double*[NumberVars];
 //neighbors_x = int(pow(2,neigh)*dim);
 neighbors_x = int(width);

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
	
	cout<<i<<" "<<dim<<" "<<width<<" "<<neigh<<" --> ";
	    for(int l=0; l<dim; l++) cout<<vector_r[l]<<" ";
                  cout<<endl;
	
        for(j=0; j<dim; j++)
           for(k=1; k<=neigh; k++)
	   {
	     auxv =  vector_r[j];
            
             if(auxv-k <0)  vector_r[j] = width -k;
             else vector_r[j]--;
              auxn=  ConvertNum(dim,width,vector_r);
              cout<<"auxn "<<auxn<<endl;
             lattice[i][0]++;
             lattice[i][lattice[i][0]]=auxn;

             if(auxv+k > width-1)  vector_r[j] = k - 1;
             else vector_r[j] = auxv+1;
             auxn = ConvertNum(dim,width,vector_r);
             cout<<"auxn "<<auxn<<endl;
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
         else x[i] = 1;		 
	 // cout<<i<<" "<<lattice[i][0]<<" "<<sum<<endl; 
	}  
    //cout<<sum<<endl;
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
              //cout<<i<<" "<<lattice[i][j]<<" "<<Inter[i][j-1]<<"  "<<sum<<endl;
            }
	  }
	 }		 
	}  
    //cout<<sum<<endl;
    // if(x[0]==0)	for (i=0; i<NumberVars; i++)  x[i]=1-x[i];
  return sum;
}   


// Evaluates the fitness function of the Ising configuration and applies a  
// hill climbing bit-flip method that among the $n$ possible bit-flips always
// selects the one the improves the fitness the most
// The local-search only stops when there is not any one-spin move that could 
// improve the solutions. Every time a bit-flip is proposed
// the whole solution is evaluated

double Ising::greedy_evalfunc(unsigned int* x)   
{  
int i,best_i;  
 double val,best_val;
 
 best_val = evalfunc(x);
 best_i = -1;    
   for (i=0; i<NumberVars; i++)  
     {   
       x[i] = 1-x[i];
       val =  evalfunc(x);
       if(val>best_val)
	 {
	   best_i = i;
           best_val = val;
	  }
	 x[i] = 1-x[i];		 
	 // cout<<i<<" "<<lattice[i][0]<<" "<<sum<<endl; 
	}  
   if (best_i>-1)  x[best_i] = 1-x[best_i];	
   
  return best_val;
}   

// Local search where positions for bit-flips are selected at random
// If the fitness is improved after the bit-flip the solution is updated
// The number of moves that are proposed is  determined by the parameter ntrials

double Ising::random_evalfunc(unsigned int* x, int ntrials)   
{  
 int i,j;  
 double val,best_val;
 
   best_val = evalfunc(x);     
   for (j=0; j<ntrials; j++)  
     {   
       i = randomint(NumberVars);
       if(lattice[i][0]>0)
	 {
          x[i] = 1-x[i];
          val =  evalfunc(x);
          if(val>=best_val) best_val = val;
          else  x[i] = 1-x[i];		 
         }
	 // cout<<i<<" "<<lattice[i][0]<<" "<<sum<<endl; 
	}  
   //if (best_i>-1)  x[best_i] = 1-x[best_i];	   
  return best_val;
}   


// More efficient implementation of EvalIsingGreeedy
// Evaluates the fitness function of the Ising configuration and applies a  
// hill climbing bit-flip method that among the $n$ possible bit-flips always
// selects the one the improves the fitness the most
// The local-search only stops when there is not any one-spin move that could 
// improve the solutions. When bit-flit are applied only local fitness function is computed
// and the total energy is updated from this local change

double Ising::HC_evalfunc(unsigned int* x)   
{  
  int i,j,k,auxeq,best_i;  
  double val,best_val;  
  //cout<<"Init" <<endl;
    for (i=0; i<NumberVars; i++)  
	{   
          if(lattice[i][0]==0) x[i] = 0;
          EvalAuxVar_Left[i] = 0 ;         
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {           
	      auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
              EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];          
	  }	 		 
           EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i];
	}  


 best_i = 0;
 while(best_i>-1)
  {
   best_val = -10000;
   best_i = -1;      
   val = 0;
   for (i=0; i<NumberVars; i++)  
     {   
       if((EvalAuxVar_Right[i]>best_val) && (EvalAuxVar_Left[i]< EvalAuxVar_Right[i]) )
	 {
	   best_i = i;
           best_val = EvalAuxVar_Right[i];
	 }
        val+= EvalAuxVar_Left[i];
     }  

   if (best_i>-1) 
     { 
       //cout<<"B "<<best_i<<" "<<val/2<<" "<<best_val<<" otherval "<<evalfunc(x)<<" "<<EvalAuxVar_Left[best_i]<<" "<<EvalAuxVar_Right[best_i]<<endl;

       x[best_i] = 1-x[best_i];	
       val = val + 2*(EvalAuxVar_Right[best_i]-EvalAuxVar_Left[best_i]);
       k =  EvalAuxVar_Right[best_i];
       EvalAuxVar_Right[best_i] = EvalAuxVar_Left[best_i];
       EvalAuxVar_Left[best_i] = k;     

       //cout<<"A "<<best_i<<" "<<val/2<<" "<<best_val<<" otherval "<<evalfunc(x)<<" "<<EvalAuxVar_Left[best_i]<<" "<<EvalAuxVar_Right[best_i]<<endl;

       for (k=0; k<lattice[best_i][0]; k++)
	 {
          i = lattice[best_i][k+1]; 
          EvalAuxVar_Left[i] = 0 ;       
          for (j=1; j<lattice[i][0]+1; j++) 
	   {           
	      auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
              EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];             
	   }	
          EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i]; 
         }
     }
    
  }
   
   return val/2.0;  // The interactions are counted twice
}   



// Evaluates the fitness function applying best-flip moves until no improvements
// are possible. Then randomly selects nchanges variables and flip them 
// allowing the fitness function to decrease
// The process is repeated ntrial times  

double Ising::Tabu_evalfunc(unsigned int* x, int ntrials, int nchanges)   
{  
  int i,j,k,l,auxeq,best_i,to_change,trials;  
  double val,best_val,CurrentVal;  
  double Val,InitVal;
  Val = 0;
  trials = 0;

  for (i=0; i<NumberVars; i++) init_x[i] = x[i]; 

    for (i=0; i<NumberVars; i++)  
	{   
          tabu_moves[i] = 0;
          if(lattice[i][0]==0) x[i] = 0;
          EvalAuxVar_Left[i] = 0 ;         
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {           
	      auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
              EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];          
	  }	 		 
           EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i];
           Val += EvalAuxVar_Left[i];
	}  

    CurrentVal=Val;
    InitVal = Val;
    while( (CurrentVal<=InitVal) && (trials<ntrials))
      {
       best_i = 0;
       while(best_i>-1)
         {
           best_val = -10000;
           best_i = -1;      
           val = 0;
           for (i=0; i<NumberVars; i++)  
             {   
               if((tabu_moves[i]==0)  && (EvalAuxVar_Right[i]>best_val) && (EvalAuxVar_Left[i]< EvalAuxVar_Right[i]) )
	         {
 	           best_i = i;
                   best_val = EvalAuxVar_Right[i];
	         }
               val+= EvalAuxVar_Left[i];
             }  


           if (best_i>-1) 
             { 
                x[best_i] = 1-x[best_i];	
                val = val + 2*(EvalAuxVar_Right[best_i]-EvalAuxVar_Left[best_i]);
                k =  EvalAuxVar_Right[best_i];
                EvalAuxVar_Right[best_i] = EvalAuxVar_Left[best_i];
                EvalAuxVar_Left[best_i] = k;     
                //cout<<"A "<<best_i<<" "<<val/2<<" "<<best_val<<" otherval "<<evalfunc(x)<<" "<<EvalAuxVar_Left[best_i]<<" "<<EvalAuxVar_Right[best_i]<<endl;
                for (k=0; k<lattice[best_i][0]; k++)
	         {
                   i = lattice[best_i][k+1]; 
                   EvalAuxVar_Left[i] = 0 ;       
                   for (j=1; j<lattice[i][0]+1; j++) 
 	             {           
  	                auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
                        EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];             
	             }	
                   EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i]; 
                 }
             }    
	 }

       CurrentVal = val;
       //cout<<trials<<" "<<InitVal<<" "<<CurrentVal<<" "<<best_i<<endl;
       if (CurrentVal<=InitVal)
	 {
           for (i=0; i<NumberVars; i++)  tabu_moves[i] = 0;
	   for (l=0; l<nchanges; l++)
	      {
                to_change = randomint(NumberVars); 
                if(lattice[to_change][0]>0) 
		  {
                   x[to_change] = 1-x[to_change];	
                   val = val + 2*(EvalAuxVar_Right[to_change]-EvalAuxVar_Left[to_change]);
                   k =  EvalAuxVar_Right[to_change];
                   EvalAuxVar_Right[to_change] = EvalAuxVar_Left[to_change];
                   EvalAuxVar_Left[to_change] = k;     
                   //tabu_moves[to_change] = 1;
                   for (k=0; k<lattice[to_change][0]; k++)
	           {
                     i = lattice[to_change][k+1]; 
                     EvalAuxVar_Left[i] = 0 ;       
                     for (j=1; j<lattice[i][0]+1; j++) 
 	              {           
  	                auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
                        EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];             
	               }	
                     EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i]; 
                   }
                  }
	      }
         }        
       trials++;
      }
   
    /*
    if (CurrentVal<=InitVal)
      { 
       for (i=0; i<NumberVars; i++)  x[i] = init_x[i];
       CurrentVal = InitVal; 
      }
    */
   
   return CurrentVal/2.0;  // The interactions are counted twice
}   


// Evaluates the fitness function applying best-flip moves until no improvements
// are possible. Then, if no improvement with respect to the the initial solution
// obtained, a maximum of local_search_ntrials is applied. For each move the variable select
// the movement that decreases the fitness function the least. This movement is also included in a tabu_list. 
// After that movement, the function tries to increase the fitness again for maximum 
// local_search_ntrial trials

double Ising::SA_evalfunc(unsigned int* x, int ntrials)   
{  
  int i,j,k,auxeq,best_i,trials;  
  double val,best_val;  
  double Val;
  Val = 0;
  trials = 0;
  
    for (i=0; i<NumberVars; i++)  
	{   
          tabu_moves[i] = 0;
          if(lattice[i][0]==0) x[i] = 0;
          EvalAuxVar_Left[i] = 0 ;         
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {           
	      auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
              EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];          
	  }	 		 
           EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i];
           Val += EvalAuxVar_Left[i];
	}  

 
    val = -1000000;
    best_val = 1;
    while( (best_val>0) || (trials<ntrials && val<=Val))
         {
           best_val = -10000000;
           best_i = -1;      
           val = 0;
           for (i=0; i<NumberVars; i++)  
             {   
               if((tabu_moves[i]==0) && (lattice[i][0]>0) && (EvalAuxVar_Right[i]>best_val) )
	         {
 	           best_i = i;                     
                   best_val = EvalAuxVar_Right[i];                    
	         }              
               val+= EvalAuxVar_Left[i];
             }  
           
	   
           if(best_val<=0) 
	     {                
               trials++;  
               tabu_moves[best_i]=1;
               //cout<<trials<<" "<<best_i<<" "<<tabu_moves[best_i]<<" "<<best_val<<endl;
             }
          
                 
      if( (best_val>0) || (trials<ntrials && val<=Val))
	{   
           x[best_i] = 1-x[best_i];	
           val = val + 2*(EvalAuxVar_Right[best_i]-EvalAuxVar_Left[best_i]);
           k =  EvalAuxVar_Right[best_i];
           EvalAuxVar_Right[best_i] = EvalAuxVar_Left[best_i];
           EvalAuxVar_Left[best_i] = k;     
                //cout<<"A "<<best_i<<" "<<val/2<<" "<<best_val<<" otherval "<<evalfunc(x)<<" "<<EvalAuxVar_Left[best_i]<<" "<<EvalAuxVar_Right[best_i]<<endl;         
           for (k=0; k<lattice[best_i][0]; k++)
	         {
                   i = lattice[best_i][k+1]; 
                   EvalAuxVar_Left[i] = 0 ;       
                   for (j=1; j<lattice[i][0]+1; j++) 
 	             {           
  	                auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
                        EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];             
	             }	
                   EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i]; 
                 }
	 }         
      //if(best_val<=0)   cout<<trials<<" "<<val<<" "<<Val<<" "<<best_i<<" "<<tabu_moves[best_i]<<" "<<best_val<<endl;
       }  
     
    /*
    if (CurrentVal<=InitVal)
      { 
       for (i=0; i<NumberVars; i++)  x[i] = init_x[i];
       CurrentVal = InitVal; 
      }
    */
   
   return val/2.0;  // The interactions are counted twice
}   



// Evaluates the fitness function applying best-flip moves until no improvements
// are possible. For each bit-flip, the move that improves the fitness the most is selected.
//  A list of promising moves is updated in every step in such a way that only a reduced number
// of spins are considered 
// The algorithm stops when no improvement is achieved (local minima)

double Ising::Best_SA_evalfunc(unsigned int* x)   
{  
  int i,j,k,auxeq,best_i,aux_ind;  
  double val,best_val;  
  double Val;
  int n_promising_moves;

  Val = 0;
  n_promising_moves  = 0;

    for (i=0; i<NumberVars; i++)  
	{           
          Is_Promising[i] = -1;
          if(lattice[i][0]==0) x[i] = 0;
          EvalAuxVar_Left[i] = 0 ;         
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {           
	      auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
              EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];          
	  }	 		 
           EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i];
           if(EvalAuxVar_Right[i]>0)
	     {
               P_Moves[n_promising_moves] = i;
               Is_Promising[i] = n_promising_moves;
	       n_promising_moves++;
             } 
           Val += EvalAuxVar_Left[i];
	}  

    //cout<<"Number promising points"<<n_promising_moves<<endl; 
    val = Val;
    
    while(n_promising_moves>0)
         {         
           best_val = -1000000000;
	   for(j=0;j<n_promising_moves;j++)
          {                           
               if(EvalAuxVar_Right[P_Moves[j]]>best_val) 
		 {
                   aux_ind = j;
                   best_val = EvalAuxVar_Right[P_Moves[j]];
                 }
          }	         

           best_i = P_Moves[aux_ind];              // Which variable is selected                 
           if(best_val>0)
	     {  
              if(n_promising_moves>1)
	       {                
		  P_Moves[aux_ind] = P_Moves[n_promising_moves-1]; // The last moves comes to position aux_ind                              
                  Is_Promising[P_Moves[aux_ind]] = aux_ind;    // The position o the index is updated.                 
               }
              Is_Promising[best_i] = -1;                  // The selected move is set to -1 
              n_promising_moves--;                           
	      
              x[best_i] = 1-x[best_i];	                
              val = val + 2*(EvalAuxVar_Right[best_i]-EvalAuxVar_Left[best_i]);
              k =  EvalAuxVar_Right[best_i];
              EvalAuxVar_Right[best_i] = EvalAuxVar_Left[best_i];
              EvalAuxVar_Left[best_i] = k;                    

            
              for (k=0; k<lattice[best_i][0]; k++)
	         {
                   i = lattice[best_i][k+1];                        
                   EvalAuxVar_Left[i] = 0 ;       
                   for (j=1; j<lattice[i][0]+1; j++) 
 	             {           
  	                auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
                        EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];             
	             }
                   
                   EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i]; 

                   if( (EvalAuxVar_Right[i]>0) && (Is_Promising[i]==-1)) // If a  new promising move is created we add it
		     {
                      P_Moves[n_promising_moves] = i;                    
                      Is_Promising[i] = n_promising_moves;                                            
                      n_promising_moves++;                      
                     }
                   else if( (EvalAuxVar_Right[i]<=0) && (Is_Promising[i]>-1)) // If an old promising move is not anymore
                     {                                                  // we remove it
                      if(n_promising_moves>1)
     		      {
     	                aux_ind = Is_Promising[i];
                        P_Moves[aux_ind] = P_Moves[n_promising_moves-1]; // The last moves comes to position aux_ind                              
                        Is_Promising[P_Moves[aux_ind]] = aux_ind;    // The position o the index is updated.                       
                      }                     
                       Is_Promising[i] = -1;                  // The selected move is set to -1        
                       n_promising_moves--;                     
                     }        		 
                 }	
              
	     }
          
	   //cout<<val<<" "<<Val<<" "<<best_i<<" "<<best_val<<" "<<n_promising_moves<<endl;
	 }
              
    /*
    if (CurrentVal<=InitVal)
      { 
       for (i=0; i<NumberVars; i++)  x[i] = init_x[i];
       CurrentVal = InitVal; 
      }
    */
   
   return val/2.0;  // The interactions are counted twice
}   


// Idem Best_SA_evalfunc but instead of selecting the bitflip that improves the fitness
// the most, it is randomly selected among those that improve the fitness
//  A list of promising moves is updated in every step in such a way that only a reduced number
// of spins are considered 
// The algorithm stops when no improvement is achieved (local minima)

double Ising::Random_SA_evalfunc(unsigned int* x)   
{  
  int i,j,k,auxeq,best_i,aux_ind;  
  double val,best_val;  
  double Val;
  int n_promising_moves;
  Val = 0;
 
  n_promising_moves  = 0;
    for (i=0; i<NumberVars; i++)  
	{           
          Is_Promising[i] = -1;
          if(lattice[i][0]==0) x[i] = 0;
          EvalAuxVar_Left[i] = 0 ;         
	  for (j=1; j<lattice[i][0]+1; j++) 
	  {           
	      auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
              EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];          
	  }	 		 
           EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i];
           if(EvalAuxVar_Right[i]>0)
	     {
               P_Moves[n_promising_moves] = i;
               Is_Promising[i] = n_promising_moves;
	       n_promising_moves++;
             } 
           Val += EvalAuxVar_Left[i];
	}  

    //cout<<"Number promising points"<<n_promising_moves<<endl; 
    val = Val;
   
    while(n_promising_moves>0)
         {         
           aux_ind = randomint(n_promising_moves); // Which index of the feasible moves 
           best_i = P_Moves[aux_ind];              // Which variable is selected
           best_val = EvalAuxVar_Right[best_i];     // The improvement for this variable                       
           if(best_val>0)
	     {  
              if(n_promising_moves>1)
	       {                
		  P_Moves[aux_ind] = P_Moves[n_promising_moves-1]; // The last moves comes to position aux_ind                   
                  Is_Promising[P_Moves[aux_ind]] = aux_ind;    // The position o the index is updated.                 
               }
              n_promising_moves--;              
              Is_Promising[best_i] = -1;                  // The selected move is set to -1            

              x[best_i] = 1-x[best_i];	
              val = val + 2*(EvalAuxVar_Right[best_i]-EvalAuxVar_Left[best_i]);
              k =  EvalAuxVar_Right[best_i];
              EvalAuxVar_Right[best_i] = EvalAuxVar_Left[best_i];
              EvalAuxVar_Left[best_i] = k;                    
              for (k=0; k<lattice[best_i][0]; k++)
	         {
                   i = lattice[best_i][k+1]; 
                   EvalAuxVar_Left[i] = 0 ;       
                   for (j=1; j<lattice[i][0]+1; j++) 
 	             {           
  	                auxeq = 2*(x[i]==x[lattice[i][j]])-1;  
                        EvalAuxVar_Left[i] += auxeq*Inter[i][j-1];             
	             }	
                   EvalAuxVar_Right[i] = -1*EvalAuxVar_Left[i]; 
                   if( (EvalAuxVar_Right[i]>0) && (Is_Promising[i]==-1)) // If a  new promising move is created we add it
		     {
                      P_Moves[n_promising_moves] = i;                    
                      Is_Promising[i] = n_promising_moves;                                            
                      n_promising_moves++; 
                     }
                   else if( (EvalAuxVar_Right[i]<=0) && (Is_Promising[i]>-1)) // If an old promising move is not anymore
                     {                                                  // we remove it
                      if(n_promising_moves>1)
     		      {
     	                aux_ind = Is_Promising[i];
                        P_Moves[aux_ind] = P_Moves[n_promising_moves-1]; // The last moves comes to position aux_ind                      
                        Is_Promising[P_Moves[aux_ind]] = aux_ind;    // The position o the index is updated.
                      }
                      n_promising_moves--;
                      Is_Promising[i] = -1;                  // The selected move is set to -1              
                     }                      
                 }	      
	     }         
	   //cout<<val<<" "<<Val<<" "<<best_i<<" "<<best_val<<" "<<n_promising_moves<<endl;
	}
     
    /*
    if (CurrentVal<=InitVal)
      { 
       for (i=0; i<NumberVars; i++)  x[i] = init_x[i];
       CurrentVal = InitVal; 
      }
    */
   
   return val/2.0;  // The interactions are counted twice
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
    delete[] EvalAuxVar_Left;    
    delete[] EvalAuxVar_Right;  
    delete[] P_Moves;
    delete[] Is_Promising; 
    delete[] init_x;
    delete[] tabu_moves;
}
