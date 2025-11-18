// Problem.cpp: implementation of the methods specific to the problem.
//

#include "EDA.h"
#include "Problem.h" 
#include "AllFunctions.h"

unsigned  int* auxvector;
//	int  chainlength;
//extern Ising* MyIsing; 
// extern CNF *AllClauses;

//extern  HPProtein* FoldingProtein;

extern int  FUNC;
extern int* paramas;

int GetProblemInfo()
{


 STATES = new int[IND_SIZE];      
         
        auxvector   = new unsigned int[IND_SIZE];   
	for(int i=0;i<IND_SIZE;i++) STATES[i] = 3;
        SetParam(params);
	return(0);
}

void RemoveProblemInfo()
{
	// The memory allocated in GetProblemInfo() is returned.

	delete [] STATES;
        delete [] auxvector;

}


double Metric(int * genes)
{
	// The value of an individual consists of the number
	// of 1s that it has.
  double value; 
    
  //FoldingProtein->CallRepair(genes,IND_SIZE);
     for(int i=0;i<IND_SIZE;i++) auxvector[i] = genes[i];
     // double value  = FoldingProtein->EvalOnlyVector(IND_SIZE,auxvector);
        //cout<<value<<endl;
      //double value =  MyIsing->evalfunc(genes);
    //double value =  AllClauses->SatClauses(genes);
       	EVALUATIONS++;
	return value;
} 




/*
int GetProblemInfo()
{
	// An invidual of 20 binary genes is defined.

	//IND_SIZE = 36;

	STATES = new int[IND_SIZE];
	for(int i=0;i<IND_SIZE;i++)
		STATES[i] = 2;

	return(0);
}

void RemoveProblemInfo()
{
	// The memory allocated in GetProblemInfo() is returned.

	delete [] STATES;
}

double Metric(int * genes)
{
  // Function deceptive 3 Goldberg
  int i; 
double decep3[]={0.9, 0.8, 0.8, 0, 0.8, 0, 0, 1};
 value = 0;
 double sum = 0;
  for(int i = 0; i < IND_SIZE; i += 3)
       value += genes[ value[i] + 2 * genes[i + 1] + 4 * genes[i + 2] ];
  EVALUATIONS++;
	return value;
}


double Metric(int * genes)
{
	// The value of an individual consists of the number
	// of 1s that it has.

        double value =  eval(FUNC,genes,IND_SIZE);
       
        //For(int i=0;i<IND_SIZE;i++) value += genes[i];
	// This allows us to keep track of the number
	// of evaluations.
	EVALUATIONS++;
	return value;
}

*/

/*
int GetProblemInfo()
{ 
 
	int FAstates,i; 
	int ch[24] = {0,0.0,0,0,1,0,1,0,0,1,1,1,0,0,1,0,1,1,1,0,1,1,1}; 


	chainlength = 24; 
	chain = new int[chainlength]; 
	for(i=0;i<chainlength;i++) chain[i] = ch[i]; 
 
	FAstates = IND_SIZE/4; 
      
	STATES = new int[IND_SIZE]; 
	  
	for(int i=0;i<IND_SIZE;i+=4) 
	{
		STATES[i] = FAstates; 
		STATES[i+1] = 2; 
		STATES[i+2] = FAstates; 
		STATES[i+3] = 2;
	} 
       	return(0);
} 


void RemoveProblemInfo()
{
	// The memory allocated in GetProblemInfo() is returned.

	delete [] STATES; 
	delete [] chain;
}

double Metric(int * genes)
{
    int i,state;
	double value; 

       	i=0; 
	state = 0; 
	value = 0; 
	while(i<chainlength-1)
	{ 
           value += (chain[i+1] == genes[4*state+1+2*chain[i]]); 
           state = genes[4*state+2*chain[i]]; 
           cout<<"Val: "<<value<<" State:"<<state<<endl;
	   i++; 
	} 
        
	EVALUATIONS++;
        cout<<EVALUATIONS<<endl;
	return value;
}

*/
