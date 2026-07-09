#include "auxfunc.h" 
#include "Treeprob.h"
#include "IntTreeprob.h" 
#include <math.h> 
#include <time.h> 
 
  
 
IntProbTree::IntProbTree(int vars, int* AllInd, int psize,int clusterkey,unsigned *AllMaxValues,Popul* pop):ProbTree(vars,AllInd,psize,clusterkey,pop) 
 { 
  int i; 
  MaxValues = AllMaxValues; // It is a pointer to save memory 
  GenInd = new int*[length]; 
  for(i=0; i < length; i++) GenInd[i] = new int[actualpoolsize]; 
  ParentValues = new int*[length]; 
  Maximum_among_Minimum = 0; 
  for(i=0; i < length; i++)  
  { 
	  ParentValues[i] = new int[MaxValues[i]]; 
	  if ( MaxValues[i] > Maximum_among_Minimum ) 
		  Maximum_among_Minimum = MaxValues[i]; 
  } 
 
 } 
 
 
 
 
double IntProbTree::Prob (unsigned* vector) 
 { 
  // Se determina cual es la probabilidad del 
 // vector dado el arbol 
 
 double prob; 
 int j; 
 
  prob = 1; 
 
	for (j=0;j<length; j++ ) 
	 { 
      if (Tree[j]==-1) prob *= UnivProb(j,vector[j]); 
	  else prob *= CondProb(Tree[j],vector[Tree[j]],j,vector[j]); 
	  if (prob == 0) return 0; 
	 } 
	 
	return prob;  
} 
 
 
IntProbTree::~IntProbTree() 
 { 
  int i; 
  for(i=0; i < length; i++)  
  { 
	  delete[] GenInd[i];   
	  delete[] ParentValues[i];   
  } 
	  delete[] GenInd; 
	  delete[] ParentValues; 
 
 } 
 
double IntProbTree::UnivProb(int var, unsigned varval) 
{ 
  int Freq; 
  if(varval == 0) Freq = ParentValues[var][0] + 1; 
  else if(ParentValues[var][varval-1] == 0 )  
	              Freq = ParentValues[var][varval] - ParentValues[var][varval-1] +1; 
  else 	Freq = ParentValues[var][varval] - ParentValues[var][varval-1]; 
  return Freq/double(actualpoolsize); 
} 
 
double IntProbTree::CondProb(int parent, unsigned parentval, int son, unsigned sonval) 
{ 
   int auxval; 
   int initindex,endindex,i;    
 
	if(parentval == 0) 
	{  
	  initindex = 0; 
	  endindex = ParentValues[parent][0]; 
	} 
	else  
	{ 
	  initindex = ParentValues[parent][parentval-1] + 1; 
	  endindex = ParentValues[parent][parentval]; 
	} 
	if (endindex==(initindex-1)) return 0; // This case should not happen 
 	 
	auxval = 0; 
	for (i=initindex;i<= endindex; i++ ) 
	{ 
	       auxval+= (Pop->P[GenInd[parent][i]][son] == sonval); 
	} 
		 
    return auxval/double(endindex-(initindex-1)); 
} 
 
unsigned IntProbTree::SonValue(int parent, unsigned parentval, int son) 
{ 
	int CantSons,auxval; 
	 
	if(parentval == 0) CantSons = ParentValues[parent][0] + 1; 
	else CantSons = ParentValues[parent][parentval] - ParentValues[parent][parentval-1]; 
	auxval = (CantSons-1)*myrand(); 
	auxval = GenInd[parent][ParentValues[parent][parentval]-auxval]; 
    return Pop->P[auxval][son]; 
} 
 
int IntProbTree::UnivFreq(int variable, unsigned val) 
{ 
	if(val == 0) return ParentValues[variable][0] + 1; 
	return ParentValues[variable][val] - ParentValues[variable][val-1]; 
}	  
 
 
void IntProbTree::GenIndividual (Popul* NewPop, int pos) 
 { 
  // The vector in position pos is generated 
  
 int auxval,j,i; 
  
 
 	for (i=0;i<length; i++ ) 
	 { 
      j = NextInOrder(i); 
	 
	  if (Tree[j]==-1)  
	  { 
		  auxval = (actualpoolsize-1)*myrand(); 
	      NewPop->P[pos][j] = Pop->P[actualindex[auxval]][j]; 
	  } 
	  else  
	      NewPop->P[pos][j] =  SonValue(Tree[j],NewPop->P[pos][Tree[j]],j); 
	  } 
} 
 
 
void IntProbTree::GenPop(int From, Popul* NewPop) 
{  
	int i; 
    for(i=From; i<NewPop->psize; i++)  GenIndividual (NewPop,i);  
} 
 
void IntProbTree::CalMutInf() 
 { 
  // The Mutual Information Matrix is constructed 
 
 
   int i,k,j; 
   int aux; 
   unsigned current_pair[2]; 
   int *indexpairs; 
   int remaining_configurations,BivFreq; 
   double BivProb, UnivProb_j,UnivProb_k; 
 
       
	indexpairs = new int[actualpoolsize];        
	 
 
	 for(j=0; j<length-1; j++)         // For all possible pairs of variables 
	  	for(k=j+1 ; k<length; k++) 
		{ 
         aux = j*(2*length-j+1)/2 +k-2*j-1; 
		 MutualInf[aux]=0; 
      	  
	     for(i=0; i<actualpoolsize; i++) indexpairs[i] = actualindex[i]; 
		 remaining_configurations = actualpoolsize; 
 	     while(remaining_configurations > 0) 
          { 
			    BivFreq = 0;                
				current_pair[0] = Pop->P[indexpairs[0]][j]; 
				current_pair[1] = Pop->P[indexpairs[0]][k]; 
 
			    i = 0; 
			    while(i< remaining_configurations) 
				{  
                 if( current_pair[0] == Pop->P[indexpairs[i]][j] &&  
				   current_pair[1] == Pop->P[indexpairs[i]][k] )  
				 { 
                  BivFreq++; 
				  indexpairs[i] = indexpairs[remaining_configurations-1]; 
				  remaining_configurations--; 
				 } 
			     else i++; 
				} 			     
			  BivProb = (BivFreq)/double(actualpoolsize); 
			  UnivProb_j = UnivProb(j,current_pair[0]); 
			  UnivProb_k = UnivProb(k,current_pair[1]); 
  		      if (BivProb > 0.000000001)  
                MutualInf[aux]+=BivProb*(log(BivProb/(UnivProb_j*UnivProb_k))); 
			} 
		} 
		delete[] indexpairs; 
  }	           			 
 
void IntProbTree::MakeProbStructures() 
{ 
	int i,j,k,g,auxindex; 
	int *indexvals; 
    unsigned l; 
	indexvals = new int[actualpoolsize]; 
 
 for(i=0; i<length; i++) 
 { 
	for(j=0; j<actualpoolsize; j++)  indexvals[j] = actualindex[j]; 
	g = 0; 
    for(j=0; j<actualpoolsize-1; j++) 
	 { 
       for(k=j+1; k<actualpoolsize; k++) 
	   { 
		   if(Pop->P[indexvals[j]][i]>Pop->P[indexvals[k]][i]) 
		   { auxindex = indexvals[j]; 
             indexvals[j] = indexvals[k]; 
			 indexvals[k] = auxindex; 
		   } 
       }   
	 } 
 
	 l = 0; 
	 k = 0; 
 
     while((l<MaxValues[i]) && (k< actualpoolsize)) 
	 { 
		 
		while (l<MaxValues[i] && Pop->P[indexvals[k]][i] > l) 
		{ 
		   if (l==0) ParentValues[i][l]=0; 
		   else ParentValues[i][l] = ParentValues[i][l-1]; 
		   l++; 
		} 
		 
		while ( (k<actualpoolsize)  && (Pop->P[indexvals[k]][i] == l) ) 
		{ 
             GenInd[i][g++] = indexvals[k]; 
			 k++; 
		} 
		ParentValues[i][l] = g-1; 
		l++; 
	} 
         
   } 
 
 delete[] indexvals; 
} 
 
 
 
