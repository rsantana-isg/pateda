

class IntegerTreeModel: public AbstractTreeModel 
 { 
	 public: 
      
	  unsigned *MaxValues; // Maxima cantidad de valores para cada variable 
	  int **GenInd; 
	  int **ParentValues; 
	  unsigned Maximum_among_Minimum; 
 
	   
	  IntegerTreeModel(int,int*,int,Popul*); 
	 virtual ~IntegerTreeModel(); 
 
         virtual void ProcessPop(Popul*);	  
	 virtual void CalMutInf(); 
	 virtual void UpdateModel(double*); 
	 virtual void UpdateModel(); 
	 void CalMutInf_from_ProbFvect(double*); 
	 virtual double Prob(unsigned*); 
	 virtual void GenIndividual (Popul*,int); 
	 int UnivFreq(int,unsigned); 
	 double UnivProb(int, unsigned); 
	 double UnivProb_from_ProbFvect(int,unsigned,double*); 
	 double CondProb(int, unsigned,int, unsigned); 
	 void MakeProbStructures();  
	 unsigned SonValue(int,unsigned,int); 
	 virtual void SetPop(Popul*);     
 }; 
 
 
 
class IntegerUnivModel: public IntegerTreeModel 
 { 
	 public: 
      
	  IntegerUnivModel(int,int*,int,Popul*); 
	  virtual ~IntegerUnivModel(){}; 
     virtual void ProcessPop(Popul*); 
	 virtual void MakeTree(int);	  
	 virtual void UpdateModel(double* vector); 
	 virtual void UpdateModel(); 
	 }; 
 




IntegerTreeModel::IntegerTreeModel(int vars, int* AllInd, int clusterkey,Popul* pop):AbstractTreeModel(vars,AllInd,clusterkey,pop,1.0) 
 { 
  int i; 
  MaxValues = pop->dim; // It is a pointer to save memory 
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
 
 
double IntegerTreeModel::Prob (unsigned* vector) 
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
 
 
IntegerTreeModel::~IntegerTreeModel() 
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
 
double IntegerTreeModel::UnivProb(int var, unsigned varval) 
{ 
  int Freq; 
  if(ParentValues[var][varval] == ForbidValue) return 0; 
   
  if(varval==0 || ParentValues[var][varval-1] == ForbidValue )  
	              Freq = ParentValues[var][varval] + 1; 
  else 
  { 
	  Freq = ParentValues[var][varval] - ParentValues[var][varval-1]; 
  } 
  return Freq/double(actualpoolsize); 
} 
 
void IntegerTreeModel::ProcessPop(Popul* pop) 
{ 
	SetPop(pop); 
    rootnode = RandomRootNode(); 
	MakeProbStructures(); 
    CalMutInf(); 
    MakeTree(rootnode); 
} 
 
double IntegerTreeModel::UnivProb_from_ProbFvect(int var, unsigned varval,double* vector) 
{ 
  double prob; 
  prob = 0; 
   int initindex,endindex,i;    
 
   if(ParentValues[var][varval] == ForbidValue) return 0; 
 
	if(varval == 0 || ParentValues[var][varval-1] ==ForbidValue) 
	{  
	  initindex = 0; 
	  endindex = ParentValues[var][varval]; 
	} 
	else  
	{ 
	  initindex = ParentValues[var][varval-1] + 1; 
	  endindex = ParentValues[var][varval]; 
	} 
	if (endindex==(initindex-1)) return 0; // This case should not happen 
 	 
 
	for (i=initindex;i<= endindex; i++ ) 
	{ 
		  int auxind; 
		  auxind = GenInd[var][i]; 
          auxind = Pop->P[auxind][var]; 
	      if (auxind) prob += (vector[auxind]); 
	} 
  return prob; 
} 
 
 
double IntegerTreeModel::CondProb(int parent, unsigned parentval, int son, unsigned sonval) 
{ 
   int auxval; 
   int initindex,endindex,i;    
    
   if(ParentValues[parent][parentval] == ForbidValue || ParentValues[son][sonval] == ForbidValue ) return 0; 
 
   if(parentval == 0 || ParentValues[parent][parentval-1] ==ForbidValue) 
	{  
	  initindex = 0; 
	  endindex = ParentValues[parent][parentval]; 
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
 
unsigned IntegerTreeModel::SonValue(int parent, unsigned parentval, int son) 
{ 
	int CantSons,auxval; 
	 
	if(ParentValues[parent][parentval] == ForbidValue) return ForbidValue; 
	if(parentval == 0 || ParentValues[parent][parentval-1] == ForbidValue) CantSons = ParentValues[parent][parentval] + 1; 
	else CantSons = ParentValues[parent][parentval] - ParentValues[parent][parentval-1]; 
	if (CantSons==0) return ForbidValue; 
	auxval = randomint(CantSons); 
	auxval = GenInd[parent][ParentValues[parent][parentval]-auxval]; 
    return Pop->P[auxval][son]; 
} 
 
int IntegerTreeModel::UnivFreq(int variable, unsigned val) 
{ 
	if(ParentValues[variable][val] == ForbidValue) return 0; 
	if(val == 0 || ParentValues[variable][val-1] == ForbidValue ) return ParentValues[variable][val] + 1; 
	return ParentValues[variable][val] - ParentValues[variable][val-1]; 
}	  
 
 
void IntegerTreeModel::GenIndividual (Popul* NewPop, int pos) 
 { 
  // The vector in position pos is generated 
  
 int auxval,j,i; 
  
    if (DoReArrangeTrees) ReArrangeTree(); 
 
 	for (i=0;i<length; i++ ) 
	 { 
      j = NextInOrder(i); 
	 
	  if (Tree[j]==-1)  
	  { 
		  auxval = randomint(actualpoolsize); 
	      NewPop->P[pos][j] = Pop->P[actualindex[auxval]][j]; 
	  } 
	  else  
	      NewPop->P[pos][j] =  SonValue(Tree[j],NewPop->P[pos][Tree[j]],j); 
	  } 
} 
 
 
void IntegerTreeModel::CalMutInf() 
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
  		      if (BivProb > 0)  
                MutualInf[aux]+=BivProb*(log(BivProb/(UnivProb_j*UnivProb_k))); 
			} 
		} 
		delete[] indexpairs; 
  }	           			 
 
void IntegerTreeModel::SetPop(Popul* pop) 
{ 
	Pop = pop; 
} 
 
void IntegerTreeModel::MakeProbStructures() 
{ 
	int i,j,g,k,l,auxindex; 
	int *indexvals; 
	indexvals = new int[actualpoolsize]; 
   for(i=0; i<length; i++) 
   {for(k=0; k<MaxValues[i]; k++) ParentValues[i][k]= ForbidValue;} 
 
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
		   if (l==0) ParentValues[i][l]=ForbidValue; 
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
 
void IntegerTreeModel::UpdateModel(double *vector) 
{ 
  CalMutInf_from_ProbFvect(vector); 
  rootnode = RandomRootNode(); 
  MakeTree(rootnode); 
 } 
 
void IntegerTreeModel::UpdateModel() 
{ 
  CalMutInf(); 
  rootnode = RandomRootNode(); 
  MakeTree(rootnode); 
 } 
 
 
 
void IntegerTreeModel::CalMutInf_from_ProbFvect(double *vector) 
 { 
  // The Mutual Information Matrix is constructed 
 
 
   int i,k,j; 
   int aux; 
   unsigned current_pair[2]; 
   int *indexpairs; 
   int remaining_configurations; 
   double BivProb, UnivProb_j,UnivProb_k; 
 
       
	indexpairs = new int[genepoollimit];        
 
	 for(j=0; j<length-1; j++)         // For all possible pairs of variables 
	  	for(k=j+1 ; k<length; k++) 
		{ 
         aux = j*(2*length-j+1)/2 +k-2*j-1; 
		 MutualInf[aux]=0; 
      	  
	     for(i=0; i<genepoollimit; i++) indexpairs[i] = i; 
		 remaining_configurations = genepoollimit; 
 	     while(remaining_configurations > 0) 
          { 
			    BivProb = 0;                
				current_pair[0] = Pop->P[indexpairs[0]][j]; 
				current_pair[1] = Pop->P[indexpairs[0]][k]; 
 
			    i = 0; 
			    while(i< remaining_configurations) 
				{  
                 if( current_pair[0] == Pop->P[indexpairs[i]][j] &&  
				   current_pair[1] == Pop->P[indexpairs[i]][k] )  
				 { 
                  BivProb += vector[i]; 
				  indexpairs[i] = indexpairs[remaining_configurations-1]; 
				  remaining_configurations--; 
				 } 
			     else i++; 
				} 			     
			   
			  UnivProb_j = UnivProb_from_ProbFvect(j,current_pair[0],vector); 
			  UnivProb_k = UnivProb_from_ProbFvect(k,current_pair[1],vector); 
  		      if (UnivProb_j*UnivProb_k>0 && BivProb > 0.000000001)  
                MutualInf[aux]+=BivProb*(log(BivProb/(UnivProb_j*UnivProb_k))); 
			} 
		} 
		delete[] indexpairs; 
  }	           			 
 
void IntegerUnivModel::MakeTree(int rootn) 
 { 
 	int i; 
	 for(i=0; i<length; i++)  
	 { 
		 Tree[i] = -1; 
		 Queue[i] = i; 
	 } 
 }	 
 
void IntegerUnivModel::UpdateModel() 
{ 
  //MakeTree(0); 
} 
 
void IntegerUnivModel::UpdateModel(double *vector) 
{ 
  //MakeTree(rootnode); 
} 
 
void IntegerUnivModel::ProcessPop(Popul* pop) 
{ 
	SetPop(pop); 
    rootnode = 0; 
	MakeProbStructures(); 
    MakeTree(rootnode); 
} 
 
IntegerUnivModel::IntegerUnivModel(int vars, int* AllInd, int clusterkey,Popul* pop):IntegerTreeModel(vars,AllInd,clusterkey,pop) 
 { 
 } 
 
