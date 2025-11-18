#include "auxfunc.h" 
#include "Popul.h" 
#include "Constraints.h" 
 
AdditiveConstraints::AdditiveConstraints(int vars,int munit, int mxunit, int typeconst, int thresh) 
{ 
	int i; 
	minunit = munit; 
    maxunit = mxunit; 
	typeconstraint = typeconst; 
	cantvars = vars; 
	threshold =  thresh; 
 
	unitdim = (maxunit-minunit+1); 
	UnitValues = new double[unitdim]; 
	 
	if (typeconstraint == 0)  
	{ 
     unitprob = new double*[1]; 
	 unitprob[0] = new double[cantvars]; 
	} 
	else 
    {	 
       unitprob = new double* [unitdim]; 
  	   for(i=0; i<unitdim; i++) 
   	    unitprob[i] = new double[cantvars]; 
	}  
 
	} 
 
void AdditiveConstraints::ResetUnitValues() 
{ 
	  int i,j; 
	  if (typeconstraint == 0)  
	  { 
    	for(j=0; j<unitdim; j++) UnitValues[j] = 0;       
	  } 
	  else 
      { 
		  for(i=0; i<unitdim; i++) 
           for(j=0; j<cantvars; j++) unitprob[i][j] = 1/double(cantvars);       
	  } 
 
} 
 
 
void AdditiveConstraints::InitUnitationValues() 
{ 
	int j,auxunit; 
	double alpha,totunit; 
 
	alpha = 0.0001; 
    UnitValues[0] = alpha; 
	totunit = alpha; 
 
	for(j=1;j<unitdim;j++) 
	{ 
	  auxunit = minunit+j; 
	  UnitValues[j] = UnitValues[j-1]* (cantvars - auxunit+1)/(auxunit); 
	  totunit += UnitValues[j]; 
	} 
 
	for(j=0;j<unitdim;j++) UnitValues[j] /= totunit; 
} 
 
void AdditiveConstraints::InitUnivariateMarginals() 
{ 
  int i,j; 
  if (typeconstraint == 0)  
  { 
	  for(j=0; j<cantvars; j++) unitprob[0][j] = 1/double(cantvars);       
  } 
  else 
  { 
  for(i=0; i<unitdim; i++) 
   for(j=0; j<cantvars; j++) unitprob[i][j] = 1/double(cantvars);       
  } 
} 
 
 
void AdditiveConstraints::GenInitIndividual (Popul* NewPop, int pos) 
{ 
  // The vector in position pos is generated 
double cutoff,tot,current_total; 
int i,j,tobeset; 
int* index; 
 
 
// First the unitation value o 
tobeset = Find_Unit_tobeset(); 
 
current_total = 1; 
 
if (tobeset==0 || tobeset== cantvars) 
{   
	for (i=0; i<cantvars; i++) NewPop->P[pos][i] = tobeset/cantvars; 
	return; 
} 
 
index = new int[cantvars]; 
for (i=0; i<cantvars; i++)  
{ 
	index[i] = i; 
	NewPop->P[pos][i] = 0; 
} 
// The following is generation with replacement 
i=0; 
while ( i< tobeset) 
{ 
  cutoff = myrand() * current_total;	 
  j = 0; 
  tot = 0; 
   if(cutoff>0) 
   { 
	 while ( (cutoff>tot) && (j<cantvars-i) ) 
	 {  
		 tot += unitprob[0][index[j]]; 
		 j++; 
     }  
	 j--; 
   } 
     current_total -=  unitprob[0][index[j]]; 
     NewPop->P[pos][index[j]] = 1; 
	 index[j] = index[cantvars-i-1]; 
	 i++; 
} 
  delete[] index; 
  return; 
} 
 
void AdditiveConstraints::GenInitPop(int From, Popul* NewPop) 
{  
	int i; 
    for(i=From; i<NewPop->psize; i++) 
	{ 
		GenInitIndividual (NewPop,i);  
	} 
} 
 
void AdditiveConstraints::InitPop(Popul* pop) 
{ 
  
  InitUnitationValues(); 
  InitUnivariateMarginals(); 
  GenInitPop(0,pop); 
 } 
 
 
AdditiveConstraints::~AdditiveConstraints() 
{ 
	int i; 
	delete[] UnitValues; 
	if (typeconstraint == 0)  delete[] unitprob[0]; 
	else for(i=0; i<unitdim; i++) delete[] unitprob[i];		 
	delete[] unitprob; 
} 
 
int AdditiveConstraints::Find_Unit_tobeset() 
{ 
 
double cutoff,tot; 
int j; 
 
// The unitation value of the new vector is selected 
cutoff = myrand();	 
while(cutoff==0) cutoff = myrand();	 
  j = 0; 
  tot = 0; 
	 while ( (cutoff>tot) && (j<unitdim) ) 
	 {  
		 tot += UnitValues[j]; 
		 j++; 
     }  
	  
return minunit + (j-1); 
 
} 
 
SConstraintUnivariateModel::SConstraintUnivariateModel(int vars, int* AllInd,int clusterkey,Popul* pop, int munit, int mxunit, int thresh):UnivariateModel(vars,AllInd,clusterkey,pop),AdditiveConstraints(vars,munit,mxunit,0,thresh) 
{ 
	AuxUnitValues = new double[unitdim]; 
} 
 
void SConstraintUnivariateModel::PopMutation(Popul* NewPop, int From,int MutType, double MutRate) 
{  
	int i,individual; 
	int MutatedIndividuals; 
 
     
    MutatedIndividuals = int((NewPop->psize - From)*NewPop->vars* MutRate/2); 
	 
    for(i=0; i<MutatedIndividuals; i++) 
	{ 
		individual = From + randomint(NewPop->psize - From); 
		IndividualMutation(NewPop,MutType,individual,1);  
	} 
} 
 
void SConstraintUnivariateModel::IndividualMutation(Popul* NewPop,int MutType,int individual, int MutatedGenes) 
{ // Mutacion swaping 
  int i,pickedgene1,pickedgene2,aux; 
	for(i=0; i<MutatedGenes; i++) 
	{ 
		pickedgene1 = randomint(NewPop->vars); 
        pickedgene2 = randomint(NewPop->vars); 
		aux = NewPop->P[individual][pickedgene1]; 
		NewPop->P[individual][pickedgene1] = NewPop->P[individual][pickedgene2]; 
		NewPop->P[individual][pickedgene2] = aux; 
     } 
} 
 
void SConstraintUnivariateModel::CalProb() 
{ 
   int i,j,l,k,auxunit,totunit,a,b; 
   totunit = 0; 
 
	// Univariate probabilities are calculated 
    for(j=0; j<length; j++) AllProb[j] = 0;       
	ResetUnitValues();       
	 
	if(actualpoolsize>0) 
	{ 
	 for(l=0; l<actualpoolsize; l++) 
	 { 
		i = actualindex[l]; 
		auxunit = 0; 
		for(j=0; j<length; j++)  
		{ 
			AllProb[j]+=Pop->P[i][j]; 
			auxunit+=Pop->P[i][j]; 
		} 
		UnitValues[auxunit-minunit]++;       
	 }   
  
	 auxunit = 0; 
	 for(j=0; j<unitdim; j++)  
	 { 
		 AuxUnitValues[j] = 0; 
		 a= ((j-threshold)>0)?j-threshold:0;  
		 b= ((j+threshold)>unitdim)?unitdim:j+threshold; 
 
        for(k=a; k<=b; k++)  // The unit prob. are thresheld 
		{ 
         AuxUnitValues[j] += UnitValues[k]; 
		 auxunit += UnitValues[k]; 
 		} 
	  	totunit += UnitValues[j]*(j+minunit); 
  	     
     } 
     for(j=0; j<length; j++) 
	 { 
		 unitprob[0][j] = AllProb[j] / double(totunit); 
		 AllProb[j] /= actualpoolsize;  
	 } 
	 for(j=0; j<unitdim; j++) UnitValues[j] =AuxUnitValues[j] / double(auxunit); 
	 
		  
	}  
} 
 
/* 
void SConstraintUnivariateModel::GenIndividual (Popul* NewPop, int pos) 
{ 
  // The vector in position pos is generated 
double cutoff,tot,current_total,reftotal; 
int i,j,tobeset; 
int* index; 
 
//printf("%d ",pos); 
// First the unitation value o 
tobeset = Find_Unit_tobeset(); 
 
current_total = 1; 
reftotal = 0; 
 
if (tobeset==0 || tobeset== length) 
{   
	for (i=0; i<length; i++) NewPop->P[pos][i] = tobeset/length; 
	return; 
} 
 
index = new int[length]; 
for (i=0; i<length; i++)  
{ 
	index[i] = i; 
	NewPop->P[pos][i] = 0; 
	reftotal+= unitprob[0][i]; 
} 
// The following is generation with replacement 
i=0; 
while ( i< tobeset) 
{ 
  cutoff = myrand() * current_total;	 
  j = 0; 
  tot = 0; 
   if(cutoff>0) 
   { 
	 while ( (cutoff>tot) && (j<length) ) 
	 {  
		 tot += unitprob[0][index[j]]; 
		 j++; 
     }  
	 j--; 
   } 
     if(NewPop->P[pos][index[j]] ==0) 
	 { 
+      NewPop->P[pos][index[j]] = 1; 
	  index[j] = j; 
	  i++; 
	 } 
 
} 
  delete[] index; 
  return; 
} 
 
*/ 
 
void SConstraintUnivariateModel::GenIndividual (Popul* NewPop, int pos) 
{ 
  // The vector in position pos is generated 
double cutoff,tot,current_total; 
int i,j,tobeset; 
int* index; 
 
 
// First the unitation value o 
tobeset = Find_Unit_tobeset(); 
 
current_total = 1; 
 
if (tobeset==0 || tobeset== length) 
{   
	for (i=0; i<length; i++) NewPop->P[pos][i] = tobeset/length; 
	return; 
} 
 
index = new int[length]; 
for (i=0; i<length; i++)  
{ 
	index[i] = i; 
	NewPop->P[pos][i] = 0; 
} 
// The following is generation with replacement 
i=0; 
while ( i< tobeset) 
{ 
  cutoff = myrand() * current_total;	 
  j = 0; 
  tot = 0; 
   if(cutoff>0) 
   { 
	 while ( (cutoff>tot) && (j<length-i) ) 
	 {  
		 tot += unitprob[0][index[j]]; 
		 j++; 
     }  
	 j--; 
   } 
     current_total -=  unitprob[0][index[j]]; 
     NewPop->P[pos][index[j]] = 1; 
	 index[j] = index[length-i-1]; 
	 i++; 
} 
  delete[] index; 
  return; 
} 
 
 
 
double SConstraintUnivariateModel::Prob (unsigned* vector) 
 { 
  // Se determina cual es la probabilidad del 
 // vector dado el arbol 
 
 double prob; 
 int j; 
 int auxunit; 
 
  prob = 1; 
  auxunit = 0; 
 
	for (j=0;j<length; j++ ) 
	 { 
	  auxunit += vector[j]; 
      if (vector[j]==1)  
	 	  prob *= AllProb[j];         
	  else prob *= (1-AllProb[j]); 
	  if (prob == 0) return 0; 
	 } 
	if(auxunit<minunit || auxunit>maxunit) return 0; 
	return (prob*UnitValues[auxunit-minunit]);  
} 
 
void SConstraintUnivariateModel::InitPop(Popul* pop) 
{ 
	AdditiveConstraints::InitPop(pop); 
 } 
 
 
	   
	 
SConstraintUnivariateModel::~SConstraintUnivariateModel() 
{ 
	delete[] AuxUnitValues; 
} 
 
 
CConstraintUnivariateModel::CConstraintUnivariateModel(int vars, int* AllInd,int clusterkey,Popul* pop,int munit, int mxunit, int thresh):UnivariateModel(vars,AllInd,clusterkey,pop),AdditiveConstraints(vars,munit,mxunit,1,thresh) 
{ 
} 
 
 
 
CConstraintUnivariateModel::~CConstraintUnivariateModel() 
{  
} 
 
 
void CConstraintUnivariateModel::CalProb() 
{ 
   int i,j,l,auxunit; 
	// Univariate probabilities are calculated 
    
     for(i=0; i<length; i++) 
	 { 
		 AllProb[i] = 0; 
	     for(j=0; j<unitdim; j++)  
		 { 
			 unitprob[j][i] = 0; 
             if (i==length-1)  UnitValues[j] = 0;   
		 }   
	 } 
        
   	if(actualpoolsize>0) 
	{ 
	 for(l=0; l<actualpoolsize; l++) 
	 { 
		i = actualindex[l]; 
		auxunit = 0; 
		for(j=0; j<length; j++)  
		{ 
			AllProb[j]+=Pop->P[i][j]; 
			auxunit+=Pop->P[i][j]; 
		} 
        for(j=0; j<length; j++)  
		{ 
			unitprob[auxunit-minunit][j]+=Pop->P[i][j]; 
		}  
		UnitValues[auxunit-minunit]++;       
	 }   
     for(i=0; i<length; i++) AllProb[i] /= actualpoolsize;  
     for(i=0; i<length; i++) 
	     for(j=0; j<unitdim; j++)  
		 { 
			 unitprob[j][i] /= ((j+minunit)*UnitValues[j]); 
             if (i==length-1)  UnitValues[j] /= actualpoolsize;   
		 }   
     }  
} 
 
 
void CConstraintUnivariateModel::GenIndividual (Popul* NewPop, int pos) 
{ 
  // The vector in position pos is generated 
double cutoff,tot,current_total; 
int i,j,tobeset; 
int* index; 
 
tobeset = Find_Unit_tobeset(); 
 
current_total = 1; 
 
if (tobeset==0 || tobeset== length) 
{   
	for (i=0; i<length; i++) NewPop->P[pos][i] = tobeset/length; 
	return; 
} 
 
index = new int[length]; 
for (i=0; i<length; i++)  
{ 
	index[i] = i; 
	NewPop->P[pos][i] = 0; 
} 
// The following is generation with replacement 
i=0; 
while ( i< tobeset) 
{ 
  cutoff = myrand() * current_total;	 
  j = 0; 
  tot = 0; 
  if(cutoff>0) 
  { 
	 while ( (cutoff>tot) && (j<length-i) ) 
	 {  
		 tot += unitprob[tobeset-minunit][index[j]]; 
		 j++; 
     }  
	 j--; 
  } 
     current_total -=  unitprob[tobeset-minunit][index[j]]; 
     NewPop->P[pos][index[j]] = 1; 
	 index[j] = index[length-i-1]; 
	 i++; 
} 
  delete[] index; 
  return; 
} 
 
 
double CConstraintUnivariateModel::Prob (unsigned* vector) 
 { 
  // Se determina cual es la probabilidad del 
 // vector dado el arbol 
 
 double prob; 
 int j; 
 int auxunit; 
 
  prob = 1; 
  auxunit = 0; 
 
   
	for (j=0;j<length; j++ ) auxunit += vector[j]; 
	if(auxunit<minunit || auxunit>maxunit) return 0; 
	 
	for (j=0;j<length; j++ )  
	 {	   
      if (vector[j]==1)  
	 	  prob *= AllProb[j];         
	  else prob *= (1-AllProb[j]); 
	  if (prob == 0) return 0; 
	 } 
 
	return (prob*UnitValues[auxunit-minunit]);  
} 
 
void CConstraintUnivariateModel::InitPop(Popul* pop) 
{ 
	AdditiveConstraints::InitPop(pop); 
 } 
 
 
void ConstraintBinaryTreeModel::CalProb() 
 { 
   int aux,h,i,j,k,l,auxunit; 
     
   ResetUnitValues(); 
 
   for(j=0; j<length-1; j++) 
	  { 
		AllProb[j]=0; 
 
		for(k=j+1 ; k<length; k++) 
			{ 
              aux = j*(2*length-j+1)/2 +k-2*j-1; 
			  AllSecProb[0][aux]=0; 
			  AllSecProb[1][aux]=0; 
			  AllSecProb[2][aux]=0; 
			  AllSecProb[3][aux]=0; 
         if(actualpoolsize>0) 
		 { 
			  for(l=0; l<actualpoolsize; l++) 
				  { 
					  //Se calcula la probabilidad de cada gen en genepool		 
				    i = actualindex[l]; 
					++AllSecProb[2*Pop->P[i][j]+Pop->P[i][k]][aux]; 
					auxunit = 0; 
					if (k==j+1) AllProb[j]+=Pop->P[i][j]; 
					if (j==0 && k==1) 
					{ 
						auxunit = 0; 
                        for(h=0; h<length; h++) auxunit+=Pop->P[i][h]; 
                        UnitValues[auxunit-minunit]++;    
					} 
				  } 
 
              AllSecProb[0][aux]= AllSecProb[0][aux] / actualpoolsize; 
			  AllSecProb[1][aux]= AllSecProb[1][aux] / actualpoolsize; 
              AllSecProb[2][aux]= AllSecProb[2][aux] / actualpoolsize; 
			  AllSecProb[3][aux]= AllSecProb[3][aux] / actualpoolsize; 
} 
 
			 } 
 
	 } 
 
  AllProb[length-1]=0; 
  for(j=0; j<unitdim; j++)  UnitValues[j] /= actualpoolsize;  
      
  
if (actualpoolsize >0) 
{ 
  for(i=0; i<actualpoolsize; i++) AllProb[length-1]+=Pop->P[actualindex[i]][length-1];  
  for(i=0; i<length; i++) AllProb[i] = AllProb[i] / actualpoolsize;  
 } 
} 
 
 
double ConstraintBinaryTreeModel::Prob (unsigned* vector) 
 { 
  // The probability of the vector given the tree 
 // is calculated 
 
 double auxprob,aux2,aux1,prob; 
 int aux,j,auxunit; 
 
 aux1= 0; 
 prob = 1; 
 auxunit = 0; 
 
	for (j=0;j<length; j++ ) auxunit += vector[j]; 
	if(auxunit<minunit || auxunit>maxunit) return 0; 
	 
	for (j=0;j<length; j++ ) 
	 { 
	  if (Tree[j]==-1) prob = (vector[j]==1)?prob*AllProb[j]:prob*(1-AllProb[j]);  
	  else  
	  {	   
		 if (j<Tree[j])  
		 { 
				 aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
	        	 aux2=(AllSecProb[2*vector[j]+vector[Tree[j]]][aux]+aux1); 
		 } 
		 else  
		 {  
 		 	      aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
				  aux2=(AllSecProb[2*vector[Tree[j]]+vector[j]][aux]+aux1); 
		 } 
		 
		 //auxprob= (vector[Tree[j]]==1)?(aux2/AllProb[Tree[j]]):(aux2/(1-AllProb[Tree[j]]));   
		 if( vector[Tree[j]]==1 && AllProb[Tree[j]]>0) auxprob = aux2/AllProb[Tree[j]]; 
		 else if( vector[Tree[j]]==0 && AllProb[Tree[j]] < 1) auxprob = aux2/(1-AllProb[Tree[j]]); 
		 else auxprob = 0; 
          
		 if (auxprob == 0) return 0; 
		 else prob*=auxprob; 
	  } 
	 
	} 
 
	return (prob*UnitValues[auxunit-minunit]);  
} 
 
ConstraintBinaryTreeModel::ConstraintBinaryTreeModel(int vars, int* AllInd,int clusterkey,Popul* pop, int munit, int mxunit ):BinaryTreeModel(vars,AllInd,clusterkey,pop,1.0), AdditiveConstraints(vars,munit,mxunit,0,0) 
 { 
	DoReArrangeTrees = 1; //By definition the generation is done by  
                    // selecting a new root any time an individual is generated.  
} 
 
 
ConstraintBinaryTreeModel::~ConstraintBinaryTreeModel() 
 { 
 } 
 
 
 
void ConstraintBinaryTreeModel::GenIndividual (Popul* NewPop, int pos) 
 { 
  // The vector in position pos is generated 
 // The generation has to fulfil the constraints 
 double auxprob,aux2,cutoff,aux1; 
 int aux,j,i,tobeset,auxunit,set_to0,set_to1; 
  
   if (DoReArrangeTrees) ReArrangeTree(); 
   tobeset = Find_Unit_tobeset(); 
   auxunit = 0; 
     
 	for (i=0;i<length; i++ ) 
	 { 
       
	  set_to1 = (i== length + auxunit -tobeset); 
      set_to0 = (auxunit == tobeset); 
 
	  if (set_to1)  
		  while(i<length)  
		  { 
		   j = NextInOrder(i); 
		   NewPop->P[pos][j] = 1; 
		   auxunit++; 
		   i++; 
		  } 
      else  
	   if (set_to0)  
		  while(i<length)  
		  { 
		   j = NextInOrder(i); 
		   NewPop->P[pos][j] = 0; 
		   i++; 
		  } 
	   else 
	   {    
        j = NextInOrder(i); 
	    cutoff = myrand(); 
	    if (Tree[j]==-1)  
		{ 
		 if (cutoff > AllProb[j]) NewPop->P[pos][rootnode]=0; 
		 else NewPop->P[pos][rootnode]=1; 
		 auxunit += NewPop->P[pos][rootnode]; 
		}  
	    else  
		{	   
	     if (NewPop->P[pos][Tree[j]]==1) 
		 { 
	   	  if (j<Tree[j])  
		  { 
		   aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
		   aux2=(AllSecProb[3][aux]+aux1); 
		  } 
		  else  
		  { 
 		   aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		   aux2=(AllSecProb[3][aux]+aux1); 
		  } 
		  auxprob=aux2/AllProb[Tree[j]]; 
		 } 
		else 
		{ 
         if(j<Tree[j])  
		 { 
		  aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
		  aux2=(AllSecProb[2][aux]+aux1); 
		 } 
		 else  
		 { 
		  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		  aux2=(AllSecProb[1][aux]+aux1); 
		 } 
		auxprob=aux2/(1-AllProb[Tree[j]]); 
		} 
		if (cutoff > auxprob) NewPop->P[pos][j]=0; 
  	    else NewPop->P[pos][j]=1; 
		auxunit += NewPop->P[pos][j]; 
	  } 
	 } 
	}// end of the for 
		//printf("auxunit =  %d", auxunit); 
} 
 
void ConstraintBinaryTreeModel::InitPop(Popul* pop) 
{ 
	AdditiveConstraints::InitPop(pop); 
 } 
