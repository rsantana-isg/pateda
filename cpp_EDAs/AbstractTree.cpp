#include <iostream> 
#include <fstream>
#include "auxfunc.h" 
#include "Popul.h" 
//#include "Treeprob.h" 
#include "AbstractTree.h" 
#include <math.h> 
#include <time.h> 
 
extern int now; 
extern FILE* outfile; 
 
using namespace std;
  
 
AbstractProbModel::AbstractProbModel(int vars, int* AllInd,int clusterkey,Popul* pop) 
 { 
  int i,j; 
  length = vars;  
  Pop = pop; 
  actualpoolsize =Pop->psize; 
  Prior = 1;
  genepoollimit = Pop->psize; 
  for(i=0; i < genepoollimit; i++) actualpoolsize += ( AllInd[i] == clusterkey ); 
  actualindex = new int[actualpoolsize]; 
  
  j = 0; 
  for(i=0; i < genepoollimit; i++)  
  { 
	 if ( AllInd[i] == clusterkey ) actualindex[j++] = i; 
  } 
   
 } 
 
AbstractProbModel::AbstractProbModel(int vars, int apsize) 
 { 
  Prior = 1;   
  length = vars;  
  actualpoolsize = apsize;  
  genepoollimit = 0; 
  toclean = 0;
  actualindex = (int *) 0;
 } 

 AbstractProbModel::AbstractProbModel(int vars) 
 { 
  Prior = 1;   
  length = vars;  
  genepoollimit = 0; 
  toclean = 0;
  actualindex = (int *) 0;
 } 


void AbstractProbModel::SetGenePoolSize(int size) 
{ 
  genepoollimit= size;   
}   
 
void AbstractProbModel::SetPop(Popul* pop)
 { 
	 Pop = pop;
 }
 
void AbstractProbModel::GenPop(int From, Popul* NewPop) 
{  
	int i; 
    for(i=From; i<NewPop->psize; i++) 
	{ 
		GenIndividual (NewPop,i);  
	} 
} 
 
void AbstractProbModel::InitPop(Popul* pop) 
{  
  pop->RandInit(); 
} 
 
 
void AbstractProbModel::SetGenPoolLimit(int gpl) 
{ 
    //int i; 
  genepoollimit = gpl; 
  //actualpoolsize = 0; //This has to be fixed with the constructor 
  //Para considerar genotype clustering debe cambiarse 
  //for(i=0; i < genepoollimit; i++) actualpoolsize += ( AllInd[i] == clusterkey ); 
} 
 
 
 void AbstractProbModel::PopMutation(Popul* NewPop, int From,int MutType, double MutRate) 
{  
	int i,individual; 
	int MutatedIndividuals; 
 
     
    MutatedIndividuals = int((NewPop->psize - From)*NewPop->vars* MutRate); 
	 
    for(i=0; i<MutatedIndividuals; i++) 
	{ 
		individual = From + randomint(NewPop->psize - From); 
		IndividualMutation(NewPop,MutType,individual,1);  
	} 
} 
 
 
void AbstractProbModel::IndividualMutation(Popul* NewPop,int MutType,int individual, int MutatedGenes) 
{ // Mutacion binaria 
  int i,pickedgene; 
	for(i=0; i<MutatedGenes; i++) 
	{ 
		pickedgene = randomint(NewPop->vars); 
		NewPop->P[individual][pickedgene] = 1 - NewPop->P[individual][pickedgene]; 
     } 
} 
 
AbstractProbModel::~AbstractProbModel() 
 { 
   if (actualindex != (int *)0) delete[] actualindex; 
 } 
 
double AbstractProbModel::Calculate_Best_Prior(int N, int dim, double penalty) 
{ 
  double prior; 
 
   prior  = (penalty*N)/(length*pow(2,dim-1)); 
   return prior;  
 
} 
 
     
 
 
 

void AbstractProbModel::PutPriors(int Prior,int npoints,double valprior)
{   
    // double univ_prior,biv_prior;
    //actualpoolsize = npoints;       
           switch(Prior) 
                     { 
                       case 0: break; // No prior 
			 case 1: SetPrior(1,1,npoints);				                        
                                     break; // Recommended prior                      
                     } 
	   //cout<<"here is**********************   "<<npoints<<endl;
}
  

UnivariateModel::UnivariateModel(int vars, int* AllInd,int clusterkey,Popul* pop):AbstractProbModel(vars,AllInd,clusterkey,pop) 
{ 
    AllProb = new double[length];  
} 

UnivariateModel::UnivariateModel(int vars, int* AllInd,int clusterkey):AbstractProbModel(vars) 
{ 
    AllProb = new double[length];  
} 

UnivariateModel::UnivariateModel(int vars):AbstractProbModel(vars) 
{ 
    AllProb = new double[length];  
} 
 
void UnivariateModel::UpdateModel() 
{ 
 CalProb(); 
} 
 
void UnivariateModel::SetPrior(double prior_r_univ,double prior_r,int N) 
 { 
   int j; 
    
    	 for(j=0; j<length; j++) 
	  { 
	    AllProb[j]= (AllProb[j]*N+prior_r_univ)/ (N+ 2*prior_r_univ); 
          } 
  
} 

void UnivariateModel::CalProb(Popul* pop, int Npoints) 
{ 
   int i,j,l; 
	// Univariate probabilities are calculated 
    for(j=0; j<length; j++) AllProb[j]=1;       
     for(l=0; l<Npoints; l++) 
	 { 
	   i =  l; //actualindex[l]; 
		for(j=0; j<length; j++) AllProb[j]+= pop->P[i][j]; 
	 }   
     //for(j=0; j<length; j++) cout<<" j="<<j<<" P="<<AllProb[j];
     // cout<<endl;  

     for(j=0; j<length; j++) AllProb[j] = AllProb[j] / (Npoints+2);  

     //for(j=0; j<length; j++) cout<<" j="<<j<<" P="<<AllProb[j];
     //cout<<endl;

} 



void UnivariateModel::CalProb() 
{ 
   int i,j,l; 
	// Univariate probabilities are calculated 
    for(j=0; j<length; j++) AllProb[j]=1;       
    cout<<actualpoolsize<<endl;
	 for(l=0; l<actualpoolsize; l++) 
	 { 
	   i =  l; //actualindex[l]; 
		for(j=0; j<length; j++) AllProb[j]+= Pop->P[i][j]; 
	 }   
     for(j=0; j<length; j++) AllProb[j] = AllProb[j] / (actualpoolsize+2);  

     //for(j=0; j<length; j++) cout<<" j="<<j<<" P="<<AllProb[j];
     //	 cout<<endl;
	 


} 
 
void UnivariateModel::CalProbFvect(double *vector) 
 { 
   int i,j,l; 
   // Univariate probabilities are calculated 
   // from a vector of joint probabilities 
    
    for(j=0; j<length; j++) AllProb[j]=0;       
	if(actualpoolsize>0) 
	 for(l=0; l<actualpoolsize; l++) 
	 { 
		i = actualindex[l]; 
		for(j=0; j<length; j++)  
			if (Pop->P[i][j]==1) AllProb[j] += vector[i]; 
	 }   
} 

void UnivariateModel::CalProbFvect(Popul* pop, double *vector, int npoints) 
 { 
   int i,j,l; 
   // Univariate probabilities are calculated 
   // from a vector of joint probabilities 
    
    for(j=0; j<length; j++) AllProb[j]=0;       
	
	 for(l=0; l<npoints; l++) 
	 { 
	     i = l;  //actualindex[l]; 
		for(j=0; j<length; j++)  
			if (pop->P[i][j]==1) AllProb[j] += vector[i]; 
	 }   
	 //for(j=0; j<length; j++) cout<<" j="<<j<<" P="<<AllProb[j];
	 // for(j=0; j<length; j++) cout<<" "<<AllProb[j]<<" ";
         //cout<<endl;
}  

void UnivariateModel::GenIndividual (Popul* NewPop, int pos) 
{ 
  // The vector in position pos is generated 
 double cutoff; 
 int i; 
  	for (i=0;i<length; i++ ) 
	 { 
          cutoff = myrand(); 
	  if (cutoff > AllProb[i]) NewPop->P[pos][i]=0; 
	  else NewPop->P[pos][i]=1; 
     }  
} 
 
void UnivariateModel::ResetProb() 
{ 
  int j; 
  for(j=0; j<length; j++) AllProb[j]=0;       
} 
 
double UnivariateModel::Prob (unsigned* vector) 
 { 
  // Se determina cual es la probabilidad del 
 // vector dado el arbol 
 
 double prob; 
 int j; 
 
  prob = 1; 
 
	for (j=0;j<length; j++ ) 
	 { 
      if (vector[j]==1) prob *= AllProb[j]; 
	  else prob *= (1-AllProb[j]); 
	  if (prob == 0) return 0; 
	 } 
	 
	return prob;  
} 
 
UnivariateModel::~UnivariateModel() 
{ 
    delete[] AllProb;  
} 
 
 
AbstractTreeModel::AbstractTreeModel(int vars, int* AllInd,int clusterkey,Popul* pop, double complexity):AbstractProbModel(vars,AllInd,clusterkey,pop) 
 { 
  int i,which,df; 
  Complexity=complexity;
  MutualInf = new double [length*(length-1)/2]; 
  Tree = new int[length]; 
  Queue =  new int [length]; 
  DoReArrangeTrees = 0; 
  for(i=0; i<length; i++)  
   { 
    Tree[i] = -1;  
    Queue[i] = i; 
   } 

    which = 2; //Calculate the chi for a given prob  
    df = 1;  
    threshchival = FindChiVal(complexity,which,df); 
 } 
  
AbstractTreeModel::AbstractTreeModel(int vars,double complexity, int apsize):AbstractProbModel(vars,apsize) 
 { 
  int i,df,which; 
  Complexity=complexity;
  MutualInf = new double [length*(length-1)/2]; 
  Tree = new int[length]; 
  Queue =  new int [length]; 
  DoReArrangeTrees = 0; 
  for(i=0; i<length; i++)  
   { 
    Tree[i] = -1;  
    Queue[i] = i; 
   }   
which = 2; //Calculate the chi for a given prob  
    df = 1;  
    threshchival = FindChiVal(complexity,which,df); 

 } 
 
  
AbstractTreeModel::AbstractTreeModel(int vars,double complexity):AbstractProbModel(vars) 
 { 
  int i,df,which; 
  Complexity=complexity;
  MutualInf = new double [length*(length-1)/2]; 
  Tree = new int[length]; 
  Queue =  new int [length]; 
  DoReArrangeTrees = 0; 
  for(i=0; i<length; i++)  
   { 
    Tree[i] = -1;  
    Queue[i] = i; 
   } 
    which = 2; //Calculate the chi for a given prob  
    df = 1;  
    threshchival = FindChiVal(complexity,which,df); 
 } 
 




int  AbstractTreeModel::Included_Edge(int a, int b) 
{ 
  return ( (Tree[a]==b) || (Tree[b]==a)); 
} 
 
 
 void  AbstractTreeModel::Reorder_Edges(int a, int b) 
{              
  int i,aux,fa; 
    
  i = Tree[b]; 
  Tree[b] = a; 
  fa = b; 
  aux = 0; 
 
  while (aux != -1) 
    { 
     aux = Tree[i]; 
     Tree[i] = fa; 
     fa = i; 
     i = aux; 
    } 
} 

int  AbstractTreeModel::Has_descendants(int a) 
{ int i; 
 i=0;
 while(i<length)
 {
  if (Tree[i]==a) return 1; 
  i++;
 }
 return 0;
}

int  AbstractTreeModel::Oldest_Ancestor(int a) 
{ int i; 
 
  i = a; 
  while (Tree[i] != -1) i = Tree[i];  
  return i; 
} 

int  AbstractTreeModel::Other_root(int a) 
{ 
 int i; 
   i = 0; 
   while ( i<length)
   {
       if (Tree[i]==-1 && i!=a) return i;
    i++;
   }
  return -1; 
}

int  AbstractTreeModel::Edge_Cases(int a, int b) 
{   
  int olda, oldb; 
  /* This function gives the position of an edge related to a tree. These are:
   -1 : The edge is in the tree
   -2 : The two vertices are connected in the tree (although not directly)
    0 : The two vertices have parents and belong to different connected components
    1 : The two vertices are independent
    2 : The first is independent, the second not, the first is set as son of the second
    3 : The second is independent, the first not, the second is set as son of the first 
    4 : Both are dependent, the second b  is set as son of the first a
    5 : Both are dependent, the first a is set as son of the first b
  */

  if(Included_Edge(a,b)) return -1; // The edge is already in the tree
  
 if( (Tree[a] != -1 ) && (Tree[b] != -1 )  ) // Both variables are in the tree 
   { 
     olda = Oldest_Ancestor(a);  
     oldb = Oldest_Ancestor(b);      
     if (olda==oldb) return -2;     // Already connected   
     else 
     {  
      if(a<b) return 6;
      else return 7;
     }  // Different connected components (The tree has to be re-ordered)
   } 
 else  
   if( (Tree[a] == -1) && (Tree[b] == -1)  && !Has_descendants(a) && !Has_descendants(b) ) // None of them is in the tree 
   return 1;
 else 
  if( Tree[a] == -1 && Other_root(a)>-1 && !Has_descendants(a) && ( (Tree[b] != -1)|| Has_descendants(b)) ) 
   return 2;
 else 
  if( (Tree[a] != -1 || Has_descendants(a) )  && (Tree[b] == -1 &&  Other_root(b)>-1 && !Has_descendants(b)) ) 
   return 3;
 else 
  if( Tree[a] == -1 && Other_root(a)>-1  && Has_descendants(a) && a!=Oldest_Ancestor(b) && ( Tree[b] != -1 || Has_descendants(b) || Other_root(b)==-1) ) 
   return 4;
 if( (Tree[a] != -1 || Has_descendants(a)|| Other_root(a)==-1)  && Tree[b] == -1 && Other_root(b)>-1  && Has_descendants(b) && b!=Oldest_Ancestor(a)) 
   return 5;

   return -1; 

} 

int  AbstractTreeModel::Correct_Edge(int a, int b) 
{   
   if(Included_Edge(a,b)) return -1; // The edge is already in the tree
  if(Oldest_Ancestor(a) == Oldest_Ancestor(b)) return -2;
  else if( (Tree[a] != -1 ) && (Tree[b] != -1 )  )
     {  
      if(a<b) return 0;
      else return 1;
     }  // Different connected components (The tree has to be re-ordered)
 else 
  if( Tree[a] == -1 && rootnode!=a )  return 2;
 else 
  if( Tree[b] == -1 && rootnode!=b )  return 3;
  
   return -1; 

} 


int  AbstractTreeModel::Add_Edge(int a, int b) 
{   
  int olda, oldb,auxroot; 
  
  if(Included_Edge(a,b)) return -1; // The edge is already in the tree
  
 if( (Tree[a] != -1) && (Tree[b] != -1)) // Both variables are in the tree 
   { 
     olda = Oldest_Ancestor(a);  
     oldb = Oldest_Ancestor(b);      
     if (olda==oldb) return -1;        //If they no have common ancestor 
     else                              // can be connected 
       { 
         if (a<b) Reorder_Edges(a,b); 
         else  Reorder_Edges(b,a);           
       } 
   } 
 else  // It is assumed that it is possible to insert the edge
   if( (Tree[a] == -1) && (Tree[b] == -1)) // None of them is in the tree 
   {   
       if (a!=rootnode)  Tree[a] = b;
       else    Tree[b] = a;   
   } 
 else 
  if( (Tree[a] == -1 ) && (Tree[b] != -1)) 
   { 
     oldb = Oldest_Ancestor(b); 
     if (a==oldb) return -1; 
     else 
      {
        if(a==rootnode)
         {
          auxroot = Other_root(a);
          if (auxroot==-1) return -1;
          else  rootnode =  auxroot;
         }
          Tree[a] = b;        
      }                   
   } 
   else  if( (Tree[b] == -1) && (Tree[a] != -1)) 
    { 
     olda = Oldest_Ancestor(a); 
     if (b==olda) return -1; 
     else 
      {
        if(b==rootnode)
         {
          auxroot = Other_root(b);
          if (auxroot==-1) return -1;
          else rootnode =  auxroot;
         } 
	Tree[b] = a;                              
      } 
    }
return 1; 
 
} 
 
 
 
// This function reorder the direction of edges when a new 
// edge has been added between non connected components. 
 
 
 int AbstractTreeModel::RandomRootNode() 
 { 
  return randomint(length); 
 } 
 
 void AbstractTreeModel::ReArrangeTree() 
 { 
	// This function changes the current root of 
	// a tree but without changing the structure 
 
  int oldparent, newparent, aux; 
  oldparent = RandomRootNode(); 
  rootnode = oldparent; 
  newparent = -1; 
  while (oldparent != -1) 
  {  
	  aux = Tree[oldparent]; 
	  Tree[oldparent] = newparent; 
	  newparent = oldparent; 
	  oldparent = aux; 
  } 
  ArrangeNodes(); 
} 
 
/* 
void AbstractTreeModel::MakeTree(int rootn) 
 { 
  // En cada paso se incorpora el nodo que no estando en el arbol 
  // tiene el mayor valor de informacion mutua con alguno de los 
  // nodos que ya estan en el arbol, el cual sera ademas su padre 
 
	double max,threshhold,auxm; 
	int maxsonindex; 
	int maxfatherindex; 
	int i,j,k,aux; 
 
         maxsonindex=0; 
         maxfatherindex=0; 
 
	 for(i=0; i<length; i++) Tree[i]=i; 
	 Tree[rootn]=-1; 
	 threshhold=-100;//0.005; 
 
 
	for(i=0; i<length-1; i++)  // Para los n-1 nodos que faltan por incorporar 
	 { 
		max=-10; 
		for(j=0; j<length; j++) 
		 for(k=0; k<length; k++) 
		  { 
			 if (Tree[j]==j && Tree[k]!=k ) 
				  { 
 
					  aux = j*(2*length-j+1)/2 +k-2*j-1; 
					   
					  if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1; 
					  else aux = k*(2*length-k+1)/2 +j-2*k-1; 
					   
					  auxm=MutualInf[aux]; 
					   
					  if (auxm>max) 
						 { 
								maxsonindex=j; 
								maxfatherindex=k; 
								max=auxm; 
						 } 
				  } 
 
		  } 
		 if (max>=threshhold) Tree[maxsonindex]=maxfatherindex; 
		 else Tree[maxsonindex]=-1; 
	 } // Llegado este punto se supone que todos los nodos esten en el arbol 
  ArrangeNodes(); 
 }	 
*/ 
 
 
void AbstractTreeModel::MakeTree(int rootn) 
 { 
   // Constructs a tree with a maximum number of nodes connected. 
 
	double max,threshhold,auxm; 
	int maxsonindex; 
	int maxfatherindex; 
	int i,j,k,jj,kk,aux; 
        int* auxsample;

         maxsonindex=0; 
         maxfatherindex=0; 
         TreeL = 0;
        
         //PrintMut();
         auxsample = new int[length];
        
	 for(i=0; i<length; i++)
         {
           Tree[i]=i; 
           auxsample[i] = i;
         }
         
          RandomPerm(length,length,auxsample); 
	  //for(i=0; i<length; i++) cout<<auxsample[i]<<" ";
          //cout<<endl;

	 Tree[rootn]=-1; 
	 threshhold=-100;//0.005; 
 

 
	for(i=0; i<length-1; i++)  
	 { 
		max=-1000; 
                //i = auxsample[ii];
		for(jj=0; jj<length; jj++)
		{
                 j = auxsample[jj];
		 for(kk=0; kk<length; kk++) 
		  { 
		    k = auxsample[kk];
                                if (Tree[j]==j && Tree[k]!=k ) 
				    { 
 
					  aux = j*(2*length-j+1)/2 +k-2*j-1; 
					   
					  if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1; 
					  else aux = k*(2*length-k+1)/2 +j-2*k-1; 
					   
					  auxm=MutualInf[aux]; 
					  //if (length==28) cout<<" auxm "<<auxm<<" k "<<k<<" j "<<j<<endl; 
					  if (auxm>=max) 
						 { 
								maxsonindex=j; 
								maxfatherindex=k; 
								max=auxm;
								//if (length==28) cout<<" max "<<max<<" maxfatind "<<k<<" maxsonind "<<j<<endl;
								//cout<<" max "<<max<<" maxfatind "<<k<<" maxsonind "<<j<<endl;
						 }
					     else if ((max-auxm)<0.0000000001 &&  myrand()>0.5) 
                                           { 
								maxsonindex=j; 
								maxfatherindex=k; 
								max=auxm;
								//if (length==28) cout<<" max "<<max<<" maxfatind "<<k<<" maxsonind "<<j<<endl;               
								}
				  } 
 
		  }
		} 
	
		if (max>threshhold &&  2*genepoollimit*max>threshchival)
                   {
		       // cout<<"*****************************************"<<endl;
		       // cout<<i<<" max "<<max<<" threshchi "<<threshchival<<" genepool "<<genepoollimit<<" other "<< 2*genepoollimit*max<<endl;    
                    Tree[maxsonindex]=maxfatherindex;
                    TreeL += max;
		    //if (length==28) cout<<i<<" max "<<max<<" maxfatind "<<maxfatherindex<<" maxsonind "<<maxsonindex<<endl;             
                   } 
                else Tree[maxsonindex]=-1; 
	 }  
 
	delete[] auxsample;
	ArrangeNodes(); 
 }	 
 






 



void AbstractTreeModel::MakeRandomTree(int rootn) 
 { 
   
	int* Unconnected; 
	int* Connected; 
	int i, conn, unconn, uind, cind; 
 
	Unconnected = new int[length]; 
    Connected = new int[length]; 
    for(i=0; i<length; i++) Tree[i]=i; 
	 
	Connected[0] = rootn; 
	Tree[rootn] = -1; 
	conn = 1; 
 
    for(i=0; i<length; i++) Unconnected[i] = i; 
    Unconnected[rootn] = Unconnected[length-1]; 
	unconn = length - 1; 
	 
	while (unconn>0)
	{ 
	  uind = randomint(unconn); 
	  cind = randomint(conn); 
      Connected[conn++] = Unconnected[uind]; 
	  Tree[Unconnected[uind]] = Connected[cind]; 
	  Unconnected[uind] = Unconnected[-1 + unconn-- ]; 
	} 
 
	delete[] Unconnected; 
    delete[] Connected; 
 
        for(i=0; i<length; i++) if(Tree[i]==i) Tree[i]=-1; 
 
	ArrangeNodes(); 
 
 }	 
 
void AbstractTreeModel::PrintModel() 
 { 
  	 
	int j; 
 
	 
        for(j=0; j<length; j++)  
          {	   
	    if(Tree[Queue[j]]!=-1) cout<<" ("<<Queue[j]<<"|"<<Tree[Queue[j]]<<") "; 
	    else cout<<" ("<<Queue[j]<<") "; 
             // if(Tree[Queue[j]]!=-1) fprintf(outfile, "(%d|%d) ",Queue[j],Tree[Queue[j]] ); 
             // else  fprintf(outfile, "(%d) ",Queue[j]); 
          } 
        cout<<endl;
                   
 }	 
 
 

void AbstractTreeModel::PrintMut() 
{ 
  int aux,j,k; 



	for(j=0; j<length; j++) 
	{ 
	  	for(k=0 ; k<length; k++) 
		{ 
          if (k==j) 
		  { 
		    //fprintf(f1,"%f ",0.0);  	        
			printf("%f ",0.0);  	        
			   
		  }	   
		   else if(j<k) 
		   {  
  			aux = j*(2*length-j+1)/2 +k-2*j-1; 
			printf("%f ",MutualInf[aux]); 
			//fprintf(f1,"%f ",MutualInf[aux]);  
		   } 
		   else 
			{  
			aux = k*(2*length-k+1)/2 +j-2*k-1; 
			printf("%f ",MutualInf[aux]); 
			//fprintf(f1,"%f ",MutualInf[aux]);  
		   } 
		}	 
		//fprintf(f1,"\n"); 
		printf("\n"); 
	} 
	//fclose(f1); 
}
 
// This function reorder the direction of edges when a new 
// edge has been added between non connected components. 
 
 
/* 
void AbstractTreeModel::MakeTree(int rootn) 
 { 
  // En cada paso se incorpora el nodo que no estando en el arbol 
  // tiene el mayor valor de informacion mutua con alguno de los 
  // nodos que ya estan en el arbol, el cual sera ademas su padre 
 
	double max,threshhold,auxm; 
	int maxsonindex; 
	int maxfatherindex; 
	int i,j,k,aux; 
 
         maxsonindex=0; 
         maxfatherindex=0; 
 
	 for(i=0; i<length; i++) Tree[i]=i; 
	 Tree[rootn]=-1; 
	 threshhold=-100;//0.005; 
 
 
	for(i=0; i<length-1; i++)  // Para los n-1 nodos que faltan por incorporar 
	 { 
		max=-10; 
		for(j=0; j<length; j++) 
		 for(k=0; k<length; k++) 
		  { 
			 if (Tree[j]==j && Tree[k]!=k ) 
				  { 
 
					  aux = j*(2*length-j+1)/2 +k-2*j-1; 
					   
					  if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1; 
					  else aux = k*(2*length-k+1)/2 +j-2*k-1; 
					   
					  auxm=MutualInf[aux]; 
					   
					  if (auxm>max) 
						 { 
								maxsonindex=j; 
								maxfatherindex=k; 
								max=auxm; 
						 } 
				  } 
 
		  } 
		 if (max>=threshhold) Tree[maxsonindex]=maxfatherindex; 
		 else Tree[maxsonindex]=-1; 
	 } // Llegado este punto se supone que todos los nodos esten en el arbol 
  ArrangeNodes(); 
 }	 
*/ 
 
 


void AbstractTreeModel::ArrangeNodes() 
{  
	int j,p,current; 
	/* 
    Queue[0] = rootnode; 
	current = 0; 
    p = 0; 
	while (p < length) 
	{ 
		for(j=0; j<length; j++) 
			if (Tree[j]==Queue[p] || (Tree[j]==-1 && j !=rootnode) ) Queue[++current]=j; 
		   p++;  
    } 
 
	*/ 
  current = 0; 
  for(j=0; j<length; j++) if (Tree[j]==-1) Queue[current++]=j; 
  p = 0; 
	while (p < length) 
	{ 
		for(j=0; j<length; j++) 
			if (Tree[j]==Queue[p] ) Queue[current++]=j; 
		   p++;  
        }	     
        
	/* 
  for(i=0; i<length; i++)   
    { 
      cout<<Queue[i]<<"  "<<Tree[Queue[i]]<<endl; 
    } 
	*/ 
} 


void AbstractProbModel::CalculateILikehood(Popul* EPop, double* PopProb) 
{ 	 
 int j; 
 double auxprob; 
   
    Likehood = 0;  
    TreeProb = 0; 
 
         for(j=0; j < genepoollimit; j++) 
	 { 
            auxprob  = Prob(EPop->Ind(j)); 
           
            TreeProb += auxprob; 
	    if (auxprob>0)  Likehood  += (NPoints*PopProb[j]*(log(auxprob)));  
            //else  Likehood  -= (genepoollimit*PopProb[j]*(log(PopProb[j])));  
            
	    //cout<<auxprob<<" "<<PopProb[j]<<endl; 
         } 
         //cout<<endl; 
           
} 
 
 
void AbstractTreeModel::MutateTree() 
{  
// The parent of the last node in the tree is changed 
 
	Tree[Queue[length-1]] = randomint(length); 
    while (Tree[Queue[length-1]]== Queue[length-1]) Tree[Queue[length-1]] = randomint(length); 
} 
 
 
void AbstractTreeModel::CleanTree() 
 { 
  int i; 
  DoReArrangeTrees = 0; 
  for(i=0; i<length; i++)  
   { 
    Tree[i] = -1;  
    Queue[i] = i; 
   } 
 }  
 

AbstractTreeModel::~AbstractTreeModel() 
 { 
  delete[] Tree; 
  delete[] Queue; 
  delete[] MutualInf; 
} 
 
 
int AbstractTreeModel::NextInOrder(int previous) 
{ 
	return Queue[previous]; 
} 
 
  
BinaryTreeModel::BinaryTreeModel(int vars, int* AllInd,int clusterkey,Popul* pop, double complexity):AbstractTreeModel(vars,AllInd,clusterkey,pop,complexity) 
 { 
  int i; 
  AllProb = new double[length];  
  AllSecProb = new double*[4];   
  for(i=0; i < 4; i++) AllSecProb[i] = new double[length*(length-1)/2]; 
  BestConf = new unsigned[length]; 
  RootCharge = new int[length]; 
 } 
 
 
BinaryTreeModel::BinaryTreeModel(int vars, double complexity, int apsize):AbstractTreeModel(vars,complexity,apsize) 
 { 
  int i; 
  AllProb = new double[length];   
  AllSecProb = new double*[4];   
  for(i=0; i < 4; i++) AllSecProb[i] = new double[length*(length-1)/2]; 
  BestConf = new unsigned[length]; 
  RootCharge = new int[length]; 
 } 
 
BinaryTreeModel::BinaryTreeModel(BinaryTreeModel* other, int _Marca):AbstractTreeModel(other->length,other->Complexity) 
 { 
  int i,j,aux; 
  AllProb = new double[length];  
  AllSecProb = new double*[4];   
  for(i=0; i < 4; i++) AllSecProb[i] = new double[length*(length-1)/2]; 
  Marca = _Marca;
  BestConf = new unsigned[length];  
  RootCharge = new int[length]; 
  for(i=0; i < length; i++) 
   {
    Tree[i] = other->Tree[i];
    Queue[i] = other->Queue[i];
    AllProb[i] = other->AllProb[i];
    RootCharge[i] = other->RootCharge[i];
    for(j=i+1; j < length; j++) 
    {
      aux = i*(2*length-i+1)/2 +j-2*i-1;
      AllSecProb[0][aux] = other->AllSecProb[0][aux]; 
      AllSecProb[1][aux] = other->AllSecProb[1][aux]; 
      AllSecProb[2][aux] = other->AllSecProb[2][aux]; 
      AllSecProb[3][aux] = other->AllSecProb[3][aux];
    }
   }

 } 

int BinaryTreeModel::FindRootNode() 
 { 
	 // El nodo raiz del arbol se puede escoger aleatoriamente, 
	 // Siguiendo a De Bonet aqui se escoge el de menor entropia incondicional 
	// Se determina la variable con menor entropia incondicional 
    int j,jj; 
	double min; 
    int minindex=0; 
	double aux=0; 
      int* auxsample;

        auxsample = new int[length];
        for(j=0; j<length; j++)  auxsample[j] = j;
         
         
        RandomPerm(length,int(length),auxsample); 
        
        minindex= auxsample[randomint(length)]; 
        
        if( (AllProb[minindex]<1) && (AllProb[minindex]>0) )  min=-((AllProb[minindex]*log(AllProb[minindex])+ (1-AllProb[minindex])*log(1-AllProb[minindex]))); 
        else min=10;
	
	//cout<<minindex<<" "<<AllProb[minindex]<<" "<<min<<endl;         
	for(jj=0; jj<length; jj++) 
	  {           
	   j = auxsample[jj];
           if( (AllProb[j]<1) && (AllProb[j]>0) )
	     {
         	aux= -((AllProb[j]*log(AllProb[j])+ (1-AllProb[j])*log(1-AllProb[j]))); 
			//cout<<AllProb[j]<<" "<<min<<" "<<aux<<endl;
		  if (aux<min) 
			 { 
				minindex=j; 
				min=aux; 
			 } 
                 
	    }
	 } 
 
  delete[] auxsample;
  return minindex; 
 
 } 
 
 

void BinaryTreeModel::PrintAllProbs() 
{ 
  int aux,i,j,k; 


    for(i=0; i<4; i++) 
      {
	for(j=0; j<length; j++) 
	{ 
	  	for(k=0 ; k<length; k++) 
		{ 
          if (k==j) 
		  { 
		    //fprintf(f1,"%f ",0.0);  	        
			printf("%f ",0.0);  	        
			   
		  }	   
		   else if(j<k) 
		   {  
  			aux = j*(2*length-j+1)/2 +k-2*j-1; 
			printf("%f ",AllSecProb[i][aux]); 
			//fprintf(f1,"%f ",MutualInf[aux]);  
		   } 
		   else 
			{  
			aux = k*(2*length-k+1)/2 +j-2*k-1;
                        printf("%f ",AllSecProb[i][aux]);  
			
		   } 
		}	 
		//fprintf(f1,"\n"); 
		printf("\n"); 
	} 
	//fclose(f1); 
        printf("\n\n"); 
      }
} 
 

void BinaryTreeModel::ImportProb( double** SP, double* UP ) 
{ 
 int i,j; 
  for (j=0;j<4;j++) 
    for(i=0; i<length*(length-1)/2; i++) AllSecProb[j][i] = SP[j][i]; 
  for(i=0; i<length; i++) AllProb[i] = UP[i]; 
} 
 
void BinaryTreeModel::ImportMutInf( double* MI ) 
{ 
 int i; 
    for(i=0; i<length*(length-1)/2; i++) MutualInf[i] = MI[i]; 
} 

void BinaryTreeModel::ImportMutInfFromTree( BinaryTreeModel* other ) 
{
    ImportMutInf(other->MutualInf);
}  


void BinaryTreeModel::ImportProbFromTree(BinaryTreeModel* other ) 
{ 
   ImportProb(other->AllSecProb, other->AllProb); 

} 
 

void BinaryTreeModel::PutInMutInfFromTree( BinaryTreeModel* other ) 
{
    int i,aux; 
    for(i=0; i<length; i++) 
       	 if(other->Tree[i]>-1)
         {
           if (other->Tree[i]<i) 
                 aux = other->Tree[i]*(2*length-other->Tree[i]+1)/2 +i-2*other->Tree[i]-1; 
	    else aux = i*(2*length-i+1)/2 +other->Tree[i]-2*i-1; 
            MutualInf[aux] = -100;
         }	
} 


 void BinaryTreeModel::FindLikehood() 
{ 	 
 int j; 
    
    Likehood = 0;  
    TreeProb = 0; 

 for(j=0; j<length; j++)
 {
        if (Tree[j]==-1) 
          Likehood += UnivGainLikehood(j); 
        else  
          Likehood += BivGainLikehood(Tree[j],j);      
 }
}

void BinaryTreeModel::SetNPoints(int NP)
{
    NPoints = NP;
}

 
 


void BinaryTreeModel::CalProb() 
 { 
   int aux,i,j,k,l; 
	// Primer paso 
	// Se calculan todas las probabilidades de primer y segundo orden 
	// para todas las variables del cromosoma, teniendo en cuenta solo 
	// aquellos individuos incluidos en el conjunto seleccionado 
 
     
	 for(j=0; j<length-1; j++) 
	  { 
		AllProb[j]=2*Prior; 
 
		for(k=j+1 ; k<length; k++) 
			{ 
              aux = j*(2*length-j+1)/2 +k-2*j-1; 
			  AllSecProb[0][aux]=Prior; 
			  AllSecProb[1][aux]=Prior; 
			  AllSecProb[2][aux]=Prior; 
			  AllSecProb[3][aux]=Prior; 
if(actualpoolsize>0) 
{ 
			  for(i=0; i<actualpoolsize; i++) 
				  { 
					  //Se calcula la probabilidad de cada gen en genepool		 
				  
					++AllSecProb[2*Pop->P[i][j]+Pop->P[i][k]][aux]; 
					if (k==j+1) AllProb[j]+=Pop->P[i][j]; 
 
				  } 
 
              AllSecProb[0][aux]= AllSecProb[0][aux] / (actualpoolsize+4*Prior); 
  	      AllSecProb[1][aux]= AllSecProb[1][aux] /(actualpoolsize+4*Prior); 
              AllSecProb[2][aux]= AllSecProb[2][aux] / (actualpoolsize+4*Prior); 
	       AllSecProb[3][aux]= AllSecProb[3][aux] /(actualpoolsize+4*Prior); 
} 
 
			 } 
 
	 } 
 
  AllProb[length-1]=2*Prior; 
   
if (actualpoolsize >0) 
{ 
  for(i=0; i<actualpoolsize; i++) AllProb[length-1]+=Pop->P[i][length-1];  
  for(i=0; i<length; i++) AllProb[i] = AllProb[i] / (actualpoolsize+4*Prior);  
 } 
} 

void BinaryTreeModel::InitTree(int InitTreeStructure, int CNumberPoints, double* pvect, Popul* pop, int NumberPoints)
{
  
  cout<<"InitTree "<< InitTreeStructure<<endl;
    //Prior = 1;
 SetGenPoolLimit(CNumberPoints);
 rootnode =  FindRootNode(); //RandomRootNode();  
 SetNPoints(NumberPoints);
	  
  if(InitTreeStructure==0 )
         {
          MakeRandomTree(rootnode);
          RandParam(); 
         }
        else if(InitTreeStructure==1)
         { 
          MakeRandomTree(rootnode);
          CalProbFvect(pop,pvect,CNumberPoints,1); 
         }
       else if(InitTreeStructure==2)
         {
          CalProbFvect(pop,pvect,CNumberPoints,1);   
          CalMutInf(); 
	  MakeTree(rootnode); 
          MutateTree();          
	} 
     	else   
	{  
          CalProbFvect(pop,pvect,CNumberPoints,1);   
          CalMutInf(); 
	  MakeTree(rootnode); 
          //MutateTree(); 
	}  

}

    
    
double BinaryTreeModel::Calculate_Best_Prior(int N, int NumberVars, int dim, double penalty) 
{ 
  double prior; 
 
   prior  = (penalty*N)/(NumberVars*pow(2,dim-1)); 
   return prior;  
 
} 
 
void BinaryTreeModel::CalProbUnif() 
{ int j;
 for(j=0; j<length; j++) AllProb[j] = 0.5; //myrand();
 }
 


void BinaryTreeModel::NormalizeProbabilities(int cprior) 
 { 
   int aux,j,k; 
       	 for(j=0; j<length; j++) 
	  { 
  	   for(k=j+1; k<length; k++) 
		{ 
                   aux = j*(2*length-j+1)/2 +k-2*j-1; 
		   AllSecProb[0][aux]/=(actualpoolsize+4*cprior); 
                   AllSecProb[1][aux]/=(actualpoolsize+4*cprior);
                   AllSecProb[2][aux]/=(actualpoolsize+4*cprior); 
                   AllSecProb[3][aux]/=(actualpoolsize+4*cprior);
                } 
             AllProb[j]/=(actualpoolsize+4*cprior);   		
	  }       
} 


void BinaryTreeModel::CalProbFvect(Popul* EPop, double *vector,int howmany,int cprior) 
 { 
   int aux,i,j,k; 
   genepoollimit = howmany; 
      	 for(j=0; j<length-1; j++) 
	  { 
		AllProb[j]= 2* cprior; 
         
		for(k=j+1 ; k<length; k++) 
			{ 
                          aux = j*(2*length-j+1)/2 +k-2*j-1; 
			  AllSecProb[0][aux]=  cprior; 
			  AllSecProb[1][aux]=  cprior; 
			  AllSecProb[2][aux]=  cprior; 
			  AllSecProb[3][aux]=  cprior; 
 
			  for(i=0; i<genepoollimit; i++) 
				  { 
			  	   AllSecProb[2*EPop->P[i][j]+EPop->P[i][k]][aux]+=(vector[i]*actualpoolsize); 

                                   
			 	   if ((k==j+1) && (EPop->P[i][j]==1)) AllProb[j]+=(vector[i]*actualpoolsize);   
				   // cout<<" i "<<i<<" j "<<j<<" k "<<k<<" "<<AllSecProb[0][aux]<<" "<<AllSecProb[1][aux]<<" "<<AllSecProb[2][aux]<<" "<<AllSecProb[3][aux]<<" "<<aux<<endl; 
			          } 
		
		 } 
		
	 } 

 AllProb[length-1]=2*cprior; 
  for(i=0; i<genepoollimit; i++) 
  { 
	  if (EPop->P[i][length-1]==1) AllProb[length-1]+=vector[i]*actualpoolsize; 
  }  
  
   NormalizeProbabilities(cprior);

 
} 
 
void BinaryTreeModel::CalProbFvect(Popul* EPop, double *vector,int cprior) 
 { 
   int aux,i,j,k; 
   genepoollimit = Pop->psize; 
 
    	 for(j=0; j<length-1; j++) 
	  { 
		AllProb[j]=2*cprior; 
         
		for(k=j+1 ; k<length; k++) 
			{ 
              aux = j*(2*length-j+1)/2 +k-2*j-1; 
	                  AllSecProb[0][aux]=cprior; 
			  AllSecProb[1][aux]=cprior; 
			  AllSecProb[2][aux]=cprior; 
			  AllSecProb[3][aux]=cprior; 
 
			  for(i=0; i<genepoollimit; i++) 
				  { 
					  //Se calcula la probabilidad de cada gen en genepool		 
						AllSecProb[2*EPop->P[i][j]+EPop->P[i][k]][aux]+=(vector[i]*actualpoolsize); 
						if ((k==j+1) && (EPop->P[i][j]==1)) AllProb[j]+=(vector[i]*actualpoolsize);  
			  } 
		
/*	   
                for(i=0; i<4; i++) 
				  { 
					if( AllSecProb[0][aux]< 1/(genepoollimit)) AllSecProb[0][aux] = 0; 
                    if( AllSecProb[1][aux]< 1/(genepoollimit)) AllSecProb[1][aux] = 0; 
					if( AllSecProb[2][aux]< 1/(genepoollimit)) AllSecProb[2][aux] = 0; 
					if( AllSecProb[3][aux]< 1/(genepoollimit)) AllSecProb[3][aux] = 0; 
					if ((k==j+1) && (AllProb[j]< 1/genepoollimit) ) AllProb[j] = 0; 
				  } 
*/		     
			 } 
 
	 }  
   
 AllProb[length-1]=2*cprior; 
  for(i=0; i<genepoollimit; i++) 
  { 
	  if (EPop->P[i][length-1]==1) AllProb[length-1]+=vector[i]*actualpoolsize; 
  }  
  
  NormalizeProbabilities(cprior);
 }
 
 
 
 
 
void BinaryTreeModel::SetPrior(double prior_r_univ,double prior_r,int N) 
 { 
   int j,aux; 
   //Esta forma de calcular los priors esta comprobada solo para priors uniformes (i.e. prior_r_univ = prior_r)
   Prior = prior_r_univ;
   /*
     	 for(j=0; j<length; j++) 
	  { 	    
  	    //AllProb[j]= (AllProb[j]*N+ 2*prior_r_univ)/ (N+ 4*prior_r_univ); 
            cout<<AllProb[j]<<" ";
              if (Tree[j]>-1) 
                { 
		 if (j<Tree[j]) aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
                 else  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		 // cout<<j<<" "<<Tree[j]<<" "<<AllSecProb[0][aux]<<" "<<AllSecProb[1][aux]<<" "<<AllSecProb[2][aux]<<" "<<AllSecProb[3][aux]<<" "<<endl; 	 
                 AllSecProb[0][aux] = (AllSecProb[0][aux]*N +prior_r) / (N+ 4*prior_r); 
                 AllSecProb[1][aux] = (AllSecProb[1][aux]*N +prior_r) / (N+ 4*prior_r); 
                 AllSecProb[2][aux] = (AllSecProb[2][aux]*N +prior_r) / (N+ 4*prior_r); 
                 AllSecProb[3][aux] = (AllSecProb[3][aux]*N +prior_r) / (N+ 4*prior_r); 
                 //cout<<j<<" "<<Tree[j]<<" "<<AllSecProb[0][aux]<<" "<<AllSecProb[1][aux]<<" "<<AllSecProb[2][aux]<<" "<<AllSecProb[3][aux]<<" "<<endl; 
         
} 
	 
 }
	 cout<<endl;
   */
}
 


void BinaryTreeModel::AdjustBivProb(int Best_Edge_i,int Best_Edge_j,double* besttm)
{
    int aux;
  if (Best_Edge_j<Best_Edge_i) 
   {
    aux =Best_Edge_j*(2*length-Best_Edge_j+1)/2 +Best_Edge_i-2*Best_Edge_j-1; 
    AllSecProb[0][aux] = besttm[0];
    AllSecProb[1][aux] = besttm[2]; 
    AllSecProb[2][aux] = besttm[1];
    AllSecProb[3][aux] = besttm[3];   
   }
  else
   {
    aux = Best_Edge_i*(2*length-Best_Edge_i+1)/2 +Best_Edge_j -2*Best_Edge_i-1;
    AllSecProb[0][aux] = besttm[0];
    AllSecProb[1][aux] = besttm[1]; 
    AllSecProb[2][aux] = besttm[2];
    AllSecProb[3][aux] = besttm[3];
   }
  AllProb[Best_Edge_i] = besttm[2]+besttm[3];  
  AllProb[Best_Edge_j] = besttm[1]+besttm[3];
}

void BinaryTreeModel::NormalizeBiv(int i, int j)
{
    double auxsum;
    int aux;
   

 
  if (j<i)   
   {
    aux = j*(2*length-j+1)/2 +i-2*j-1;
    auxsum =  AllSecProb[0][aux] + AllSecProb[1][aux] + AllSecProb[2][aux] + AllSecProb[3][aux];

    AllSecProb[0][aux] /= auxsum;
    AllSecProb[1][aux] /= auxsum;
    AllSecProb[2][aux] /= auxsum;
    AllSecProb[3][aux] /= auxsum;

    AllProb[i] =  AllSecProb[1][aux]+ AllSecProb[3][aux];
    AllProb[j] =  AllSecProb[2][aux]+ AllSecProb[3][aux];
   }
   else 
   {
    aux = i*(2*length-i+1)/2 +j -2*i-1; 
    auxsum =  AllSecProb[0][aux] + AllSecProb[1][aux] + AllSecProb[2][aux] + AllSecProb[3][aux];

    AllSecProb[0][aux] /= auxsum;
    AllSecProb[1][aux] /= auxsum;
    AllSecProb[2][aux] /= auxsum;
    AllSecProb[3][aux] /= auxsum;

    AllProb[i] =  AllSecProb[2][aux]+ AllSecProb[3][aux];
    AllProb[j] =  AllSecProb[1][aux]+ AllSecProb[3][aux];
   }
 
}



double BinaryTreeModel::UnivGainLikehood(int par)
{
    double gain=0;

   if (AllProb[par]>0)  gain = NPoints*AllProb[par]*log(AllProb[par]);
    if (AllProb[par]<1) gain  += NPoints*(1-AllProb[par])*log(1-AllProb[par]); 
 return  gain;
}


double  BinaryTreeModel::BivGainLikehood(int par,int son)
{
    int aux;
    double gain;   
    gain = 0;
     if (son<par)
           {
             aux = son*(2*length-son+1)/2 +par-2*son-1;
             if (AllProb[par]<1 && AllSecProb[0][aux]>0 ) gain += NPoints*AllSecProb[0][aux]*log(AllSecProb[0][aux]/(1-AllProb[par])); 
         if (AllProb[par]>0 && AllSecProb[1][aux]>0) gain += NPoints*AllSecProb[1][aux]*log(AllSecProb[1][aux]/(AllProb[par]));   
         if (AllProb[par]<1 && AllSecProb[2][aux]>0) gain += NPoints*AllSecProb[2][aux]*log(AllSecProb[2][aux]/(1-AllProb[par])); 
         if (AllProb[par]>0 && AllSecProb[3][aux]>0) gain += NPoints*AllSecProb[3][aux]*log(AllSecProb[3][aux]/(AllProb[par]));
           }
           else
           {
            aux = par*(2*length-par+1)/2 +son -2*par-1;
           if (AllProb[par]<1 && AllSecProb[0][aux]>0) gain += NPoints*AllSecProb[0][aux]*log(AllSecProb[0][aux]/(1-AllProb[par])); 
         if (AllProb[par]<1 && AllSecProb[1][aux]>0)  gain += NPoints*AllSecProb[1][aux]*log(AllSecProb[1][aux]/(1-AllProb[par]));   
         if (AllProb[par]>0 && AllSecProb[2][aux]>0) gain += NPoints*AllSecProb[2][aux]*log(AllSecProb[2][aux]/(AllProb[par])); 
         if (AllProb[par]>0 &&  AllSecProb[3][aux]>0) gain += NPoints*AllSecProb[3][aux]*log(AllSecProb[3][aux]/(AllProb[par]));
           }
     return gain;
}

void BinaryTreeModel::ConstructTree()
{
 double bestgain,gain;
 int besta, bestb;
 besta =0;  bestb =0; //finit

 int i,j,aux,resp,number_parents;

 bestgain = 1;
 
 number_parents = 0;

 while (bestgain>0)
 {
  bestgain = -10;
   for(i=0; i < length-1; i++)
   for(j=i+1; j < length; j++)
   {
     if( (resp=Correct_Edge(i,j))>-1) 
          {
           gain = CalculateGainLikehood(i,j,resp)-number_parents*Complexity;
           //cout<<"i "<<i<<" j "<<j<<" gain  "<<gain<<endl;
           if (gain>bestgain)
	     {
  	      besta = i;
              bestb = j;
              bestgain = gain;
             }            
	  } //else cout<<i<<" "<<j<<" Can not be added"<<endl;
      
   }   
     
  if (bestgain>0)
    {
     aux = Add_Edge(besta,bestb); 
     number_parents++;
     //cout<<"Edge-- ("<< besta<<","<<bestb<<")  Contrib  "<<bestgain<<endl;    
    }
 }
 ArrangeNodes(); 
}




double BinaryTreeModel::CalculateGainLikehood(int a,int b,int type_edge)
{
    int par,son;
    double gain;

    par = 0; son = 0; //finit

 if(type_edge== 0 || type_edge== 2)
  {  son = a;
     par = b;
  }
  else if(type_edge== 1 || type_edge== 3) 
  { 
       son = b;
       par = a;               
  } 

    gain=(BivGainLikehood(par,son)- UnivGainLikehood(son)) ;   
  return gain;
 
}

void BinaryTreeModel::Propagation(int proptype)
{
  
   CollectEvidence(proptype);
   if(proptype) FinalUniv(); 
   DistributeEvidence(proptype);
   if(proptype) FinalUniv(); 
   //PrintProbMod(); 
}

void BinaryTreeModel::NewPropagation(int proptype)
{
    double *solaps0,*solaps1; 

    solaps0 = new double[length]; 
    solaps1 = new double[length];

    InitCliques(solaps0,solaps1 );
/* for(k=0; k<length; k++)  
 { 
     solaps0[k]=1;solaps1[k]=1;
     }*/
   
   NewCollectEvidence(proptype,solaps0,solaps1);
  
   //NewDistributeEvidence(proptype,solaps0,solaps1);
  
   DistributeEvidence(proptype);

   delete[] solaps0;
   delete[] solaps1;
   //PrintProbMod(); 
}


/*
void BinaryTreeModel::CreateTreeFromPropagation(int proptype)
{
   RootCharge = new int[length]; 
   FindRootCharges(); 
   CollectEvidence(proptype);
   DistributeEvidence(proptype);
   FinalUniv(); 
   PrintProbMod(); 

   delete[] RootCharge;

}
*/
void BinaryTreeModel::FinalUniv()
{
    int i,j,k,aux;
  for(k=0; k<length; k++)  
          {	   
	    i = Queue[k];
            j = Tree[i];
	    if(j ==-1 && RootCharge[i]!=-1) j=RootCharge[i];
	    if (j !=-1)
	    {              

             if (j<i)   
              {
                aux = j*(2*length-j+1)/2 +i-2*j-1;
                AllProb[i] =  AllSecProb[1][aux]+ AllSecProb[3][aux];
                AllProb[j] =  AllSecProb[2][aux]+ AllSecProb[3][aux];
              }
             else 
              {
               aux = i*(2*length-i+1)/2 +j -2*i-1; 
               AllProb[i] =  AllSecProb[2][aux]+ AllSecProb[3][aux];
               AllProb[j] =  AllSecProb[1][aux]+ AllSecProb[3][aux];
              }
            }
	  }
}
 
void BinaryTreeModel::NewDistributeEvidence(int proptype,double* solaps0,double* solaps1)
{
   int father,j,k,node;
   double summarg[2];
  
   double lamdaprop[2];

     for(j=0; j<length; j++)  
          {	   
	    node = Queue[j];
            father = Tree[node];
	    if(father!=-1) 
              {            
		 // NormalizeBiv(father,node);
               if (proptype) FindSumMarg(node,father,node, &*summarg); //Here the solap is node
               else          FindMaxMarg(node,father,node, &*summarg); //Here the solap is node
            
            for(k=j+1; k<length; k++)  
		if(Tree[Queue[k]]== node) 
               {
		   
                 Findlamdaprop(&*summarg,solaps0[node],solaps1[node],&*lamdaprop);
                 //solaps0[node] = summarg[0]; solaps1[node] = summarg[1];
                 IncorporateEvidence(Queue[k],node, Queue[k],&*lamdaprop);               
	       } 
     }
  }
}


void BinaryTreeModel::InitCliques(double* solaps0,double* solaps1)
{
   int father,j,node;
   double summarg[2];
  
     for(j=0; j<length; j++)  
     {	
            summarg[0]=-1; summarg[1]=-1;      
	    node = Queue[j];
            father = Tree[node];
	    if(father!=-1)  FindSumMarg(node,father,node, &*summarg); 
            else if(RootCharge[node]!=-1)  FindSumMarg(RootCharge[node],node,node, &*summarg); 
            solaps0[node] = 1-AllProb[node];
            solaps1[node] = AllProb[node];
	    /* if(summarg[0]+summarg[1]>0) 
	    { 
             solaps0[node] = summarg[0]/(summarg[0]+summarg[1]);
             solaps1[node] = summarg[1]/(summarg[0]+summarg[1]);
            }
            else
            {
             solaps0[node] = summarg[0];
             solaps1[node] = summarg[1];
            } 
	    */
     }
}


void BinaryTreeModel::DistributeEvidence(int proptype)
{
   int father,j,k,node;
   double summarg[2];
   double summargtarg[2];
   double lamdaprop[2];

     for(j=0; j<length; j++)  
          {	   
	    node = Queue[j];
            father = Tree[node];
	    if(father!=-1) 
              {            
		 // NormalizeBiv(father,node);
               if (proptype) FindSumMarg(node,father,node, &*summarg); //Here the solap is node
               else          FindMaxMarg(node,father,node, &*summarg); //Here the solap is node
              }
            else if(RootCharge[node]!=-1)
	      {
	      if (proptype) FindSumMarg(RootCharge[node],node,node, &*summarg); 
               else  FindMaxMarg(RootCharge[node],node,node, &*summarg); 
              }
   
            for(k=j+1; k<length; k++)  
		if(Tree[Queue[k]]== node) 
               {
                if (proptype) FindSumMarg(Queue[k],node, node,&*summargtarg); 
                else          FindMaxMarg(Queue[k],node, node,&*summargtarg);  
                 Findlamdaprop(&*summarg,&*summargtarg,&*lamdaprop);
                 IncorporateEvidence(Queue[k],node, Queue[k],&*summarg);               
	       } 
     }
  }



void BinaryTreeModel::IncorporateEvidence(int i,int j, int father, double *summarg)
{
    int aux;
  if (j<i)   
   {
    aux = j*(2*length-j+1)/2 +i-2*j-1;
    if (father != i)
     {
       AllSecProb[0][aux] *= summarg[0];
       AllSecProb[1][aux] *= summarg[1]; 
       AllSecProb[2][aux] *= summarg[0];
       AllSecProb[3][aux] *= summarg[1];
     }
    else 
     {       
       AllSecProb[0][aux] *= summarg[0];
       AllSecProb[1][aux] *= summarg[0]; 
       AllSecProb[2][aux] *= summarg[1];
       AllSecProb[3][aux] *= summarg[1];
     }
    
   }
   else 
   {
    aux = i*(2*length-i+1)/2 +j -2*i-1;
    if (father == i)
     {
       AllSecProb[0][aux] *= summarg[0];
       AllSecProb[1][aux] *= summarg[1]; 
       AllSecProb[2][aux] *= summarg[0];
       AllSecProb[3][aux] *= summarg[1];
     }
    else 
     {       
       AllSecProb[0][aux] *= summarg[0];
       AllSecProb[1][aux] *= summarg[0]; 
       AllSecProb[2][aux] *= summarg[1];
       AllSecProb[3][aux] *= summarg[1];
     }
   }
  
}

void BinaryTreeModel::CollectEvidence(int proptype)
{
   int father,j,node;
   double summarg[2];
   double summargtarg[2];
   double lamdaprop[2];

     for(j=length-1; j>=0; j--)  
          {	   
	    node = Queue[j];
            father = Tree[node]; 
	  if(father!=-1)
          {
	    if(Tree[father]!=-1 ) 
             {
		 // NormalizeBiv(father,node);
              if (proptype) 
                  {
                   FindSumMarg(node,father,father, &*summarg);
                   FindSumMarg(Tree[father],father, father, &*summargtarg);
		  }    
              else 
                  {
                    FindMaxMarg(node,father,father, &*summarg);
                    FindSumMarg(Tree[father],father, father,&*summargtarg);
                  }   
	     }
	    else if(Tree[father]==-1 &&  RootCharge[father] != -1 && node != RootCharge[father])
             {
              if (proptype) 
                  {
                   FindSumMarg(node,father,father, &*summarg);
                   FindSumMarg(father,RootCharge[father],RootCharge[father],&*summargtarg);
		  }    
              else 
                  {
                    FindMaxMarg(node,father,father, &*summarg);
                    FindSumMarg(father, RootCharge[father],father,&*summargtarg);
                  }   
	     }
            if(!(Tree[father]==-1 &&  node == RootCharge[father]))
	    {
               Findlamdaprop(&*summarg,&*summargtarg,&*lamdaprop); 
             
	       if(Tree[father]!=-1) IncorporateEvidence(Tree[father],father, Tree[father],&*lamdaprop);	//&*lamdaprop
	       else IncorporateEvidence(father,RootCharge[father],RootCharge[father],&*lamdaprop); //&*summarg
            }
           }
          }                 
         }






void BinaryTreeModel::NewCollectEvidence(int proptype,double* solaps0,double* solaps1)
{
   int father,j,node;
   double summarg[2];
   double lamdaprop[2];

     for(j=length-1; j>=0; j--)  
       {	   
	  node = Queue[j];
          father = Tree[node]; 
	  if(father!=-1)
          {
	    if(Tree[father]!=-1 ) 
             {
	      FindMaxMargR(node,father,father, &*summarg);                    
	      Findlamdaprop(&*summarg,solaps0[father],solaps1[father],&*lamdaprop); 
              IncorporateEvidence(Tree[father],father,Tree[father],&*lamdaprop);
             }
            else if (RootCharge[father] != -1 && node != RootCharge[father])
             {
              FindMaxMargR(node,father,father, &*summarg);                    
	      Findlamdaprop(&*summarg,solaps0[father],solaps1[father],&*lamdaprop);               IncorporateEvidence(RootCharge[father],father,RootCharge[father],&*lamdaprop);
             }

          }
        }                 
}

void BinaryTreeModel::UpdateCliqFromConf(unsigned* Conf)
{
   int father,node,j;
   double lamdaprop[2];

    for (j=0;j<=Marca;j++)

 {
      node = Queue[j];
      father = Tree[node];
      if(j<Marca)
       {
        if(Conf[node]==0)
        {
         lamdaprop[0] = 1; lamdaprop[1] = 0;
        }
        else
        {
         lamdaprop[0] = 0; lamdaprop[1] = 1;
        }
       }
      else   
       {
        if(Conf[node]==0)
        {
         lamdaprop[0] = 0;  lamdaprop[1] = 1;
        }
        else
        {
         lamdaprop[0] = 1;  lamdaprop[1] = 0;
        }
       }

      if(father!=-1) 
          IncorporateEvidence(father,node,father,&*lamdaprop);	
      else if(father==-1 &&  RootCharge[node] != -1)
           IncorporateEvidence(RootCharge[node],node,RootCharge[node],&*lamdaprop);
      else
      { //Se supone que el tree esta completamente conectado
	     if (j<Marca) AllProb[node] = Conf[node];
             else AllProb[node] = 1-Conf[node];
         } 
    }                 
 }


void BinaryTreeModel::FindBestConf()
{
   int father,node,j,aux;
    
   for(j=0; j<length; j++)  
          {	   
	    node = Queue[j];
            father = Tree[node];
	    if(father==-1 && RootCharge[node]==-1)
	    {
		if (AllProb[node]>=0.5) BestConf[node] = 1;
                else  BestConf[node] = 0;              
            }
            else
	    if(father==-1 && RootCharge[node]!=-1)
            {
		father  =RootCharge[node];
            if (node<father)   
              {
                aux = node*(2*length-node+1)/2 +father-2*node-1;  
                if (AllSecProb[0][aux]+AllSecProb[1][aux]>AllSecProb[2][aux]+AllSecProb[3][aux])               
		  BestConf[node] = 0;
               else  
                  BestConf[node] = 1;                  
	      }
              else
              {
               aux = father*(2*length-father+1)/2 +node-2*father-1;
   if (AllSecProb[0][aux]+AllSecProb[2][aux]>=AllSecProb[1][aux]+AllSecProb[3][aux])               
		  BestConf[node] = 0;
               else  
                  BestConf[node] = 1; 
              }
            }
            else
            {
             if (node<father)   
              {
               aux = node*(2*length-node+1)/2 +father-2*node-1; 
	       if(AllSecProb[BestConf[father]][aux]>=AllSecProb[BestConf[father]+2][aux])
		  BestConf[node] = 0;
               else  
                  BestConf[node] = 1;
              }
              else 
              {
               aux = father*(2*length-father+1)/2 +node-2*father-1;
               if(AllSecProb[2*BestConf[father]][aux]>=AllSecProb[2*BestConf[father]+1][aux])
		   BestConf[node] = 0;
               else  
                   BestConf[node] = 1;
  
              }

             }

           }
      
      }





void BinaryTreeModel::FindRootCharges()
{ 
    int i,j;
  for(i=0; i<length; i++)
    if (Tree[i]==-1) 
       {
	   j=0;
           while (j<length && Tree[j]!=i) j++;
           if(j>length-1) RootCharge[i]=-1;
           else RootCharge[i]=j;
       }
 
}

void BinaryTreeModel::Findlamdaprop(double *summarg,double *summargtarg,double *lamdaprop)
{
    if (summargtarg[0]>0) lamdaprop[0] = summarg[0] / summargtarg[0]; else lamdaprop[0] =0;
    if (summargtarg[1]>0) lamdaprop[1] = summarg[1] / summargtarg[1]; else lamdaprop[1] =0;
}

void BinaryTreeModel::Findlamdaprop(double *summarg,double m0,double m1,double *lamdaprop)
{
    if (m0>0) lamdaprop[0] = summarg[0] / m0; else lamdaprop[0] =0;
    if (m1>0) lamdaprop[1] = summarg[1] / m1; else lamdaprop[1] =0;
}

void BinaryTreeModel::FindSumMarg(int i,int j, int father, double *summarg)
{
    int aux;
  if (j<i)  
   {
    aux = j*(2*length-j+1)/2 +i-2*j-1; 
    if (father != i)
     {
	summarg[0] = AllSecProb[0][aux]+AllSecProb[1][aux];
        summarg[1] = AllSecProb[2][aux]+AllSecProb[3][aux];
     }
    else 
     {       
        summarg[0] = AllSecProb[0][aux]+AllSecProb[2][aux];
        summarg[1] = AllSecProb[1][aux]+AllSecProb[3][aux]; 
     }
   }
   else 
   {
    aux = i*(2*length-i+1)/2 +j -2*i-1;
    if (father == i)
     {
	summarg[0] = AllSecProb[0][aux]+AllSecProb[1][aux];
        summarg[1] = AllSecProb[2][aux]+AllSecProb[3][aux];
     }
    else 
     {       
        summarg[0] = AllSecProb[0][aux]+AllSecProb[2][aux];
        summarg[1] = AllSecProb[1][aux]+AllSecProb[3][aux]; 
     }
   }
 }

void BinaryTreeModel::FindMaxMarg(int i,int j, int father, double *summarg)
{
    int aux;

    summarg[0] = 0; summarg[1] = 0;
  if (j<i)  
   {
    aux = j*(2*length-j+1)/2 +i-2*j-1; 
    if (father != i)
     {
	if (max(AllSecProb[0][aux],AllSecProb[1][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[1][aux])/(AllSecProb[0][aux]+AllSecProb[1][aux]);
        if (max(AllSecProb[2][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[2][aux],AllSecProb[3][aux])/(AllSecProb[2][aux]+AllSecProb[3][aux]);
     }
    else 
     {       
       if (max(AllSecProb[0][aux],AllSecProb[2][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[2][aux])/(AllSecProb[0][aux]+AllSecProb[2][aux]);
       if (max(AllSecProb[1][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[1][aux],AllSecProb[3][aux])/(AllSecProb[1][aux]+AllSecProb[3][aux]); 
     }
   }
   else 
   {
    aux = i*(2*length-i+1)/2 +j -2*i-1;
    if (father == i)
     {
	if (max(AllSecProb[0][aux],AllSecProb[1][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[1][aux])/(AllSecProb[0][aux]+AllSecProb[1][aux]);
        if (max(AllSecProb[2][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[2][aux],AllSecProb[3][aux])/(AllSecProb[2][aux]+AllSecProb[3][aux]);
     }
    else 
     {       
        if (max(AllSecProb[0][aux],AllSecProb[2][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[2][aux])/(AllSecProb[0][aux]+AllSecProb[2][aux]);
        if (max(AllSecProb[1][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[1][aux],AllSecProb[3][aux])/(AllSecProb[1][aux]+AllSecProb[3][aux]); 
     }
   }
 }



void BinaryTreeModel::FindMaxMargR(int i,int j, int father, double *summarg)
{
    int aux;

    summarg[0] = 0; summarg[1] = 0;
  if (j<i)  
   {
    aux = j*(2*length-j+1)/2 +i-2*j-1; 
    if (father != i)
     {
	if (max(AllSecProb[0][aux],AllSecProb[1][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[1][aux]);
        if (max(AllSecProb[2][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[2][aux],AllSecProb[3][aux]);
     }
    else 
     {       
       if (max(AllSecProb[0][aux],AllSecProb[2][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[2][aux]);
       if (max(AllSecProb[1][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[1][aux],AllSecProb[3][aux]); 
     }
   }
   else 
   {
    aux = i*(2*length-i+1)/2 +j -2*i-1;
    if (father == i)
     {
	if (max(AllSecProb[0][aux],AllSecProb[1][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[1][aux]);
        if (max(AllSecProb[2][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[2][aux],AllSecProb[3][aux]);
     }
    else 
     {       
        if (max(AllSecProb[0][aux],AllSecProb[2][aux])>0) summarg[0] = max(AllSecProb[0][aux],AllSecProb[2][aux]);
        if (max(AllSecProb[1][aux],AllSecProb[3][aux])>0) summarg[1] = max(AllSecProb[1][aux],AllSecProb[3][aux]); 
     }
   }
 }

void BinaryTreeModel::CalMutInf() 
 { 
   int j,k; 
	 // Segundo paso 
	 // Se construye un arbol que contendra un conjunto optimo 
	 // de las dependencias de primer orden. El criterio de optimalidad 
	 // para incorporar una relacion al conjunto es maximizar la 
	 // la informacion mutua entre las variables. ( VER Kullback-Lieber divergence ) 
 
	 // Se halla la informacion mutua. 
 
   double aux1,aux2; 
   int aux; 
 
 
      aux1 =0; 

	 for(j=0; j<length-1; j++) 
	  	for(k=j+1 ; k<length; k++) 
			{ 
                     
				aux = j*(2*length-j+1)/2 +k-2*j-1; 
				MutualInf[aux]=0; // (AllProb[j] !=1 && AllProb[k] !=1 ) 
 
				aux2=(AllSecProb[0][aux]+aux1); 
				if (aux2 > 0.000000001) //0.000000001 
                                MutualInf[aux]+=aux2*(log(aux2/((1-AllProb[j])*(1-AllProb[k])))); 
				
                                aux2=(AllSecProb[1][aux]+aux1); 
				if (aux2 > 0.000000001 ) //.000000001 
				MutualInf[aux]+=aux2*(log(aux2/((1-AllProb[j])*(AllProb[k])))); 
                                
                                aux2=(AllSecProb[2][aux]+aux1); 
				if (aux2 > 0.000000001 )  
				MutualInf[aux]+=aux2*(log(aux2/((AllProb[j])*(1-AllProb[k])))); 
                               
                                aux2=(AllSecProb[3][aux]+aux1); 
				if (aux2 > 0.000000001)  
				MutualInf[aux]+=aux2*(log(aux2/((AllProb[j])*(AllProb[k])))); 
		
                       } 
	
}	           			 
    

void BinaryTreeModel::UpdateSecProb(BinaryTreeModel* other)
{
   int j,k,aux; 

    for(j=0; j<length; j++) 
     if(Tree[j]!= -1)
	{
	    k= Tree[j];
          if(j<Tree[j]) aux = j*(2*length-j+1)/2 +k-2*j-1; 
          else   aux = k*(2*length-k+1)/2 +j-2*k-1;
            AllSecProb[0][aux]= other->AllSecProb[0][aux];
            AllSecProb[1][aux]= other->AllSecProb[1][aux];
            AllSecProb[2][aux]= other->AllSecProb[2][aux];
            AllSecProb[3][aux]= other->AllSecProb[3][aux];
	}
    
 }

void BinaryTreeModel::UpdateSecProb(BinaryTreeModel* another, BinaryTreeModel* other)
{
   int j,k,aux; 

    for(j=0; j<length; j++) 
     if(Tree[j]!= -1)
	{
	    k= Tree[j];
          if(j<Tree[j]) aux = j*(2*length-j+1)/2 +k-2*j-1; 
          else   aux = k*(2*length-k+1)/2 +j-2*k-1;
	  if(another->AllSecProb[0][aux]==0) AllSecProb[0][aux] = 0;
          else   AllSecProb[0][aux]= other->AllSecProb[0][aux];
          if(another->AllSecProb[1][aux]==0) AllSecProb[1][aux] = 0;
          else    AllSecProb[1][aux]= other->AllSecProb[1][aux];
           if(another->AllSecProb[2][aux]==0) AllSecProb[2][aux] = 0;
          else   AllSecProb[2][aux]= other->AllSecProb[2][aux];
          if(another->AllSecProb[3][aux]==0) AllSecProb[3][aux] = 0;
          else    AllSecProb[3][aux]= other->AllSecProb[3][aux];
	}
    else AllProb[j] = another->AllProb[j];
 }






void BinaryTreeModel::PutBivInTree(double** SecProbPros) 
 { 
   //The mutual information is calculated from a matrix of bivariate prob.
   // that are not consistent
   int j,k,aux; 

    for(j=0; j<length; j++) 
     if(Tree[j]!= -1)
	{
	    k= Tree[j];
          if(j<Tree[j]) aux = j*(2*length-j+1)/2 +k-2*j-1; 
          else   aux = k*(2*length-k+1)/2 +j-2*k-1;
            AllSecProb[0][aux]=SecProbPros[0][aux];
            AllSecProb[1][aux]=SecProbPros[1][aux];
            AllSecProb[2][aux]=SecProbPros[2][aux];
            AllSecProb[3][aux]=SecProbPros[3][aux];	
	}
 }


void BinaryTreeModel::CalMutInf(double** SecProbPros) 
 { 
   //The mutual information is calculated from a matrix of bivariate prob.
   // that are not consistent
   int j,k,aux,zeroentries; 
   double aux2,APj,APk;    
   
	 for(j=0; j<length-1; j++)  
	  	for(k=j+1 ; k<length; k++) 
			{ 
			    aux = j*(2*length-j+1)/2 +k-2*j-1; 
			    MutualInf[aux]=0; 
                            APj = SecProbPros[2][aux] + SecProbPros[3][aux];
                            APk = SecProbPros[1][aux] + SecProbPros[3][aux];
                            zeroentries=0;
				aux2=(SecProbPros[0][aux]); 
				if (aux2 > 0.000000001) 
                MutualInf[aux]+=aux2*(log(aux2/((1-APj)*(1-APk)))); 
				else  zeroentries++;
 
				aux2=(SecProbPros[1][aux]); 
				if (aux2 > 0.000000001 ) 
				MutualInf[aux]+=aux2*(log(aux2/((1-APj)*(APk)))); 
				else  zeroentries++;
				aux2=(SecProbPros[2][aux]); 
				if (aux2 > 0.000000001 )  
				MutualInf[aux]+=aux2*(log(aux2/((APj)*(1-APk)))); 
				else  zeroentries++;
				aux2=(SecProbPros[3][aux]); 
				if (aux2 > 0.000000001)  
				MutualInf[aux]+=aux2*(log(aux2/((APj)*(APk)))); 
				else  zeroentries++;
			   if(zeroentries>1) MutualInf[aux]=-100;
		} 
 
		 
}	           			 
  
void BinaryTreeModel::TreeStructure(double** AllContrib) 
 { 
   // Creates the structure of the tree based on the gain in the likehood
   // as opposite to the case when mutual information is employed 
   			 
double max,threshhold,auxm,newlikeh,oldlikeh; 
int maxsonindex, maxfatherindex,i,j,k,l,number_parents,full; 
 
         maxsonindex=0; 
         maxfatherindex=0; 
 	 for(i=0; i<length; i++) Tree[i]=i; 
	 Tree[rootnode]=-1; 
	 threshhold=-100;//0.005; 
         number_parents=0;
         newlikeh = Likehood; oldlikeh=Likehood-1.0;
         i=0;
         full = 0;
	while ( !full && newlikeh>=oldlikeh) 
	 { 
               	max=0; 
                oldlikeh = newlikeh;
		for(j=0; j<length; j++) 
		 for(k=0; k<length; k++) 
		  { 
			 if (Tree[j]==j && Tree[k]!=k ) 
                 	     { 
				   auxm=AllContrib[k][j]; 				
		  		   if (auxm>max) 
				     { 
					maxsonindex=j; 
					maxfatherindex=k; 
					max=auxm; 
				      } 
			     } 
			
		  } 
                 
               

		 if (max>0)
                    {
			newlikeh = oldlikeh + max - Complexity; //*log(NPoints);
                      if(newlikeh>oldlikeh)
		       {
                        Tree[maxsonindex]=maxfatherindex; 
                        number_parents++;
                       }
                      else full = 1;
                       // A new parent was added
                      //cout<<newlikeh<< "  "<<oldlikeh<<"  "<<max<<"  "<<Complexity<<endl; 
		    }
		 else 
                   {
		    l=0;
                    while(l<length && Tree[l] !=l) l++;
                    if (l<length) Tree[l] = -1;
                    else full = 1;                                        
                   }
                 
	 }  
 
    for(i=0; i<length; i++)    if (Tree[i] == i) Tree[i] = -1;  
  
  ArrangeNodes(); 
 }	 
 

void BinaryTreeModel::MakeTreeLog() 
 { 
  
   int j,k; 
   double** AllContrib;

   AllContrib = new double*[length];
   for(j=0; j<length; j++) AllContrib[j] = new double[length];
 

	 for(j=0; j<length; j++)  
	  	for(k=0 ; k<length; k++) 
			{ 
			 if (j!=k) AllContrib[j][k] = (BivGainLikehood(j,k)- UnivGainLikehood(k)) ;                           				
		        } 
/*
 for(j=0; j<length; j++)  
 {
     for(k=0 ; k<length; k++) cout<< AllContrib[j][k]<<" ";
     cout<<endl;
 }
*/

 TreeStructure(AllContrib);

for(j=0; j<length; j++) delete[] AllContrib[j];
delete[] AllContrib;
 
}	           			 




 // The following is a more efficient way 
 // of generating the new population by using  
 // the order of the tree just once 
/* 
void ProbTree::GenPop(int From, Popul* NewPop) 
 { 
  // Tercer paso 
   //Se procede a generar todos los individuos de la nueva poblacion 
  // Se utiliza para ello el arbol 
 
double auxprob, cutoff,aux2,aux1; 
 int aux,i,j,p,current; 
 int Npopsize; 
 
 Npopsize=NewPop->psize; 
  
 for (p=0;p<length; p++ ) 
	 { 
        j = NextInOrder(p); 
		for(i=From; i<Npopsize; i++) 
		  {  
		    cutoff = myrand();			 
			if (Tree[j] == -1) 
            { 
			 if (cutoff > AllProb[j]) NewPop->P[i][rootnode]=0; 
		     else NewPop->P[i][j]=1; 
			} 
			else 
            { 
		     if (NewPop->P[i][Tree[j]]==1) 
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
				 
				if (cutoff > auxprob) NewPop->P[i][j]=0; 
				else NewPop->P[i][j]=1; 
			  } 
		} //first for 
	}//second for 
   
} 
*/ 
 
void BinaryTreeModel::ResetProb() 
 { 
   int aux,j,k; 
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
			} 
 
	 } 
 
  AllProb[length-1]=0; 
  genepoollimit=0; 
} 







double BinaryTreeModel::SumProb (Popul* Pop, int howmany) 
 {  
   int i; 
   double resp; 
 
   resp = 0; 
 
   for (i=0;i<howmany; i++ ) resp += Prob(Pop->P[i],0);    
   return resp;  
 }  

double BinaryTreeModel::Prob(unsigned* vector,int cprior) 
 { 
  // The probability of the vector given the tree 
 // is calculated 
 
 double auxprob,aux2,aux1,prob; 
 int aux,j; 
 
 aux1= 0; 
 prob = 1; 
 
	for (j=0;j<length; j++ ) 
	 { 
          
	  if (Tree[j]==-1) prob = (vector[j]==1)?prob*((AllProb[j]*actualpoolsize+2*cprior)/(actualpoolsize+4*cprior)):prob*(((1-AllProb[j])*actualpoolsize+2*cprior)/(actualpoolsize+4*cprior));  
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
	      if((cprior==1) ||( vector[Tree[j]]==1 && AllProb[Tree[j]]>0)) auxprob = (aux2*actualpoolsize+cprior)/(AllProb[Tree[j]]*actualpoolsize+2*cprior); 
		 else if( (cprior==1) || (vector[Tree[j]]==0 && AllProb[Tree[j]] < 1)) auxprob = (aux2*actualpoolsize+cprior)/((1-AllProb[Tree[j]])*actualpoolsize+2*cprior);
	
                 if (auxprob == 0) return 0; 
		 else prob*=auxprob; 
	  } 
	 
	} 
return prob;  
} 


void BinaryTreeModel::SetNoParents() 
 { 
 int i; 
  for (i=0;i<length; i++ ) Tree[i] = -1;
 }
 
void BinaryTreeModel::RandParam() 
 { 
  // Generate the parameters of the tree randomly 
  
 
 double aux2,aux1; 
 int aux,j,i; 
 
	for (i=0;i<length; i++ ) 
	 { 
          j = NextInOrder(i); 
	  if (Tree[j]==-1)  
	    { 
              //cout<<"j="<<j<<"  Tree[j]="<<Tree[j]<<endl; 
              aux1 = myrand(); 
              aux2 = myrand(); 
              AllProb[j] = aux1/(aux1+aux2);  
              //cout<<"AllProb="<<AllProb[j]<<endl; 
           }  
	  else  
	  {	 
	    //cout<<"j="<<j<<"  Tree[j]="<<Tree[j]<<endl; 
          
		 if (j<Tree[j])  
		 { 
				 aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
	        	 aux1 = myrand(); 
                         aux2 = myrand(); 
                         AllSecProb[0][aux] =  aux1/(aux1+aux2) * (1-AllProb[Tree[j]]); 
                         AllSecProb[2][aux] =  aux2/(aux1+aux2) * (1-AllProb[Tree[j]]); 
 
                         aux1 = myrand(); 
                         aux2 = myrand(); 
                         AllSecProb[1][aux] = (aux1/(aux1+aux2)) * (AllProb[Tree[j]]); 
                         AllSecProb[3][aux] = (aux2/(aux1+aux2)) * (AllProb[Tree[j]]); 
                         AllProb[j] = AllSecProb[2][aux]+AllSecProb[3][aux]; 
                         //cout<<"P[00] "<<AllSecProb[0][aux]<<"  P[01] "<<AllSecProb[1][aux]<<"  P[10] "<<AllSecProb[2][aux]<<"  P[11] "<<AllSecProb[3][aux]<<endl; 
 
		 } 
		 else  
		 {  
		   //cout<<"j="<<j<<"  Tree[j]="<<Tree[j]<<endl; 
                         aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
                         aux1 = myrand(); 
                         aux2 = myrand(); 
                         AllSecProb[0][aux] =  (aux1/(aux1+aux2)) * (1-AllProb[Tree[j]]); 
                         AllSecProb[1][aux] =  (aux2/(aux1+aux2)) * (1-AllProb[Tree[j]]); 
                         aux1 = myrand(); 
                         aux2 = myrand(); 
                         AllSecProb[2][aux] = (aux1/(aux1+aux2)) * (AllProb[Tree[j]]); 
                         AllSecProb[3][aux] = (aux2/(aux1+aux2)) * (AllProb[Tree[j]]); 
                         AllProb[j] = AllSecProb[1][aux]+AllSecProb[3][aux];	 
                         //cout<<"P[00] "<<AllSecProb[0][aux]<<"  P[01] "<<AllSecProb[2][aux]<<"  P[10] "<<AllSecProb[1][aux]<<"  P[11] "<<AllSecProb[3][aux]<<endl;	   
		 } 
				 
	  } 
	} 
} 
 
void BinaryTreeModel::PrintProbMod() 
 { 
  
 double aux2,aux3; 
 int aux,j; 
 
  
	for (j=0;j<length; j++ ) 
	 { 
	  if (Tree[j]==-1)  cout<<"P(x"<<j<<"=1)="<<AllProb[j]<<endl; 
	  else  
	  {	   
		 if (j<Tree[j])  
		 { 
				 aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
                                 aux2=AllSecProb[2][aux]; 
	        	         aux3=AllSecProb[3][aux]; 
		 } 
		 else  
		 {  
 		 	      aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
	 		      aux2=AllSecProb[1][aux]; 
                              aux3=AllSecProb[3][aux]; 
		 } 
		 
                 if(AllProb[Tree[j]]>0) cout<<"P(x"<<j<<"=1|x"<<Tree[j]<<"=1)="<<aux3/AllProb[Tree[j]]<<"---"; 
                 else   if(AllProb[Tree[j]]==0) cout<<"P(x"<<j<<"=1|x"<<Tree[j]<<"=1)="<<"0 ---"; 
                 if(AllProb[Tree[j]]<1) cout<<"P(x"<<j<<"=1|x"<<Tree[j]<<"=0)="<<aux2/(1-AllProb[Tree[j]])<<endl; 
                 else  if(AllProb[Tree[j]]==1) cout<<"P(x"<<j<<"=1|x"<<Tree[j]<<"=0)="<<"0 ---"; 
	  } 
	 
	} 
 
} 
 
BinaryTreeModel::~BinaryTreeModel() 
 { 
  int i; 
  delete[] AllProb;  
  for(i=0; i < 4; i++) delete[] AllSecProb[i];   
  delete[] AllSecProb; 
  delete[] BestConf;
  delete[] RootCharge;
 } 
 
 
 
void BinaryTreeModel::GenIndividual (Popul* NewPop, int pos, int cprior) 
 { 
  // The vector in position pos is generated 
  
 double auxprob,aux2,cutoff; 
 int aux,j,i; 

 if (DoReArrangeTrees)  ReArrangeTree(); 

  for (i=0;i<length; i++ ) 
   { 
      j = NextInOrder(i); 
	  cutoff = myrand(); 
	  if (Tree[j]==-1)  
	  { 
		if (cutoff >= ((AllProb[j]*actualpoolsize+2*cprior)/(actualpoolsize+4*cprior))) NewPop->P[pos][j]=0; 
		else NewPop->P[pos][j]=1; 
          }  
	  else  
	  {	   
	   if (NewPop->P[pos][Tree[j]]==1) 
		 { 
	   	  if (j<Tree[j])  
		  { 
		   aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
		   aux2=(AllSecProb[3][aux]); 
		  } 
		  else  
		  { 
 		   aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		   aux2=(AllSecProb[3][aux]); 
		  } 
		  auxprob=(aux2*actualpoolsize+cprior)/(AllProb[Tree[j]]*actualpoolsize+2*cprior); 

		  // if(pos==5)  cout<<" "<<AllSecProb[0][aux]<<" "<<AllSecProb[1][aux]<<" "<<AllSecProb[2][aux]<<" "<<AllSecProb[3][aux]<< endl;
		 } 
		else 
		{ 
         if(j<Tree[j])  
		 { 
		  aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
		  aux2=(AllSecProb[2][aux]); 
		 } 
		 else  
		 { 
		  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		  aux2=(AllSecProb[1][aux]); 
		 } 
	         //auxprob=aux2/(1-AllProb[Tree[j]]); 
                auxprob=(aux2*actualpoolsize+cprior)/((1-AllProb[Tree[j]])*actualpoolsize+2*cprior); 
		// if(pos==5)  cout<<" "<<AllSecProb[0][aux]<<" "<<AllSecProb[1][aux]<<" "<<AllSecProb[2][aux]<<" "<<AllSecProb[3][aux]<< endl;

		} 
  
  	    if (cutoff >= auxprob) NewPop->P[pos][j]=0; 
  	    else NewPop->P[pos][j]=1;           
	  } 
	  // cout<<" j "<<j<<" T[j] "<<Tree[j]<<" X(T[j])= "<<NewPop->P[pos][Tree[j]]<<" BivEn "<<aux2<<"  AllProb "<<AllProb[Tree[j]]<<" cprob "<<auxprob<<" cutoff "<<cutoff<<" X(j)= "<<NewPop->P[pos][j]<<endl;
  } 

 } 
 
 
void BinaryTreeModel::GenPartialIndividual (Popul* NewPop, int pos, int nvars, int *vector,int cprior) 
 { 
  // The vector in position pos is generated 
  
 double auxprob,aux2,cutoff; 
 int aux,j,i,k; 
  
 if (DoReArrangeTrees)  ReArrangeTree(); 
 i = 0;
 k = 0;

  while(i<length && k< nvars) 
     { 
      j = NextInOrder(i);
      while(i< length && vector[j]>-1) // At least one var can be gen. 
       {
	i++;
        j = NextInOrder(i);      
       } 

        vector[j] = 1;
        k++;

          cutoff = myrand(); 
	  if (Tree[j]==-1)  
	  { 
		if (cutoff >= ((AllProb[j]*actualpoolsize+2*cprior)/(actualpoolsize+4*cprior))) NewPop->P[pos][j]=0; 
		else NewPop->P[pos][j]=1; 
          }  
	  else  
	  {	   
	   if (NewPop->P[pos][Tree[j]]==1) 
		 { 
	   	  if (j<Tree[j])  
		  { 
		   aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
		   aux2=(AllSecProb[3][aux]); 
		  } 
		  else  
		  { 
 		   aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		   aux2=(AllSecProb[3][aux]); 
		  } 
		  auxprob=(aux2*actualpoolsize+cprior)/(AllProb[Tree[j]]*actualpoolsize+2*cprior); 

		 } 
		else 
		{ 
         if(j<Tree[j])  
		 { 
		  aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
		  aux2=(AllSecProb[2][aux]); 
		 } 
		 else  
		 { 
		  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 
		  aux2=(AllSecProb[1][aux]); 
		 } 
	         auxprob=(aux2*actualpoolsize+cprior)/((1-AllProb[Tree[j]])*actualpoolsize+2*cprior); 
	
		}  
  
  	    if (cutoff >= auxprob) NewPop->P[pos][j]=0; 
  	    else NewPop->P[pos][j]=1;   
          }        
	  i++; 
	 }  
  
} 

void BinaryTreeModel::UpdateModel(double *vector,int howmany,Popul* EPop) 
{ 
  CalProbFvect(EPop,vector,howmany,0); 
  rootnode = FindRootNode(); //RandomRootNode();  
  CalMutInf(); 
  MakeTree(rootnode); 
} 

 
void BinaryTreeModel::UpdateModelForest(double *vector,int howmany,Popul* EPop) 
{ 
  rootnode = RandomRootNode();
  CalProbFvect(EPop,vector,howmany,0); 
  if (Complexity == 0)
   {
    CalMutInf(); 
    MakeTree(rootnode);
   }
  else
  {
  /*CalculateILikehood(EPop,vector);  
  cout<< " Likehood Tree "<<this->Likehood<<endl; 
  PrintModel(); 
  PrintProbMod(); 
  cout<<endl; 
  CleanTree();*/
 
  CalProbFvect(EPop,vector,howmany,0); 
  CalculateILikehood(EPop,vector); 
  MakeTreeLog();
/*
  CalculateILikehood(EPop,vector); 
  cout<< " Likehood forest "<<this->Likehood<<endl; 
  PrintModel(); 
  PrintProbMod();
  cout<<endl; */
  }
 } 
 
 
void BinaryTreeModel::UpdateModel() 
{ 
  CalProb(); 
  rootnode = RandomRootNode(); 
  CalMutInf(); 
  MakeTree(rootnode); 
} 
 
 

//********************************** 


 
IntTreeModel::IntTreeModel(int vars, int* AllInd,int clusterkey,Popul* pop, double complexity, unsigned int* Cards):AbstractTreeModel(vars,AllInd,clusterkey,pop,complexity) 
 { 
  int i,j,auxinduniv,auxindbiv;    
  TotUnivEntries = 0;
  TotBivEntries = 0;
  Prior  = 0; 
  Card = new int[length];
  IndexUnivEntries = new int[length+1];
  IndexBivEntries = new int[(length*(length-1)/2)+1];
  IndexUnivEntries[0] = 0;
  IndexBivEntries[0] = 0;
  auxinduniv = 0;
  auxindbiv = 0;
 
  for(i=0; i < length; i++) 
  {
   Card[i] = Cards[i];
   TotUnivEntries +=  Card[i];
   IndexUnivEntries[++auxinduniv] = TotUnivEntries;
   for(j=i+1; j < length; j++)
   {
    TotBivEntries += (Cards[i]*Cards[j]);   
    IndexBivEntries[++auxindbiv] = TotBivEntries;     
   }
  }
  AllProb = new double[TotUnivEntries];  
  AllSecProb = new double[TotBivEntries];  
  //MutualInf = new double [length*(length-1)/2]; 
 } 
 
 
IntTreeModel::IntTreeModel(int vars, double complexity,int apsize, unsigned int* Cards):AbstractTreeModel(vars,complexity,apsize) 
 { 
 int i,j,auxinduniv,auxindbiv;    
  TotUnivEntries = 0;
  TotBivEntries = 0;
  Prior = 0; 
  Card = new int[length];
  IndexUnivEntries = new int[length+1];
  IndexBivEntries = new int[(length*(length-1)/2)+1];
  IndexUnivEntries[0] = 0;
  IndexBivEntries[0] = 0;
  auxinduniv = 0;
  auxindbiv = 0;
  for(i=0; i < length; i++) 
  {
   Card[i] = Cards[i];
   TotUnivEntries +=  Card[i];
   IndexUnivEntries[++auxinduniv] = TotUnivEntries;
   for(j=i+1; j < length; j++)
   {
    TotBivEntries += (Cards[i]*Cards[j]);   
    IndexBivEntries[++auxindbiv] = TotBivEntries;     
   }
  }
  //cout<<length<<" "<<TotUnivEntries<<"  "<<TotBivEntries<<"  "<<endl;
  AllProb = new double[TotUnivEntries]; 
  AllSecProb = new double[TotBivEntries];
   //MutualInf = new double [length*(length-1)/2]; 
 } 
 
void IntTreeModel::SetNPoints(int NP)
{
    NPoints = NP;
}

 
void IntTreeModel::CalProb() 
 { 
     int i,j,k,l,ind_u,ind_b,aux;
	
     memset(AllProb, 0, sizeof(double)*TotUnivEntries);
     memset(AllSecProb, 0, sizeof(double)*TotBivEntries);

        for(j=0; j<length-1; j++) 
	  { 
             ind_u = IndexUnivEntries[j];
             for(k=j+1 ; k<length; k++) 
		{ 
                 aux = j*(2*length-j+1)/2 +k-2*j-1;   
		 ind_b =  IndexBivEntries[aux];
                
		   for(l=0; l<actualpoolsize; l++) 
		    { 
		       i = actualindex[l]; 
		       AllSecProb[ind_b+Card[j]*Pop->P[i][j]+Card[k]*Pop->P[i][k]]++;   	       if (k==j+1) AllProb[ind_u+Pop->P[i][j]]++; 
    		     } 
   		    
		} 
 
       if (actualpoolsize >0) 
        { 
         ind_u = IndexUnivEntries[0];
         for(i=0; i<actualpoolsize; i++)
	   AllProb[ind_u+Pop->P[i][length-1]]++ ;             
        }  
       } 

	for(j=0; j<length*(length-1)/2; j++)
         { 
	   if(j<length)  AllProb[j] /= actualpoolsize;
           AllSecProb[j] /= actualpoolsize;
         }
 } 

void IntTreeModel::InitTree(int InitTreeStructure, int CNumberPoints, double* pvect, Popul* pop, int NumberPoints)
{
 SetGenPoolLimit(CNumberPoints);
 rootnode = RandomRootNode();    
 SetNPoints(NumberPoints); 
  if(InitTreeStructure==0 )
         {
          MakeRandomTree(rootnode);
          RandParam(); 
         }
        else if(InitTreeStructure==1)
         { 
          MakeRandomTree(rootnode);
          CalProbFvect(pop,pvect,CNumberPoints,1); 
         }
       else if(InitTreeStructure==2)
         {
          CalProbFvect(pop,pvect,CNumberPoints,1);   
          CalMutInf(); 
	  MakeTree(rootnode); 
          MutateTree();       
	} 
     	else   
	{  
          CalProbFvect(pop,pvect,CNumberPoints,1);   
          CalMutInf(); 
	  MakeTree(rootnode); 
          //MutateTree(); 
	} 
}

    
 
void IntTreeModel::CalProbFvect(Popul* EPop, double *vector,int howmany,int cprior) 
 { 
    int j,k,l,ind_u,ind_b,aux;
    double tot;
     genepoollimit = howmany;
    
 
     memset(AllProb, 0, sizeof(double)*TotUnivEntries);
     memset(AllSecProb, 0, sizeof(double)*(TotBivEntries));
    
       for(j=0; j<length-1; j++) 
	  { 
             ind_u = IndexUnivEntries[j];
             for(k=j+1 ; k<length; k++) 
		{ 
                 aux = j*(2*length-j+1)/2 +k-2*j-1; 
		 ind_b =  IndexBivEntries[aux];
                  tot = 0;

                  for(l=0; l<genepoollimit; l++) 
		    { 
		       AllSecProb[ind_b+Card[k]*EPop->P[l][j]+EPop->P[l][k]] += vector[l];     
                       tot += vector[l];
                       //cout<<j<<" "<<k<<"  "<<ind_b+Card[k]*EPop->P[l][j]+EPop->P[l][k]<<"  "<<AllSecProb[ind_b+Card[k]*EPop->P[l][j]+EPop->P[l][k]]<<"  "<<tot<<endl;
 		       if (k==j+1) AllProb[ind_u+EPop->P[l][j]] += vector[l]; 
    		     } 
	          }      
       } 
  if ( genepoollimit >0) 
        { 
         ind_u = IndexUnivEntries[length-1];
         for(l=0; l<genepoollimit; l++)
	   AllProb[ind_u+EPop->P[l][length-1]] += vector[l];             
        }  
/*
	for(j=0; j<length*(length-1)/2; j++)
         { 
	     cout<<j<<" --> ";
          for(int jj=0; jj<4; jj++)
	      cout<<AllSecProb[IndexBivEntries[j]+jj]<<" ";
	  cout<<endl;
         } 
*/
                                  
  
} 





 void IntTreeModel::CalMutInf() 
 { 
   int j,k,l,m,aux; 
   double aux1,aux2; 
     memset(MutualInf, 0, sizeof(double)*length*(length-1)/2); 
        for(j=0; j<length-1; j++) 
   	   for(k=j+1 ; k<length; k++) 
	     {               
	       aux = j*(2*length-j+1)/2 +k-2*j-1; 
	        for(l=0; l<Card[j]; l++) 
   	          for(m=0; m<Card[k]; m++) 
	          { 
                    aux1 = (AllProb[IndexUnivEntries[j]+l]*AllProb[IndexUnivEntries[k]+m]);
                    aux2 = AllSecProb[IndexBivEntries[aux]+l*Card[k]+m];
 		    if (aux1>0.000000001 &&  aux2 > 0.000000001) //0.000000001 
                       MutualInf[aux] += aux2*log(aux2/aux1); 
		    //if(k==length-1) cout<<j<<" "<<MutualInf[aux]<<" "<<AllProb[IndexUnivEntries[j]+l]<<" "<<AllProb[IndexUnivEntries[k]+m]<<" "<<aux2<<endl;
	 
                }
	     }	 
        	if ( MutualInf[aux]<0)	 MutualInf[aux] = 0;	 
  
 }   

 

 void IntTreeModel::CalMutInfDeception() 
 { 
   int j,k,l,m,aux; 
   double aux1,aux2; 
   double maxbiv,maxuniv; 
   int bestufreq,bestbfreq;
   int totextract = 0;
   int benign = 0;

     memset(MutualInf, 0, sizeof(double)*length*(length-1)/2); 
        for(j=0; j<length-1; j++) 
   	   for(k=j+1 ; k<length; k++) 
	     {               
	       aux = j*(2*length-j+1)/2 +k-2*j-1; 
               maxuniv = 0; 
               maxbiv = 0;
               bestufreq = -1;
               bestbfreq = -1;
	        for(l=0; l<Card[j]; l++) 
   	          for(m=0; m<Card[k]; m++) 
	          { 
                    aux1 = (AllProb[IndexUnivEntries[j]+l]*AllProb[IndexUnivEntries[k]+m]);
                    aux2 = AllSecProb[IndexBivEntries[aux]+l*Card[k]+m];
 		    if (aux1>0.000000001 &&  aux2 > 0.000000001) 
		      {
                       MutualInf[aux] += aux2*log(aux2/aux1);
                        if(aux2 > maxbiv) 
                            {
                              maxbiv =  aux2;
                              bestbfreq = IndexBivEntries[aux]+l*Card[k]+m;
			    }
                         if(aux1 > maxuniv) 
                            {
                              maxuniv = aux1;
                              bestufreq = IndexBivEntries[aux]+l*Card[k]+m;
                            }  
                      } 	       	  			//cout<<"toextract is "<<totextract<<endl;
		  } 	
	       if  (bestbfreq!=bestufreq) 
                       {
			   //MutualInf[aux] = 0; 
                           totextract++;
                        } 
                     else  
                        {
			   MutualInf[aux] = -10;
                           benign ++;
                         } 
	       //PrintMut();
	     }	           
	     	           		
	// cout<<"malign  "<<totextract<<" benign "<<benign<<endl;	 
}   



 void IntTreeModel::CalMutInf(unsigned int** intermatrix) 
 { 
   int j,k,l,m,aux; 
   double aux1,aux2; 
     memset(MutualInf, 0, sizeof(double)*length*(length-1)/2); 
        for(j=0; j<length-1; j++) 
   	   for(k=j+1 ; k<length; k++) 
	     {               
	       if(intermatrix[j][k]==1)
               {
                 aux = j*(2*length-j+1)/2 +k-2*j-1; 
	         for(l=0; l<Card[j]; l++) 
   	          for(m=0; m<Card[k]; m++) 
	          { 
                    aux1 = (AllProb[IndexUnivEntries[j]+l]*AllProb[IndexUnivEntries[k]+m]);
                    aux2 = AllSecProb[IndexBivEntries[aux]+l*Card[k]+m];
 		    if (aux1>0.000000001 &&  aux2 > 0.000000001) //0.000000001 
                       MutualInf[aux] += aux2*log(aux2/aux1); 
	 
                  }
	       }
	     }	           			 
 }   




void IntTreeModel::CalMutInf(double* SecProbPros, double* UnivProbPros) 
 { 
   //The mutual information is calculated from a matrix of bivariate prob.
   // that are not consistent
   int j,k,l,m,aux; 
   double aux2; 
     memset(MutualInf, 0, sizeof(double)*length*(length-1)/2); 
   
     for(j=0; j<length-1; j++) 
   	   for(k=j+1 ; k<length; k++) 
	     {               
	       aux = j*(2*length-j+1)/2 +k-2*j-1; 
	        for(l=0; l<Card[j]; l++) 
   	          for(m=0; m<Card[k]; m++) 
	          { 
		    aux2 = SecProbPros[IndexBivEntries[aux]+l*Card[k]+m];
 		    if (aux2 > 0.000000001) 
                       MutualInf[aux] += aux2*log(aux2/(UnivProbPros[IndexUnivEntries[j]+l]*UnivProbPros[IndexUnivEntries[k]+m])); 
		   
                  }
	     }	       

}	           		

void IntTreeModel::CalProbFvect(Popul* EPop, double *vector,int cprior) 
{ 
   int j,k,l,ind_u,ind_b,aux;
   
//Necesario incorporar los priors en el memset.

     genepoollimit = EPop->psize;
    
     //cout<<TotUnivEntries<<" "<<TotBivEntries<<" "<<genepoollimit<<endl;


     memset(AllProb, 0, TotUnivEntries*sizeof(double));
     memset(AllSecProb, 0, TotBivEntries*sizeof(double));
   
        for(j=0; j<length-1; j++) 
	  { 
             ind_u = IndexUnivEntries[j];
             for(k=j+1 ; k<length; k++) 
		{ 
                 aux = j*(2*length-j+1)/2 +k-2*j-1; 
		 ind_b =  IndexBivEntries[aux];
               	   for(l=0; l<genepoollimit; l++) 
		    { 
		       AllSecProb[ind_b+Card[k]*EPop->P[l][j]+EPop->P[l][k]] += vector[l];   
		       if (k==j+1) AllProb[ind_u+EPop->P[l][j]] += vector[l]; 
    		     }    		    
		} 
      
	  }
       if (genepoollimit >0) 
        { 
         ind_u = IndexUnivEntries[length-1];
         for(int i=0; i<genepoollimit; i++)
	   AllProb[ind_u+EPop->P[i][length-1]] += vector[i];             
        }  
        
/*
	for(j=0; j<length*(length+1)/2; j++)
         { 
	   if(j<length)  AllProb[j] /= genepoollimit;
           AllSecProb[j] /= genepoollimit;
         } 
*/
} 




	

void IntTreeModel::CalProbFvect(Popul* EPop, double *vector,int cprior, unsigned int** matrix) 
{ 
   int j,k,l,ind_u,ind_b,aux;
   
    genepoollimit = EPop->psize; 
  
    memset(AllProb, 0, TotUnivEntries*sizeof(double));
    memset(AllSecProb, 0, TotBivEntries*sizeof(double));
   
      for(j=0; j<length-1; j++) 
	  { 
             ind_u = IndexUnivEntries[j];
             for(k=j+1 ; k<length; k++) 
		{ 
		  if (matrix[j][k]>0)
	           {
                     aux = j*(2*length-j+1)/2 +k-2*j-1; 
 		     ind_b =  IndexBivEntries[aux]; 
                     for(l=0; l<genepoollimit; l++)  AllSecProb[ind_b+Card[k]*EPop->P[l][j]+EPop->P[l][k]] += vector[l];   
   		   }    	
                 if (k==j+1) for(l=0; l<genepoollimit; l++) AllProb[ind_u+EPop->P[l][j]] += vector[l];     		         		    
		}       
	  }
       if (genepoollimit >0) 
        { 
         ind_u = IndexUnivEntries[length-1];
         for(int i=0; i<genepoollimit; i++)
	   AllProb[ind_u+EPop->P[i][length-1]] += vector[i];             
        }     
} 


        			 
  
void IntTreeModel::TreeStructure(double** AllContrib) 
 { 
   // Creates the structure of the tree based on the gain in the likehood
   // as opposite to the case when mutual information is employed 
   			 
double max,threshhold,auxm,newlikeh,oldlikeh; 
int maxsonindex, maxfatherindex,i,j,k,l,number_parents,full; 
 
         maxsonindex=0; 
         maxfatherindex=0; 
 	 for(i=0; i<length; i++) Tree[i]=i; 
	 Tree[rootnode]=-1; 
	 threshhold=-100;//0.005; 
         number_parents=0;
         newlikeh = Likehood; oldlikeh=Likehood-1.0;
         i=0;
         full = 0;
	while ( !full && newlikeh>=oldlikeh) 
	 { 
               	max=0; 
                oldlikeh = newlikeh;
		for(j=0; j<length; j++) 
		 for(k=0; k<length; k++) 
		  { 
			 if (Tree[j]==j && Tree[k]!=k ) 
                 	     { 
				   auxm=AllContrib[k][j]; 				
		  		   if (auxm>max) 
				     { 
					maxsonindex=j; 
					maxfatherindex=k; 
					max=auxm; 
				      } 
			     } 
			
		  } 
                 
               

		 if (max>0)
                    {
			newlikeh = oldlikeh + max - Complexity; //*log(NPoints);
                      if(newlikeh>oldlikeh)
		       {
                        Tree[maxsonindex]=maxfatherindex; 
                        number_parents++;
                       }
                      else full = 1;
                       // A new parent was added
                      //cout<<newlikeh<< "  "<<oldlikeh<<"  "<<max<<"  "<<Complexity<<endl; 
		    }
		 else 
                   {
		    l=0;
                    while(l<length && Tree[l] !=l) l++;
                    if (l<length) Tree[l] = -1;
                    else full = 1;                                        
                   }
                 
	 }  
 
    for(i=0; i<length; i++)    if (Tree[i] == i) Tree[i] = -1;  
  
  ArrangeNodes(); 
 }	 

 
double IntTreeModel::Prob (unsigned* vector, int cprior) 
 { 
  // The probability of the vector given the tree 
 // is calculated 
 
 double auxbiv, auxuniv,prob; 
 int aux,j; 
 prob = 1.0; 
 
	for (j=0;j<length; j++ ) 
	 {        
	  if (Tree[j]==-1) prob *=(((AllProb[IndexUnivEntries[j]+vector[j]])*actualpoolsize+ cprior)/(actualpoolsize+Card[j]* cprior));  
	  else  
	  {	   
	    if (j<Tree[j])  
	     { 
	     
              aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
	      auxbiv=AllSecProb[IndexBivEntries[aux] + vector[j]*Card[Tree[j]]+vector[Tree[j]]]; 
	             
	      /*      cout<<" j "<<j<<" T[j] "<<Tree[j]<<" BivEn "<<IndexBivEntries[aux]<<" ASP-> ";
		      for(jj=0;jj<(IndexBivEntries[aux+1]-IndexBivEntries[aux]);jj++) cout<<AllSecProb[IndexBivEntries[aux] + jj]<<", "; */
	      	      
	     } 
	     else  
	     {  
              aux = Tree[j]*(2*length-Tree[j]+1)/2 +j-2*Tree[j]-1; 
	      auxbiv= AllSecProb[IndexBivEntries[aux] + vector[Tree[j]]*Card[j]+vector[j]]; 	// cout<<" j "<<j<<" T[j] "<<Tree[j]<<" BivEn "<<IndexBivEntries[aux]<<" ASP "<<AllSecProb[IndexBivEntries[aux] + vector[Tree[j]]*Card[j]+vector[j]]<<" prob "<<prob<<endl;
	      
/*cout<<" j "<<j<<" T[j] "<<Tree[j]<<" BivEn "<<IndexBivEntries[aux]<<" ASP-> ";
  for(jj=0;jj<IndexBivEntries[aux+1]-IndexBivEntries[aux];jj++) cout<<AllSecProb[IndexBivEntries[aux] + jj]<<", ";*/	          
	     } 
             auxuniv = AllProb[IndexUnivEntries[Tree[j]]+vector[Tree[j]]];
	     //cout<<" IUE "<<IndexUnivEntries[Tree[j]]<<" auxuniv "<<auxuniv<<endl;
	     if ( cprior>0 || auxuniv>0) 
              {
               prob *= ((auxbiv*actualpoolsize+ cprior)/(auxuniv*actualpoolsize+Card[Tree[j]]* cprior));
               //cout<<"    auxbiv " <<auxbiv<<" auxuniv "<<auxuniv<<" cond "<<auxbiv/auxuniv<<" prob "<<prob<<endl;
              } 
              else return 0;
	  } 
	 
	} 
 return prob;  
} 



double IntTreeModel::SumProb (Popul* Pop, int howmany) 
 {  
   int i; 
   double resp; 
 
   resp = 0; 
 
   for (i=0;i<howmany; i++ ) resp += Prob(Pop->P[i],0);    
   return resp;  
 }   
 


IntTreeModel::~IntTreeModel() 
 { 

   delete[] IndexUnivEntries;
  delete[] IndexBivEntries;
  delete[] AllProb;  
  delete[] AllSecProb;
   delete[] Card;
 } 

void IntTreeModel::SetPrior(double prior_r_univ,double prior_r,int N) 
 { 
     Prior =  prior_r_univ;
     /*
int i,j,aux,CardBiv; 
   
   	 for(j=0; j<length; j++) 
	  { 
            for(i=0; i<Card[j]; i++) 
             AllProb[IndexUnivEntries[j]+i] = (AllProb[IndexUnivEntries[j]+i]*N+Card[j]*prior_r_univ)/ (N+ Card[j]*prior_r_univ); 
                    for(i=0; i<4*length; i++) cout<<" "<<AllSecProb[i];
                    cout<<endl;
              if (Tree[j]>-1) 
                { 
		 if (j<Tree[j]) aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
                 else  aux = Tree[j]*(2*length-Tree[j]+1)/2 +j -2*Tree[j]-1; 		 
                 CardBiv = IndexBivEntries[aux+1]-IndexBivEntries[aux];
                 for(int ii=0; ii<CardBiv; ii++) 
                 {
		     cout<<aux<<" "<<IndexBivEntries[aux]<<"  "<< j<< ","<<Tree[j]<<" "<<AllSecProb[IndexBivEntries[aux]+ii];
		     
                     AllSecProb[IndexBivEntries[aux]+ii] = (AllSecProb[IndexBivEntries[aux]+ii]*N +prior_r) / (N+ CardBiv*prior_r);   
		    
                 }
       cout<<endl;
                } 
	 } 
 for(j=0; j<2*length; j++) cout<<" "<<AllProb[j];
   cout<<endl;

   for(j=0; j<length; j++) 
	{
	    cout<<"var "<<j<<"  : ";
          for(int i=0; i<Card[j]; i++) 
           cout<<AllProb[IndexUnivEntries[j]+i]<<" ";
          cout<<endl;
        }
*/  

} 
 
void IntTreeModel::GenIndividual (Popul* NewPop, int pos) 
 { 
  // The vector in position pos is generated 

  double cutoff,tot,condprob; 
  int aux,j,i,k; 

    if (DoReArrangeTrees)  ReArrangeTree(); 
    j = 0;
 
     for (i=0;i<length; i++ ) 
	 { 
          j = NextInOrder(i); 
	  cutoff = (1-myrand()); 
          while(cutoff==0) cutoff = myrand();	 
          k = 0; tot = 0;
	  if (Tree[j]==-1)  
	  { 
	    while ( (cutoff>tot) && (k<Card[j])) 
	    {  
		 tot += (AllProb[IndexUnivEntries[j]+k]*actualpoolsize + Prior)/(actualpoolsize+ Card[j]*Prior); 
		 k++; 
            } 
	    /* if(pos==5)  cout<<" j "<<j<<" T[j] "<<Tree[j]<<" X(T[j])= "<<NewPop->P[pos][Tree[j]]<<"  AllProb "<<AllProb[IndexUnivEntries[j]+k-1]<<" cutoff "<<cutoff<<" tot "<<tot<< endl;*/
	    //NewPop->P[pos][j] = k-1; 
          }  
	  else  
	  {	     	 
           if (j<Tree[j])  
	    {
              aux = j*(2*length-j+1)/2 +Tree[j]-2*j-1; 
              while ( (cutoff>tot) && (k<Card[j])) 
	      {  
	  condprob = (AllSecProb[IndexBivEntries[aux]+k*Card[Tree[j]] +NewPop->P[pos][Tree[j]]]*actualpoolsize +Prior) / (AllProb[IndexUnivEntries[Tree[j]]+NewPop->P[pos][Tree[j]]]*actualpoolsize + Prior*Card[Tree[j]]);
		  tot += condprob ;
               	  k++; 
		  /*	if(pos==5)  cout<<" j "<<j<<" T[j] "<<Tree[j]<<" X(T[j])= "<<NewPop->P[pos][Tree[j]]<<" BivEn "<<IndexBivEntries[aux]<<" k="<<k<<" cprob "<<AllSecProb[IndexBivEntries[aux]+(k-1)*Card[Tree[j]] +NewPop->P[pos][Tree[j]]] / AllProb[IndexUnivEntries[Tree[j]]+NewPop->P[pos][Tree[j]]]<<" conj "<<AllSecProb[IndexBivEntries[aux]+(k-1)*Card[Tree[j]] +NewPop->P[pos][Tree[j]]]<<" cutoff "<<cutoff<<" tot "<<tot<< endl; */
              }  
		  /* if(pos==5)  cout<<" "<<AllSecProb[IndexBivEntries[aux]+NewPop->P[pos][Tree[j]]]<<" "<<AllSecProb[IndexBivEntries[aux]+Card[Tree[j]]+NewPop->P[pos][Tree[j]]]<<" univ "<<AllProb[IndexUnivEntries[Tree[j]]]<<"  "<<AllProb[IndexUnivEntries[Tree[j]]+1]<<endl;*/
            }
           else
	   {
             aux = Tree[j]*(2*length-Tree[j]+1)/2 +j-2*Tree[j]-1; 
             while ( (cutoff>tot) && (k<Card[j])) 
	      {  
                condprob = (AllSecProb[IndexBivEntries[aux]+Card[j]*NewPop->P[pos][Tree[j]]+k]*actualpoolsize + Prior)/ (AllProb[IndexUnivEntries[Tree[j]]+NewPop->P[pos][Tree[j]]]*actualpoolsize + Prior*Card[Tree[j]]); 
		tot +=  condprob;
	        k++; 
/* if(pos==5)  cout<<" j "<<j<<" T[j] "<<Tree[j]<<" X(T[j])= "<<NewPop->P[pos][Tree[j]]<<" BivEn "<<IndexBivEntries[aux]<<" k="<<k<<" cprob "<<(AllSecProb[IndexBivEntries[aux]+Card[j]*NewPop->P[pos][Tree[j]]+k-1] / AllProb[IndexUnivEntries[Tree[j]]+NewPop->P[pos][Tree[j]]])<<" conj "<<(AllSecProb[IndexBivEntries[aux]+Card[k-1]*NewPop->P[pos][Tree[j]]+k-1])<< " cutoff "<<cutoff<<" tot "<<tot<<endl;*/
              }
	     /*    if(pos==5)  cout<<" "<<AllSecProb[IndexBivEntries[aux]+Card[0]*NewPop->P[pos][Tree[j]]]<<" "<<AllSecProb[IndexBivEntries[aux]+Card[1]*NewPop->P[pos][Tree[j]]+1]<<" univ "<<AllProb[IndexUnivEntries[Tree[j]]+0]<<" "<<AllProb[IndexUnivEntries[Tree[j]]+1]<<endl; */
	    
          }     
          }
           NewPop->P[pos][j] = k-1; 
	 } 
   
 }




void IntTreeModel::GenPartialIndividual (Popul* NewPop, int pos, int nvars, int *vector,int cprior) 
 { 
  // The vector in position pos is generated 
    
} 


void IntTreeModel::UpdateModel(double *vector,int howmany,Popul* EPop) 
{ 
  CalProbFvect(EPop,vector,howmany,0); 
  rootnode = RandomRootNode();  
  CalMutInf(); 
  MakeTree(rootnode);   
} 

void IntTreeModel::UpdateModel() 
{ 
  CalProb(); 
  rootnode = RandomRootNode(); 
  CalMutInf(); 
  MakeTree(rootnode); 
} 



void IntTreeModel::ImportMutInf( double* MI ) 
{ 
 int i; 
    for(i=0; i<length*(length-1)/2; i++) MutualInf[i] = MI[i]; 
} 

void IntTreeModel::ImportMutInfFromTree( IntTreeModel* other ) 
{
    ImportMutInf(other->MutualInf);
}  


 
void IntTreeModel::ImportProb(double* b_prob, double* u_prob) 
{ 
 int j; 
  for (j=0;j<TotBivEntries;j++) AllSecProb[j] = b_prob[j]; 
  for (j=0;j<TotUnivEntries;j++) AllProb[j] = u_prob[j];  
} 

void IntTreeModel::ImportProbFromTree(IntTreeModel* other ) 
{ 
   ImportProb(other->AllSecProb, other->AllProb); 

} 



void IntTreeModel::PutInMutInfFromTree( IntTreeModel* other ) 
{
    int i,aux; 
    for(i=0; i<length; i++) 
       	 if(other->Tree[i]>-1)
         {
           if (other->Tree[i]<i) 
                 aux = other->Tree[i]*(2*length-other->Tree[i]+1)/2 +i-2*other->Tree[i]-1; 
	    else aux = i*(2*length-i+1)/2 +other->Tree[i]-2*i-1; 
            MutualInf[aux] = -100;
         }	
} 

/* Es necesario definir las siguientes funciones para el mixture */



int IntTreeModel::FindRootNode() 
 { 
	 // El nodo raiz del arbol se puede escoger aleatoriamente, 
	 // Siguiendo a De Bonet aqui se escoge el de menor entropia incondicional 
	// Se determina la variable con menor entropia incondicional 

        int i,j; 
	double min; 
        int minindex=0; 
	double aux=0; 
     
       for(i=0; i<(IndexUnivEntries[1]-IndexUnivEntries[0]); i++)
    	     min-= (AllProb[i]*log(AllProb[i])); 
 
	for(j=1; j<length; j++) 
	  {
           aux = 0;
           for(i=0; i<(IndexUnivEntries[j+1]-IndexUnivEntries[j]); i++)
	       aux -= (AllProb[i]*log(AllProb[i]));
	   if (aux<=min) 
	    { 
				minindex=j; 
				min=aux; 
	    } 
	   } 
 
  return minindex; 
 } 
 

 
// Implementacion del Algoritmo de Constraints Simple




SContraintUnivariateModel::SContraintUnivariateModel(int vars, int constr):UnivariateModel(vars) 
{ 
    NormAllProb = new double[length];  
    constraint = constr;
} 


void SContraintUnivariateModel::CalProbFvect(Popul* pop, double *vector, int npoints) 
 { 
   int i,j; 
   double TotProb;
    for(j=0; j<length; j++) 
      {
       AllProb[j] = 0;
       NormAllProb[j] = 0;
      }       
        TotProb = 0;	

	for(j=0; j<length; j++)
        {
	  for(i=0; i<npoints; i++) 
	  { 
	   if (pop->P[i][j]==1) AllProb[j] += vector[i]; 
	  }        
          TotProb +=  AllProb[j];
        }
        for(j=0; j<length; j++)  NormAllProb[j] = AllProb[j]/TotProb;
	 
}  

void SContraintUnivariateModel::GenIndividual (Popul* NewPop, int pos) 
{ 
 
double cutoff,tot,current_total; 
int i,j,tobeset; 
int* index; 
 
// The vector in position pos is generated 
// First the unitation value o 

current_total = 1; 
 
if (constraint ==0 || constraint == length) 
{   
	for (i=0; i<length; i++) NewPop->P[pos][i] = constraint/length; 
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
while (i< constraint) 
{ 
  cutoff = myrand() * current_total;	 
  j = 0; 
  tot = 0; 
   if(cutoff>0) 
   { 
	 while ( (cutoff>tot) && (j<length-i) ) 
	 {  
		 tot += NormAllProb[index[j]]; 
		 j++; 
          }  
	 j--; 
   } 
   current_total -=  NormAllProb[index[j]]; // The prob. are updated
   NewPop->P[pos][index[j]] = 1;            // The vector is updated
	 index[j] = index[length-i-1];      // The candidates are updated
	 i++; 
} 

  delete[] index; 
  return; 
} 
 

void SContraintUnivariateModel::SetPrior(double prior_r_univ,double prior_r,int N) 
 { 
   int j;
   double  TotProb;
   TotProb = 0;

    	 for(j=0; j<length; j++) 
	  { 
	    AllProb[j]= (AllProb[j]*N+prior_r_univ)/ (N+ 2*prior_r_univ); 
            TotProb +=  AllProb[j];
          }  	

  

   for(j=0; j<length; j++)  NormAllProb[j] = AllProb[j]/TotProb;

}   
 
SContraintUnivariateModel::~SContraintUnivariateModel() 
{ 
    delete[] NormAllProb;  
} 
 
