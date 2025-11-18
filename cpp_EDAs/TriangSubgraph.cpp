#include <math.h> 
#include <stdlib.h>   
#include <stdio.h>   
#include <string.h>   
#include "auxfunc.h"   
#include "TriangSubgraph.h"   
   
using namespace std;


clique::clique(int firstnode,int k)   
{   
	MaxLength = k;   
	vars = new int[MaxLength];
	  
	exp_card = (int*) 0;   
	marg = (double*) 0;   
	vars[0] = firstnode;   
	NumberVars = 1;    
	parent = -1;   
}   
   
clique::~clique()   
{     
	   
	delete[] vars;   
        if (exp_card != (int*) 0) delete[] exp_card;   
	//if (cards != (unsigned*) 0) delete[] cards;   
	   
	if (marg != (double*)0)   
	{   
	 delete[] marg;   
	}   
}   
   
void clique::print()   
{   
	int i;   
     for(i=0;i<NumberVars;i++) printf("%d ", vars[i]);   
	 printf("\n ");   
}   
   
void clique::Instantiate(int* vector)       
{   
	int j;   
        double  cutoff,tot;   
	   
	   
	 cutoff = myrand();	   
	 j = 0;   
	 tot = marg[0];   
         //cout<<j<<" "<<marg[0]<<" cutoff "<<cutoff<<" tot "<<tot<<endl; 
	 while ( (tot<cutoff) && (j<r_NumberCases) )   
	 {    
		 j++;   
		 tot += marg[j];   
                 //cout<<j<<" "<<marg[j]<<" tot "<<tot<<endl;   
     }    
        
	 // If the following happens there's a mistake   
	 if (j==r_NumberCases) 
           {
	       // cout<<"Algo aqui"<<endl; 
		 j = randomint(r_NumberCases);    
           }
 
	 PutCase(j,vector);   
          
}   
   
void clique::CompMargFromFather(clique* father)
{
    int* auxindex;
    int* auxcase;
    int i;

    auxindex = new int[NumberVars];
    auxcase = new int[father->NumberVars];

    for(i=0; i<NumberVars; i++) auxindex[i] = father->PosVarInClique(vars[i]);
    
    for (i = 0;i<father->r_NumberCases;i++)
      {
       father->PutCase_fv(auxindex,i,auxcase);
       marg[rvars_index_fv(auxindex,auxcase)] += father->marg[i];   
       marg[r_NumberCases] ++;   
      }
           
    delete[] auxindex;
    delete[] auxcase;
}

void clique::Instantiate(int* vector, int min, int max,int nvars, int* sum, int* varsinst)       
{   
	int j;   
        double  cutoff,tot;   
	int cond1, cond2,sumcliq,count;   
	sumcliq = 0; //rinit   
   
	 cutoff = myrand();	   
	 j = 0;   
	 tot = 0; //marg[o_index][0];   
	 cond1 = 0; cond2=0;   
	    
	 while ( ((tot<cutoff) || !(cond1 && cond2)) && (j<r_NumberCases) )   
	 {    
		 sumcliq = Sum_and_PutCase(j,vector);   
		 cond1 = (*sum+sumcliq) <= max;   
                 cond2 = (nvars-*varsinst-NumberVars) >= min-sumcliq-*sum;   
		 if (cond1 && cond2) tot += marg[j];   
		 else    
		 {   
			 cutoff -= marg[j];   
         }   
		 j++;   
     }    
        
	 // If the following happens there's a mistake   
	 if (!(cond1 && cond2))    
	 {       
	     j = randomint(r_NumberCases);   
		 count = 0;   
	     cond1 = 0;     
		 while ( !(cond1 && cond2) && (count<r_NumberCases) )   
		{    
		 sumcliq = Sum_and_PutCase(j,vector);   
		 cond1 = (*sum+sumcliq) <= max;   
         cond2 = (nvars-*varsinst-NumberVars) >= min-sumcliq-*sum;   
		 j++;   
		 count++;   
		 if (j==r_NumberCases) j=0;        
		}    
	 }	    
	    
   
      (*sum) += sumcliq;   
      (*varsinst) += NumberVars;   
	      
}   
 
int clique::VarIsInClique(int node)   
{    
	int i;   
	i=0;   
	while (i<NumberVars && vars[i] != node) i++;   
	return(i<NumberVars);   
 }   
 
int clique::PosVarInClique(int node)   
{    
	int i;   
	i=0;   
	while (i<NumberVars && vars[i] != node) i++;   
        if(i==NumberVars) return -1;
	return(i);   
 }  


void clique::PutCase(int I, int* caso )   
{   
  int i;   
   for (i=NumberVars-1; i>-1; i--)   
	{   
	  caso[vars[i]] = I / exp_card[i];   
	  I = I % exp_card[i];   
	}   
}   


void clique::PutCase(int I, unsigned int* caso )   
{   
  int i;   
   for (i=NumberVars-1; i>-1; i--)   
	{   
	  caso[vars[i]] = I / exp_card[i];   
	  I = I % exp_card[i];   
	}   
}   


int clique::GetValVar(int I, int* caso, int targvar)   
{ //Gives the value and prob. for one of the variables' given the conf   
  int i,val,found;   
  val = -1; found=0;   
  i=NumberVars-1;   
   while (i>-1 && !found)   
	{   
	  val = I / exp_card[i];   
	  I = I % exp_card[i];   
          found = (vars[i]==targvar);   
          i--;   
	}   
   return val;   
}   
   
void clique::GetValProb(int* caso, int tarvar, double* prob)   
{   
   //Gives the value and prob. for one of the variables' given the conf   
  int i;   
  prob[0]=0; prob[1]=0;   
    
 for(i=0; i<r_NumberCases; i++)  
   {   
     if(CheckCase(i,caso ))   
     {   
      if(GetValVar(i,caso,tarvar)==0) prob[0]+=marg[i];   
      else prob[1]+=marg[i];      
     }   
   }   
}   
   
int clique::CheckCase(int I, int* caso )   
{   
  int i,stillvalid;   
  stillvalid = 1;   
   
  i=NumberVars-1;   
   while(i>-1 && stillvalid)   
	{   
	  // cout<<caso[vars[i]]<<"-";
	  stillvalid = (caso[vars[i]] == -1  || caso[vars[i]] == I / exp_card[i]) ;   
	  I = I % exp_card[i];   
          i--;   
	}   
  return stillvalid;   
}   
   
void clique::PartialInstantiate(int* caso, int* auxvect)   
{ 
  int i,j,legalconf;   
    
  double tot,margtot,cutoff;   
  margtot = 0;   
  legalconf = 0;   
   
  //cout<<"r_NumberCases  "<<r_NumberCases<<endl;
  //print();
  //for (i=0; i<NumberVars; i++) cout<<caso[vars[i]]<<" ";
  //cout<<endl;   

 for (i=0; i<r_NumberCases; i++)   
  {   
      if(CheckCase(i,caso))   
       {	     
        auxvect[legalconf] = i;   
        legalconf++;   
        margtot += marg[i];
	//   cout<<"i "<<i<<" margtot "<<margtot<<endl;      
       }   
  }   

         cutoff = myrand()*margtot;	   
	 i = 0;   
	 tot = marg[auxvect[0]];   
	 //cout<<"H "<<auxvect[0]<<" "<<marg[auxvect[0]]<<endl;
	 while ( (tot<cutoff) && (i<legalconf) )   
	 {    
		 i++;   
		 tot += marg[auxvect[i]];   
                 //cout<<"i "<<i<<" tot "<<tot<<" cut "<<cutoff<<endl;   
         }    
        
	 // If the following happens there's a mistake   
	 if (i>=legalconf)   
	 {  
            cout<<"Algo justo aqui"<<endl;   
       	   i = randomint(legalconf);   
	   /*
                for (j=0;j<NumberVars;j++) cout<<vars[j]<<" ";
                 cout<<endl;
                 for (j=0;j<NumberVars;j++) cout<<auxvect[j]<<" ";
                 cout<<endl;
                 for (j=0;j<NumberVars;j++) cout<<caso[vars[j]]<<" ";
                 cout<<endl;
	   */

         }   
   
	 PutCase(auxvect[i],caso);   
         //for (i=0; i<NumberVars; i++) cout<<caso[vars[i]]<<" ";
         //cout<<endl;     
}   
   
void clique::GiveProb(int* caso, double* legprob)   
{ 
  int i,legalconf;   
    
  double margtot;   
  margtot = 0;   
  legalconf = 0;   
   
  //for(i=r_NumberCases-1; i>0; i--)  
    for (i=0; i<r_NumberCases; i++)   
    {   
      if(CheckCase(i,caso))   
       {	     
        legprob[legalconf] = marg[i]; 
        legalconf++;  
        margtot += marg[i];   
        //cout<<"i "<<i<<" marg "<<marg[i]<<" to "<<margtot<<endl;   
       }   
    }  

  //for (i=0; i< legalconf; i++) legprob[i] /=  margtot; 
            
}   
   


void clique::CondProb(int* caso, int* auxvect)   
{ //Here auxvect is an auxiliary variable that belongs to the class to   
    // avoid memory allocation (bad practice but fast)   
  int i,legalconf;   
    
  double tot,margtot,cutoff;   
  margtot = 0;   
  legalconf = 0;   
   
 for (i=0; i<r_NumberCases; i++)   
  {   
      if(CheckCase(i,caso))   
       {	     
        auxvect[legalconf] = i;   
        legalconf++;   
        margtot += marg[i];   
       }   
  }   
// there must be at least one legal configuration   
         cutoff = myrand()*margtot;	   
	 i = 0;   
	 tot = marg[auxvect[0]];   
	   
	 while ( (tot<cutoff) && (i<legalconf) )   
	 {    
		 i++;   
		 tot += marg[auxvect[i]];   
                 //cout<<"i "<<i<<" tot "<<tot<<" cut "<<cutoff<<endl;   
         }    
        
	 // If the following happens there's a mistake   
	 if (i>=legalconf)   
	 {  cout<<"Algo aqui"<<endl;   
		 i = randomint(legalconf);    
         }   
   
	 PutCase(auxvect[i],caso);   
          
}   
   
int clique::NumberInstantiated(int* caso )   
{   
  int i,numberinst;   
   
  numberinst = 0;   
  for (i=0; i<NumberVars; i++) numberinst += (caso[vars[i]] != -1);   
    
  return numberinst;   
}   
   
int clique::Sum_and_PutCase(int I, int* caso )   
{   
  int i;   
  int sum;   
  sum = 0;   
   for (i=NumberVars-1; i>-1; i--)   
	{   
	  caso[vars[i]] = I / exp_card[i];   
	  sum += caso[vars[i]];   
	  //sum += (I / exp_card[r_vars[i]]);   
	  I = I % exp_card[i];   
	}   
   return sum;   
}   
   
   
double clique::Prob(unsigned* vector)       
{   
 double auxprob;   
   
 if (marg[r_NumberCases]>0)   
  auxprob = marg[rvars_index(vector)];    
  else auxprob = 0;   
  return auxprob;   
}   
    
   
void clique::Compute(unsigned* vector, double ProbVect)     //Prob. of the dif. vectors in the population )       
{   
	 marg[rvars_index(vector)]+= ProbVect;   
	 marg[r_NumberCases] ++;   
}   


   
void clique::printmarg()     //Prob. of the dif. vectors in the population )       
{   
	int i;   
	for (i = 0;i<r_NumberCases;i++) cout<<marg[i]<<"  ";   
        cout<<endl;   
}   
 
  
void clique::Normalize(int genepoollimit)     //Normalize the marginals considering priors   
{ int i;   
	for (i = 0;i<r_NumberCases;i++) marg[i] = (marg[i]*genepoollimit+1.0)/(genepoollimit+r_NumberCases);   
}   

void clique::NormalizeBoltzmann(double t)     //Normalize the marginals considering priors   
{ int i;   
  double totprob = 0;
  double base = 2.7182818;

  for (i = 0;i<r_NumberCases;i++)
    {
      //cout<<i<<"  "<< marg[i]<<" --- "; 
      marg[i] =  pow(base,marg[i]/t);
      totprob += marg[i]; 
      //cout<<i<<"  "<< marg[i]<<"  "<<totprob<<endl; 
    }
  for (i = 0;i<r_NumberCases;i++) marg[i] = marg[i]/totprob;
}   


 void clique::Normalize(int genepoollimit, double PPrior)     //Normalize the marginals considering priors   
{ int i;   
 for (i = 0;i<r_NumberCases;i++) marg[i] =  1.0/r_NumberCases; //(marg[i]*genepoollimit+PPrior)/(genepoollimit+r_NumberCases*PPrior);   
}   

   
void clique::Fill_exp_card(unsigned* cards)   
{   
  int i, j;   
     
  for (i = 0;i<NumberVars;i++)     
   for (j=0;j<i;j++)  exp_card[i] *= cards[vars[i]];   
}   
   
 
  
int clique::rvars_index_fv(int* auxarray, int* Casos)   //index from aux. array
{   
  int i,auxresp;   
   
  auxresp = 0;   
  for (i = 0; i<NumberVars; i++)   
	  auxresp = auxresp + exp_card[i]*Casos[auxarray[i]];   
  return auxresp;   
} 

 void clique::PutCase_fv(int* auxarray, int I, int* caso )   
{   
  int i;   
   for (i=NumberVars-1; i>-1; i--)   
	{   
	  caso[i] = I / exp_card[i];   
	  I = I % exp_card[i];   
	}   
}   
   

 

int clique::rvars_index(unsigned* Casos)   
{   
  int i,auxresp;   
   
  auxresp = 0;   
  for (i = 0; i<NumberVars; i++)   
	  auxresp = auxresp + exp_card[i]*Casos[vars[i]];   
  return auxresp;   
}   
   
void clique::CreateMarg(unsigned* card)   
{     
	int i;   
   
  if( exp_card == (int*)0 ) //Marginals are initialized	   
  {   
    exp_card = new int[NumberVars];   
	//memset(exp_card , 1, sizeof(int)*(NumberVars));   
    for(i=0;i<NumberVars;i++) exp_card[i] = 1;   
    Fill_exp_card(card);   
    r_NumberCases = 1;         
    for(i=0;i<NumberVars;i++)  r_NumberCases *= card[vars[i]];   
    marg = new double[r_NumberCases+1];   
    memset(marg , 0, sizeof(double)*(r_NumberCases+1) );   
   }   
  else    
    memset(marg , 0, sizeof(double)*(r_NumberCases+1) );       
     
}   
   
void clique::Add(int node)   
{    
	 
	vars[NumberVars++] = node;   
}   
   
   
memberlistclique::memberlistclique(int cq,memberlistclique* nextcq)   
{   
	cliq = cq;   
	nextcliq = nextcq;   
}   
   

memberlistclique::~memberlistclique()   
{   
  nextcliq = (memberlistclique*)0 ;   
  //if (nextcliq != (memberlistclique*)0) delete nextcliq;
}   
   
maximalsubgraph::maximalsubgraph(int nnodes, unsigned int** Matrix, int CliqMLength, int MaxNCliq)   
{     
	int i,j;   
   
	Adjacency_Matrix = Matrix;   
	MaxNumberOfCliquesInTheGraph = MaxNCliq;   
	NumberCliques = 0;   
    NumberNodes = nnodes;   
    CliqueMaxLength = CliqMLength;   
    AddedEdges = 0;   
	TotalEdges = 0;   
       
    ListCliques = new clique*[MaxNumberOfCliquesInTheGraph];   
    CliquesSizes = new int[MaxNumberOfCliquesInTheGraph];   
    CliquesPerNodes = new memberlistclique*[NumberNodes];     
    Nodes_Degrees = new int[NumberNodes];     
    Nodes_Order = new int[NumberNodes];   
    compsub = new int[NumberNodes+1];   
    c = 0;   
    for(i=0;i<NumberNodes;i++)    
	{   
		CliquesPerNodes[i] = (memberlistclique*) 0; 
	

        Nodes_Degrees[i] = 0;   
		Nodes_Order[i] = i;   
	}    
        
    if(Adjacency_Matrix != (unsigned int**)0)
      {
	 for(i=0;i<NumberNodes;i++)    
		for(j=i+1;j<NumberNodes;j++)    
		{    
          TotalEdges += Adjacency_Matrix[i][j];   
          Nodes_Degrees[i] += Adjacency_Matrix[i][j];   
		  Nodes_Degrees[j] += Adjacency_Matrix[i][j];   
		}   
      }	   
	for(i=0;i<MaxNumberOfCliquesInTheGraph;i++) ListCliques[i] = (clique*)0;   
}	   
   
maximalsubgraph::maximalsubgraph(int nnodes,unsigned int** Matrix, int CliqMLength, int MaxNCliq, int* order)   
{     
	int i,j;   
   
	Adjacency_Matrix = Matrix;   
	MaxNumberOfCliquesInTheGraph = MaxNCliq;   
	NumberCliques = 0;   
    NumberNodes = nnodes;   
    CliqueMaxLength = CliqMLength;   
    AddedEdges = 0;   
	TotalEdges = 0;   
       
    ListCliques = new clique*[MaxNumberOfCliquesInTheGraph];    
    CliquesSizes = new int[MaxNumberOfCliquesInTheGraph];   
    CliquesPerNodes = new memberlistclique*[NumberNodes];     
    Nodes_Degrees = new int[NumberNodes];     
	Nodes_Order = new int[NumberNodes];   
  compsub = new int[NumberNodes+1];   
    c = 0;   
    for(i=0;i<NumberNodes;i++)    
	{   
		CliquesPerNodes[i] = (memberlistclique*) 0;   

        Nodes_Degrees[i] = 0;   
		Nodes_Order[i] = order[i];   
	}    

   if(Adjacency_Matrix != (unsigned int**)0)
      {
	 for(i=0;i<NumberNodes;i++)    
		for(j=i+1;j<NumberNodes;j++)    
		{    
          TotalEdges += Adjacency_Matrix[i][j];   
          Nodes_Degrees[i] += Adjacency_Matrix[i][j];   
		  Nodes_Degrees[j] += Adjacency_Matrix[i][j];   
		}   
      }	  

	   
	for(i=0;i<MaxNumberOfCliquesInTheGraph;i++) ListCliques[i] = (clique*)0;   
}	   
   
maximalsubgraph::~maximalsubgraph()   
{     
	int i;   
        memberlistclique* auxmemberlist;   
	memberlistclique* othermemberlist;   
      
	//cout<<"Now it is here deleting "<<endl;
        //  for(i=0;i<NumberNodes;i++) cout<<i<<" "<<CliquesPerNodes[i]<<endl;
           
	/*
    for(i=0;i<NumberNodes;i++)    
	{    
	     
	  cout<<i<<" Init  "<<CliquesPerNodes[i]<<endl;
        
		auxmemberlist = CliquesPerNodes[i];   
 
		while (auxmemberlist != 0)   
		{   
		   cout<<i<<" auxmember "<<auxmemberlist<<endl;                                   
                   othermemberlist = auxmemberlist->nextcliq;   
                   cout<<"nextmember "<<othermemberlist<<endl;
		   if(othermemberlist!= 0) delete auxmemberlist;   
		   auxmemberlist = othermemberlist;   
                }
	  
          //delete CliquesPerNodes[i];		   
	}   
        
	*/
	for(i=0;i<NumberCliques;i++) 
           {
            
             delete ListCliques[i];
               
	  }

   
    delete[] ListCliques;   
    delete[] CliquesSizes;   
    delete[] CliquesPerNodes;     
    delete[] Nodes_Degrees;   
	delete[] Nodes_Order;   
        delete[] compsub;   
    
}	   
   
void maximalsubgraph::CardinalityOrdering()   
{   
 int i,j,aux;   
 for(i=0;i<NumberNodes;i++)    
	for(j=i+1;j<NumberNodes;j++)    
		if(Nodes_Degrees[Nodes_Order[i]]< Nodes_Degrees[Nodes_Order[j]])   
		{   
			aux = Nodes_Order[i];   
			Nodes_Order[i] = Nodes_Order[j];   
            Nodes_Order[j] = aux;   
		}   
}   
   
void maximalsubgraph::MaximumCardinalityLexicalOrdering()   
{   
	CardinalityOrdering();   
	LexicalOrdering();   
}   
   
void maximalsubgraph::UpdateLex(int index, unsigned**& Lex)   
{   
	int i;   
    for(i=0;i<NumberNodes;i++) Lex[index][i] = 2;   
}   
   
   
int maximalsubgraph::FindNextInLex(int position,unsigned** Lex)   
{   
	int step,i,j,NextInLex;   
      
	step = 0;    
	while (step < NumberNodes && Lex[Nodes_Order[step]][0]>1) step++;   
	NextInLex = Nodes_Order[step]; // Find the first node that has not been visited yet   
   
   
    for(j=0;j<NumberNodes;j++)   
	  {   
		  step = 0;   
		  i = Nodes_Order[j];   
		  if(Lex[i][step] < 2)   
		  {   
           while (step<=position && Lex[i][step]==Lex[NextInLex][step]) step++;   
		   if (step<=position && Lex[i][step] > Lex[NextInLex][step])    
           NextInLex = i;   
		  }     
	  }   
	return NextInLex;   
}   
   
void maximalsubgraph::LexicalOrdering()   
{ // Finds the LexicalOrdering given the arbitrary ordering contained in Nodes_Order   
   
 int i,j;   
 unsigned **Lex; // Lex(i,j) == 2 means i is already in the ordering,    
 int *auxorder;   
    
 auxorder = new int[NumberNodes];   
 Lex = new unsigned*[NumberNodes];   
 for(i=0;i<NumberNodes;i++) Lex [i] = new unsigned[NumberNodes];   
   
 for(i=0;i<NumberNodes;i++)    
	for(j=0;j<NumberNodes;j++) Lex [i][j] = 0;   
     
 auxorder[0] = Nodes_Order[0];   
 UpdateLex(auxorder[0],Lex);   
    
 for(i=0;i<NumberNodes-1;i++)    
 {   
   for(j=0;j<NumberNodes;j++)   
      if(Lex [j][i] < 2 && Adjacency_Matrix[auxorder[i]][j]) Lex [j][i] = 1; // if j is a neighbor   
   auxorder[i+1] = FindNextInLex(i,Lex);   
   UpdateLex(auxorder[i+1],Lex);   
 }   
      
for(i=0;i<NumberNodes;i++)  Nodes_Order[i] = auxorder[i]; 	      
   
delete[] auxorder;   
for(i=0;i<NumberNodes;i++)  delete[] Lex[i]; 	      
delete[] Lex;   
         
}   
   
   
void  maximalsubgraph::version2(int* old, int ne, int ce)   
{   
    int nod, fixp;   
    int newne, newce, i,j,count, pos,p,s,sel,minnod;   
    int* newarray;   
    fixp = 0; newce =0; pos = 0; s=0;  //rinit   
    newarray = new int[NumberNodes+1];   
    minnod = ce; nod = 0; i=1;   
    while (i<=ce && minnod !=0)   
    {   
	p = old[i]; count = 0; j = ne+1;   
        while (j<=ce && count<minnod)   
         {   
	     if((Adjacency_Matrix[p-1][old[j]-1])==0)   
               {   
                count++;   
                pos = j;   
               }   
             j++;   
         }   
           
             if(count<minnod)   
	       {   
	        fixp = p; minnod = count;   
                if (i<=ne) s = pos;   
                else   
	        {   
		  s = i; nod=1;   
                }   
   
               }   
	i++;    
    }   
   
   
//BACKTRACKCYCLE   
   
   for(nod=minnod+nod; nod>0; nod--)   

  {   
       p = old[s]; old[s]=old[ne+1];   
       sel = p;  old[ne+1] = p;   
       newne = 0;   
       for(i=1; i<=ne; i++)   
       {   
        if(Adjacency_Matrix[sel-1][old[i]-1])   
        {   
	    newne++; newarray[newne] = old[i];   
        }   
       }   
   
       newce = newne;   
   
       for(i=ne+2; i<=ce; i++)   
       {   
        if(Adjacency_Matrix[sel-1][old[i]-1])   
        {   
	    newce++; newarray[newce] = old[i];   
        }   
       }   
   
        c++; compsub[c] = sel;   
        if (newce==0)   AddClique(compsub,c);   
        else if(newne<newce) version2(newarray,newne,newce);   
        c--; ne++;   
        if(nod>1)   
        {   
	    
  s = ne+1;   
            while(Adjacency_Matrix[fixp-1][old[s]-1]) s++;   
        }   
      }   
   delete[] newarray;    
}      
   
void maximalsubgraph::FindAllCliques()   
{   
    int i;   
    int *oldarray;   
     
    oldarray = new int[NumberNodes+1];   
    for(i=1;i<=NumberNodes;i++) oldarray[i] = i;   
    version2(oldarray,0,NumberNodes);   
    delete[] oldarray;   
}   
   
   
void maximalsubgraph::AddClique(int *allnodes, int nnodes)   
{   
  int i;   
  clique* newcliq;  
  memberlistclique *Auxmclique,*Auxmclique1;
      //cout<<NumberCliques<< " nnodes "<<nnodes<<endl;
      CliquesSizes[NumberCliques] = nnodes;   
      //cout<<CliquesSizes[NumberCliques]<<" "<<allnodes[1]<<"  "<<CliqueMaxLength<<endl;

      newcliq = new clique(allnodes[1]-1,nnodes);   
      NumberCliques++;   
      ListCliques[NumberCliques-1] = newcliq;   
      //cout<<"Clique "<<NumberCliques<<"----";   
  for(i=1; i<=nnodes;i++)   
  {   
      if(i>1) newcliq->Add(allnodes[i]-1);   
      //cout<<" "<<allnodes[i]-1;   
      /*
      Auxmclique = new memberlistclique(NumberCliques-1,(memberlistclique*)0);
      if (CliquesPerNodes[allnodes[i]-1]== (memberlistclique*)0) 
       {	 
        CliquesPerNodes[allnodes[i]-1] = Auxmclique;
        CliquesPerNodes[allnodes[i]-1]->nextcliq = (memberlistclique*)0;
       }
      else
        {
	  Auxmclique1 = CliquesPerNodes[allnodes[i]-1];
          while(Auxmclique1->nextcliq !=(memberlistclique*)0) Auxmclique1 = Auxmclique1->nextcliq;
          Auxmclique1->nextcliq = Auxmclique;
        }
       
            
      //Auxmclique = new memberlistclique(NumberCliques-1,CliquesPerNodes[allnodes[i]-1]);        
     cout<<endl<<i-1<<" == "<<CliquesPerNodes[allnodes[i]-1]<<" "<<Auxmclique<<" "<<Auxmclique->nextcliq<<endl;   
     //CliquesPerNodes[allnodes[i]-1] = Auxmclique;  
     */    
  }   
  
  //cout<<endl;   
}   
   
   
   
void maximalsubgraph::UpdateListFromClique(clique *newcliq,int NodeToAdd,int IndCliq)   
{   
  int i;   
     
  CliquesPerNodes[NodeToAdd] = new memberlistclique(IndCliq,(memberlistclique*)0);        
  for(i=0; i<newcliq->NumberVars;i++)   
  {   
     if(newcliq->vars[i] != NodeToAdd)   
    CliquesPerNodes[newcliq->vars[i]] = new memberlistclique(IndCliq,CliquesPerNodes[newcliq->vars[i]]);// The new clique is inserted from the beginning        
  }   
}   
   
void maximalsubgraph::CreateGraphCliques()   
{   
   FindAllCliques();   
   //printCliques();  
}       
   


void maximalsubgraph::CreateGraphCliquesProtein(int typemodel, int sizecliqp)   
{
   int i,j;
   if (typemodel==1) //Normal Model 
     {   
      for(j=0;j<NumberNodes-sizecliqp;j+=1)
       {
         compsub[0] = sizecliqp+1;
         for(i=j;i<j+sizecliqp+1;i++) compsub[i-j+1] = i+1;
         AddClique(compsub,sizecliqp+1);   
       }
     }
   else // Compact Model
     {
     }  
}


void maximalsubgraph::CreateGraphCliquesProteinMPM(int typemodel, int ncliques,unsigned long** listclusters)   
{
   unsigned long i,j;
  
   if (typemodel==2) // Marginal Product Model (for Affinity EDA)
     {   
      for(i=0;i<ncliques;i++)
       {
         compsub[0] = listclusters[i][0];
         for(j=1; j<listclusters[i][0]+1; j++ )  compsub[j] = (int) (listclusters[i][j]+1);
         AddClique(compsub,listclusters[i][0]);   
       }
     }
}



/*
void maximalsubgraph::CreateGraphCliquesProtein(int typemodel, int sizecliqp)   
{
   int i,j,lastvarinclique,scliq;
 
  if (typemodel==1) //Normal Model 
     {   
      for(j=0;j<NumberNodes-1;j+=sizecliqp)
       {
        
         if(j+sizecliqp+1 < NumberNodes-1)
           {
            lastvarinclique = j+sizecliqp+1;
            scliq = sizecliqp+1;
           }
	 else
           {
             lastvarinclique = NumberNodes;
             scliq =  NumberNodes - j;
           }  
          compsub[0] = scliq;

        for(i=j;i<lastvarinclique;i++) 
          {
            compsub[i-j+1] = i+1;
	    // cout<<i+1<<" ";
          }
	//cout<<endl;
         AddClique(compsub,scliq);   
       }
     }
   else // Compact Model
     {
     }  
}

*/

  


void maximalsubgraph::printCliques()   
{   
   int i;   
   for(i=0; i<NumberCliques;i++)  ListCliques[i]->print();	   
}   
   
   
void maximalsubgraph::printOrdering()   
{   
   int i;   
   for(i=0; i<NumberNodes;i++)  printf("%d ", Nodes_Order[i]);	   
   printf( "\n");   
}     
   
void maximalsubgraph::printNodesDegrees()   
{   
   int i;   
   for(i=0; i<NumberNodes;i++)  printf("%d ", Nodes_Degrees[i]);	   
   printf( "\n");   
}   
   
//********************** Procedures for Kikuchi structures ***************************   
   
   
KikuchiClique::KikuchiClique(int firstnode,int k)   
{   
	MaxLength = k;   
	vars = new int[MaxLength];   
	vars[0] = firstnode;   
	NumberVars = 1;   
        count  = 1; 
        current = (clique*)0; 
}   
   
KikuchiClique::KikuchiClique(int firstnode,int k,clique* cliq,int signc)   
{   
        father = cliq; 
	MaxLength = k;   
        sign = signc;   
        count = 1;  
     	vars = new int[MaxLength];   
	vars[0] = firstnode;   
	NumberVars = 1;   
        current = (clique*)0; 
}   
   
KikuchiClique::~KikuchiClique()   
{     
	   
	delete[] vars;   
        if(current != (clique*)0) delete current;
}   
   
void KikuchiClique::print()   
{   
	int i;   
     for(i=0;i<NumberVars;i++) printf("%d ", vars[i]);   
	 printf("\n ");   
}   
   
 int KikuchiClique::VarIsInClique(int node)   
{    
	int i;   
	i=0;   
	while (i<NumberVars && vars[i] != node) i++;   
	return(i<NumberVars);   
 }   
   


int KikuchiClique::IdemCliques(KikuchiClique* Cliq)  
{  
    int i;  
    i = 0; 
    if (Cliq->NumberVars != NumberVars) return  0;  
    else  
    {  
     while (i<NumberVars && VarIsInClique(Cliq->vars[i])) i++;  
     return(i==NumberVars); //All the variables of the other clique are in this one  
    }  
}  
  
void KikuchiClique::Add(int node)   
{    
 vars[NumberVars++] = node;   
}   
   

void KikuchiClique::CompMargFromFather()
{
    current->CompMargFromFather(father);
}
  
void KikuchiClique::CreateMarg(unsigned* dim)
{
    current->CreateMarg(dim);
}  

memberlistKikuchiClique::memberlistKikuchiClique(KikuchiClique* Cliq)   
 {   
  KCliq = Cliq;   
  nextcliq = (memberlistKikuchiClique*)0;   
 }   
  
memberlistKikuchiClique::~memberlistKikuchiClique()   
 {   
     //if(KCliq != (KikuchiClique*)0) delete KCliq;   
 }   
   
KikuchiApprox::KikuchiApprox(int k, int mvar, int extravar)   
{ 
    int i;   
 MaxLength = k;  
 MaxVars = mvar;  
 currentvar = extravar;  //When currentvar = -1 (Gen Kik Approx, else Local Kik Approx)
 numberneighbors = 0;    
 neighbors = new int[MaxVars];  
 count_overl = new int[MaxVars];  
 NumberCliques = 0;   
 level = 0;  
 FirstKCliqLevels = new memberlistKikuchiClique*[MaxLength];  
 LastKCliqLevels  = new memberlistKikuchiClique*[MaxLength];  
   
 for(i=0;i<MaxVars;i++)   count_overl[i] = 0; 
  
 for(i=0;i<MaxLength;i++)  
   { 
    FirstKCliqLevels[i] = (memberlistKikuchiClique*) 0;   
    LastKCliqLevels[i] =  (memberlistKikuchiClique*) 0;  
   }  
    
}  
KikuchiApprox::~KikuchiApprox()   
{   
  int i;  
  memberlistKikuchiClique* auxmCliq;  
  
  for(i=0;i<MaxLength;i++)  
  {  
   while (FirstKCliqLevels[i] != (memberlistKikuchiClique*) 0)  
    {  
     auxmCliq=FirstKCliqLevels[i]->nextcliq; 
     delete FirstKCliqLevels[i]->KCliq;
     delete FirstKCliqLevels[i];   
     FirstKCliqLevels[i] = auxmCliq;  
    }  
  }  
  
    delete[] neighbors;  
    delete[] count_overl;  
    delete[] FirstKCliqLevels;  
    delete[] LastKCliqLevels;  
}  
  
KikuchiClique* KikuchiApprox::FindIntersection(clique* cliq1, clique* cliq2)  
 {    
	int i,nsolaps;   
      clique* father; // pointer to the father in the list of cliques  
  
      //cliq1->print();
      //cliq2->print();

      if (cliq1->NumberVars > cliq2->NumberVars) father = cliq1;  
      else father = cliq2;  
      nsolaps = 0;   
      KikuchiClique* SolapCliq = (KikuchiClique*)0 ;   
	   
         
	for(i=0;i<cliq1->NumberVars;i++)   
       {   
	   if (cliq1->vars[i] != currentvar && cliq2->VarIsInClique(cliq1->vars[i]))  
           {   
            nsolaps += 1;   
            if (nsolaps == 1)   
              SolapCliq = new KikuchiClique(cliq1->vars[i],MaxLength,father,-1);   
            else if (nsolaps > 1)   
              SolapCliq->Add(cliq1->vars[i]);
            //cout<<"nsolaps "<<nsolaps<<endl;   
           }   
        }   
	return SolapCliq;   
 }    
    

int  KikuchiApprox::ContainsClique(KikuchiClique* cliq1, clique* cliq2 )  
{    //The second clique is the container
     int i;   
       i = 0;
       while(i<cliq1->NumberVars && cliq2->VarIsInClique(cliq1->vars[i]))  i++;            
       return (i==cliq1->NumberVars);
 }    

int  KikuchiApprox::ContainsClique(KikuchiClique* cliq1, KikuchiClique* cliq2 )  
{    //The second clique is the container
     int i;   
       i = 0;
       while(i<cliq1->NumberVars && cliq2->VarIsInClique(cliq1->vars[i]))  i++;            
       return (i==cliq1->NumberVars);
 }    



 KikuchiClique* KikuchiApprox::FindIntersection(KikuchiClique* cliq1, KikuchiClique* cliq2)   
 {    
	int i,nsolaps;   
      clique* father; // pointer to the father in the list of cliques  
  
      if (cliq1->father->NumberVars > cliq2->father->NumberVars) father = cliq1->father;  
      else father = cliq2->father;  
  
      nsolaps = 0;   
      KikuchiClique* SolapCliq = (KikuchiClique*)0 ;   
	   
	for(i=0;i<cliq1->NumberVars;i++)   
       {   
	   if (cliq2->VarIsInClique(cliq1->vars[i]))   
           {   
            nsolaps += 1;   
            if (nsolaps == 1)   
              SolapCliq = new KikuchiClique(cliq1->vars[i],MaxLength,father,-1*cliq1->sign);   
            else if (nsolaps > 1)   
              SolapCliq->Add(cliq1->vars[i]);   
           }   
        }   
	return SolapCliq;   
 }   
  
 
KikuchiClique* KikuchiApprox::FindSelfIntersection(KikuchiClique* cliq1)   
 {    
     //Only the vars that are missing are incorporated 
 
      int i,nsolaps;   
      clique* father; // pointer to the father in the list of cliques  
      father = cliq1->father;  
      nsolaps = 0;   
      KikuchiClique* SolapCliq = (KikuchiClique*)0 ;   
	 
      if(cliq1->sign == -1) 
      { 
      	for(i=0;i<cliq1->NumberVars;i++)   
         {   
	   if (count_overl[cliq1->vars[i]]<1) 
           {   
            nsolaps += 1;   
            if (nsolaps == 1)   
              SolapCliq = new KikuchiClique(cliq1->vars[i],MaxLength,father,-1*cliq1->sign);   
             else if (nsolaps > 1)   
              SolapCliq->Add(cliq1->vars[i]);   
           }   
         } 
      } 
       else 
       { 
        for(i=0;i<cliq1->NumberVars;i++)   
         {   
	   if (count_overl[cliq1->vars[i]]>1)  
           {   
            nsolaps += 1;   
            if (nsolaps == 1)   
              SolapCliq = new KikuchiClique(cliq1->vars[i],MaxLength,father,-1*cliq1->sign);   
             else if (nsolaps > 1)   
              SolapCliq->Add(cliq1->vars[i]);   
           }   
         } 
       } 
      if (SolapCliq != (KikuchiClique*)0)  SolapCliq->count  = cliq1->count; 
	return SolapCliq;   
 }   
  
 
 
 
int KikuchiApprox::AlreadyFoundApproximation()  
    {  //Determines is all the variable are only once in the Kikuchi Approximation  
	int i;  
        i = 0;  
        while((i<numberneighbors) &&  (count_overl[neighbors[i]]==1)) i++;  
        return(i==numberneighbors);  
    }  
  
int KikuchiApprox::AlreadyFoundUniqueVars(KikuchiClique* cliq)  
    {  //Determines is the variables in the clique are only once in the Kikuchi Approximation  
	 
        int i;  
        i = 0;  
        if(cliq->sign==-1) 
         while(i<cliq->NumberVars &&  count_overl[cliq->vars[i]]<=1) i++;         
        else 
         while(i<cliq->NumberVars &&  count_overl[cliq->vars[i]]>=1) i++; 
 
        return(i==cliq->NumberVars);  
	 
    }  
 
void  KikuchiApprox::SimplifyAllCliques(int lev, int sign) 
{ 
   memberlistKikuchiClique  *firstpointer,*previouspointer,*nextpointer;   
   
   firstpointer =  FirstKCliqLevels[lev]; 
   previouspointer = (memberlistKikuchiClique*)0;
   
   while (firstpointer != (memberlistKikuchiClique*)0) 
     { 
	 if ((sign == -1 && firstpointer->KCliq->count > -1) || (sign == 1 && firstpointer->KCliq->count < 1)) 
	 {
	     nextpointer = firstpointer->nextcliq;
             firstpointer->KCliq->sign = -1*firstpointer->KCliq->sign; //ROBERTO 27-4-2004
             Insert(level+1,firstpointer->KCliq);
             delete firstpointer;
             if(previouspointer == (memberlistKikuchiClique*)0) FirstKCliqLevels[lev] = nextpointer;
             else previouspointer->nextcliq = nextpointer;
             firstpointer = nextpointer;
         } 
	 else
         { 
           CheckPresentVars(firstpointer->KCliq); 
           previouspointer = firstpointer;
           firstpointer = firstpointer->nextcliq; 
         } 
     } 
   //for(i=0;i<MaxVars;i++) cout<<count_overl[i]<<" ";  
   // cout<<" Simplified  "<<endl; 
} 

void  KikuchiApprox::FindCR(clique** ListCliques,  int ncliques,int lev) 
{ 
 int i;
 memberlistKikuchiClique  *firstpointer, *secondpointer;   
   
 firstpointer =  FirstKCliqLevels[lev];
    while (firstpointer != (memberlistKikuchiClique*)0) 
     {
      firstpointer->KCliq->count = 1;
      for(i=0;i<ncliques;i++) 
       if(ContainsClique(firstpointer->KCliq,ListCliques[i]))  firstpointer->KCliq->count--;
      for(i=0;i<=lev;i++) 
       {
         secondpointer =  FirstKCliqLevels[i];
 
         if(i==lev)
	  while (secondpointer != (memberlistKikuchiClique*)0  &&  secondpointer != firstpointer ) 
          {
            if(ContainsClique(firstpointer->KCliq,secondpointer->KCliq) && firstpointer->KCliq->count*firstpointer->KCliq->sign>0)
		firstpointer->KCliq->count -= secondpointer->KCliq->count;
            secondpointer = secondpointer->nextcliq;  
          }
         else         
          while (secondpointer != (memberlistKikuchiClique*)0  &&  secondpointer != firstpointer ) 
          {
            if(ContainsClique(firstpointer->KCliq,secondpointer->KCliq))
		firstpointer->KCliq->count -= secondpointer->KCliq->count;
            secondpointer = secondpointer->nextcliq;  
          }
       }
      //cout<<firstpointer->KCliq->count<<"Clique :  "; 
      //firstpointer->KCliq->print(); 
      firstpointer = firstpointer->nextcliq;   
    } 
}
  
int  KikuchiApprox::FindKikuchiApproximation(clique** ListCliques, int ncliques)  
{  
    int auxinters,s;
       auxinters=0; 
       level = 0;  
       
         auxinters = CreateFirstListCliquesInter(ListCliques,ncliques); 
           
	 if (auxinters==0) return 0; 
	 //for(int a=0;a<MaxVars;a++) cout<<count_overl[a]<<" ";
         //cout<<endl;
         
         s = -1;
         FindCR(ListCliques,ncliques,0);
         SimplifyAllCliques(0,s); 
	 //for(int a=0;a<MaxVars;a++) cout<<count_overl[a]<<" ";
         //cout<<endl;
         while(!AlreadyFoundApproximation() && level <= MaxLength)  
         {  
	     //cout<<"level "<<level<<endl;
           level++;  
           CreateOtherListCliquesInter(level);
           s *= -1;
           FindCR(ListCliques,ncliques,level);
           SimplifyAllCliques(level,s); 
	   //for(int a=0;a<MaxVars;a++) cout<<count_overl[a]<<" ";
	   //cout<<endl;
          }  
         return 1; 
 }  
  
int  KikuchiApprox::FindOneLevelKikuchiApproximation(clique** ListCliques, int ncliques)  
{  
    int auxinters,s;
       auxinters=0; 
       level = 0;  
       
         auxinters = CreateAllListCliquesInter(ListCliques,ncliques); 
           
	 if (auxinters==0) return 0; 	      
         s = -1;
         FindCR(ListCliques,ncliques,0);
         SimplifyAllCliques(0,s); 
	 return 1; 
 }  


void KikuchiApprox::CheckPresentVars(KikuchiClique* Cliq)  
{  
    int i,j;  
   
    if (numberneighbors==0)  
    {  
	for(i=0;i<Cliq->NumberVars;i++)   
         {  
	     if (Cliq->vars[i]!=currentvar) 
	      { 
	        count_overl[Cliq->vars[i]]=  count_overl[Cliq->vars[i]]+Cliq->count;                  neighbors[numberneighbors++] = Cliq->vars[i]; 
              }  
         }           
    }  
    else  
    {  
      for(i=0;i<Cliq->NumberVars;i++)  
      {  
	  j = 0;  
          if (Cliq->vars[i]!=currentvar) 
          { 
            while (j<numberneighbors && neighbors[j] != Cliq->vars[i]) j++;  
            if(j==numberneighbors) 
            { 
             count_overl[Cliq->vars[i]] = 1; 
             neighbors[numberneighbors++] = Cliq->vars[i];           
            } 
           else  count_overl[neighbors[j]]= count_overl[neighbors[j]]+ Cliq->count;           
	  }  
      }  
    }  
 
} 
  
  
void KikuchiApprox::CheckPresentVars(clique* Cliq)  
{  
    int i,j;  
   
    if (numberneighbors==0)  
    {  
	for(i=0;i<Cliq->NumberVars;i++)   
         {  
          if (Cliq->vars[i]!=currentvar) 
	    { 
	       count_overl[Cliq->vars[i]] = 1;  
               neighbors[numberneighbors] = Cliq->vars[i]; 
	       numberneighbors++;
            } 
         }           
    }  
    else  
    {  
      for(i=0;i<Cliq->NumberVars;i++)  
      {  
	  j = 0;  
          if(Cliq->vars[i]!=currentvar) 
          { 
           while (j<numberneighbors && neighbors[j]!=Cliq->vars[i]) j++;  
           if(j==numberneighbors)  
            { 
             count_overl[Cliq->vars[i]] = 1; 
             neighbors[numberneighbors++] = Cliq->vars[i];  
	    } 
           else
              {
               count_overl[neighbors[j]]++;    
              }
          }          
      }  
    }    
     
}  
  
 
int KikuchiApprox::CreateFirstListCliquesInter(clique** ListCliques, int ncliques)   
{   
 KikuchiClique* SolapCliq;   
 int i,j; 

 for(i=0;i<ncliques;i++)  
   { 
     CheckPresentVars(ListCliques[i]);  
     //cout <<"Cliq"<<i<<endl;
     //for(int a=0;a<MaxVars;a++) cout<<count_overl[a]<<" ";
     //cout<<endl;
     for(j=i+1;j<ncliques;j++)  
     {   
         //cout<<" j "<<j<<endl;
         SolapCliq = FindIntersection(ListCliques[i], ListCliques[j]);    
         if (SolapCliq != (KikuchiClique*)0)  Insert(0,SolapCliq);  
         //if (SolapCliq != (KikuchiClique*)0) SolapCliq->print(); 
     } 
   } 
 
   return (FirstKCliqLevels[0]!=(memberlistKikuchiClique*)0);  
}   
 
  
int KikuchiApprox::CreateOtherListCliquesInter(int level)   
{   
 KikuchiClique* SolapCliq;   
  memberlistKikuchiClique  
 *firstpointer, *secondpointer;   
   
  
   firstpointer =  FirstKCliqLevels[level-1];   
 
    while (firstpointer!=(memberlistKikuchiClique*)0)   
    {       
         secondpointer = firstpointer->nextcliq;  
        
        
          while (secondpointer!=(memberlistKikuchiClique*)0)   
          { 
            SolapCliq = FindIntersection(firstpointer->KCliq,secondpointer->KCliq);   
	    //if (SolapCliq != (KikuchiClique*)0) SolapCliq->print(); 
            if (SolapCliq != (KikuchiClique*)0) Insert(level,SolapCliq);               
            secondpointer = secondpointer->nextcliq;     
          }
	 firstpointer = firstpointer->nextcliq;   
    }    
   return (FirstKCliqLevels[level]!=(memberlistKikuchiClique*)0);  
  }   
 

int KikuchiApprox::CreateAllListCliquesInter(clique** ListCliques, int ncliques)   
{   
// All the overlapping are kept in level 0
// Levels 0 and 1 are used as auxiliary

KikuchiClique* SolapCliq;   
 memberlistKikuchiClique  *firstpointer, *secondpointer;   
 int i,j; 

// First the overlapping from the main cliques are found

 for(i=0;i<ncliques;i++)  
   { 
     CheckPresentVars(ListCliques[i]);  
    for(j=i+1;j<ncliques;j++)  
     {   
         SolapCliq = FindIntersection(ListCliques[i], ListCliques[j]);    
         if (SolapCliq != (KikuchiClique*)0)  
              {
		  // If there is not an exact copy of the clique add it to the other too  
                 if (Insert(0,SolapCliq) !=-1) Insert(1,SolapCliq);   
              }
     } 
   } 

 if (FirstKCliqLevels[0]==(memberlistKikuchiClique*)0) return 0;
 firstpointer = FirstKCliqLevels[1];

// Level 1 always keeps the last overlapping found for look at the intersections

  while (firstpointer!=(memberlistKikuchiClique*)0)   
   { 
   
     while (firstpointer!=(memberlistKikuchiClique*)0)   
      {       
         secondpointer = firstpointer->nextcliq;   
         while (secondpointer!=(memberlistKikuchiClique*)0)   
          { 
            SolapCliq = FindIntersection(firstpointer->KCliq,secondpointer->KCliq);   
	    if (SolapCliq != (KikuchiClique*)0) 
              {
		  //If there is not an exact copy of the clique add it to the other too  
                if (Insert(0,SolapCliq) !=-1) Insert(2,SolapCliq);   
               }
            secondpointer = secondpointer->nextcliq;     
          }
	 firstpointer = firstpointer->nextcliq;   
      }
     CleanLevel(1);  //Level 1 is emptied of the cliques
     FirstKCliqLevels[1] =  FirstKCliqLevels[2];
     FirstKCliqLevels[2] = (memberlistKikuchiClique*)0; 
     firstpointer =  FirstKCliqLevels[1]; 
    }   
   CleanLevel(1);
   return 1;  
  }   

void KikuchiApprox::CleanLevel(int lev)
{
 memberlistKikuchiClique* auxmCliq;  

 while (FirstKCliqLevels[lev] != (memberlistKikuchiClique*) 0)  
    {  
     auxmCliq=FirstKCliqLevels[lev]->nextcliq;   
     delete FirstKCliqLevels[lev];   
     FirstKCliqLevels[lev] = auxmCliq;  
    }  

}

 
  int KikuchiApprox::Insert(int lev, KikuchiClique* Cliq)   
  {   
      int Found; // Insert either inserts (return 0) the new clique or subsumes it (return 1)   or delete it (-1)
      int Idem; 
     memberlistKikuchiClique *auxKCliq, *previousKCliq, *nextKCliq;   
     // The cliques are ordered in the list from higher size to minimum
                
    if( FirstKCliqLevels[lev]==(memberlistKikuchiClique*)0)    
                {               // First clique to enter
                 auxKCliq  = new memberlistKikuchiClique(Cliq);     
                 FirstKCliqLevels[lev] = auxKCliq;   
                 LastKCliqLevels[lev] = auxKCliq ;  
                 return 0; 
                 } 
    
     if (FirstKCliqLevels[lev]->KCliq->NumberVars<Cliq->NumberVars) 
     {  //The size of the first clique is smaller, the coming will be the first
       auxKCliq  = new memberlistKikuchiClique(Cliq); 
       auxKCliq->nextcliq = FirstKCliqLevels[lev]; 
       FirstKCliqLevels[lev] =auxKCliq; 
       return 0; 
     }       
     else if(FirstKCliqLevels[lev]->KCliq->IdemCliques(Cliq))   //they are equal
     { 
	 nextKCliq = FirstKCliqLevels[lev]; 
	 Idem = 1; 
     }       
     else  
     {         
         previousKCliq = FirstKCliqLevels[lev]; 
         nextKCliq = FirstKCliqLevels[lev]->nextcliq; 
         Idem = 0; 
     } 
 
     Found = Idem;  
     while ( (nextKCliq !=(memberlistKikuchiClique*)0) && ! Found)   
      {            
	  Idem  =  nextKCliq->KCliq->IdemCliques(Cliq);   
          Found = (Idem ||  nextKCliq->KCliq->NumberVars<Cliq->NumberVars); 
          if (!Found)  
           { 
	    previousKCliq = nextKCliq; 
            nextKCliq = nextKCliq->nextcliq;  
           }  
      }   
     if (! Found) // it is inserted according to its size
     { 
        auxKCliq  = new memberlistKikuchiClique(Cliq); 
        previousKCliq->nextcliq = auxKCliq; 
        LastKCliqLevels[lev] = auxKCliq; 
     } 
     else 
     { 
       if (Idem)   
       {  
	   //nextKCliq->KCliq->count++;  // the number of copies is increased
         if (Cliq->father->NumberVars < nextKCliq->KCliq->father->NumberVars)// the father with smallest marginal is chosen  
         nextKCliq->KCliq->father=Cliq->father;  
         delete Cliq; 
         return -1; 
       } 
       else 
       { 
         auxKCliq  = new memberlistKikuchiClique(Cliq); 
         previousKCliq->nextcliq = auxKCliq; 
         auxKCliq->nextcliq = nextKCliq; 
       } 
     }   
    return  1;   
   }   
   
 void   KikuchiApprox::FillListKikuchiCliques(memberlistKikuchiClique** ListKikuchiCliques) 
{  
  int i,j; 
  memberlistKikuchiClique *actualKcliq, *auxKCliq; 
 
     for(i=0;i<=level; i++)  
       {  
	   //cout<<"lev "<<level<<" i "<<i<<endl;
         actualKcliq = FirstKCliqLevels[i]; 
         while (actualKcliq!=(memberlistKikuchiClique*)0)  
          { 
	   for(j=0;j<actualKcliq->KCliq->NumberVars;j++)  
             { 
		 //cout<<"lev "<<i<<" j "<< j<<endl; 
		 //actualKcliq->KCliq->print();
              auxKCliq  = new memberlistKikuchiClique(actualKcliq->KCliq); 
              if(ListKikuchiCliques[actualKcliq->KCliq->vars[j]]==(memberlistKikuchiClique*)0) ListKikuchiCliques[actualKcliq->KCliq->vars[j]] = auxKCliq;
              else ListKikuchiCliques[actualKcliq->KCliq->vars[j]]->nextcliq = auxKCliq; 
             } 
	   actualKcliq=actualKcliq->nextcliq; 
          }      
       }  
 } 
   
void   KikuchiApprox::CreateMarg(unsigned* dim) 
{  
  int i,j; 
  memberlistKikuchiClique *actualKcliq; 
 
     for(i=0;i<=level; i++)  
       {  
	   // cout<<"lev "<<level<<" i "<<i<<endl;
         actualKcliq = FirstKCliqLevels[i]; 
         while (actualKcliq!=(memberlistKikuchiClique*)0)  
          { 
           actualKcliq->KCliq->current = new clique(actualKcliq->KCliq->vars[0],actualKcliq->KCliq->NumberVars);  //Creates marginal for Kik. clique       
	   for(j=1;j<actualKcliq->KCliq->NumberVars;j++)  
               actualKcliq->KCliq->current->Add(actualKcliq->KCliq->vars[j]);
           actualKcliq->KCliq->CreateMarg(dim); //marginal tables are created
           actualKcliq->KCliq->CompMargFromFather();        
	   actualKcliq=actualKcliq->nextcliq; 
          }      
       }  
 } 
   


/*

int  KikuchiApprox::FindKikuchiApproximation(clique** ListCliques,memberlistclique* actualcliq)  
{  
    int auxinters=0; 
       level = 0;  
       
         auxinters = CreateFirstListCliquesInter(ListCliques,actualcliq); 
           
	 if (auxinters==0) return 0; 
         SimplifyAllCliques(0); 
       
         while(!AlreadyFoundApproximation() && level <= MaxLength)  
         {  
           level++;  
           CreateOtherListCliquesInter(level); 
           SimplifyAllCliques(level); 
         
          }  
         return 1; 
     }  


int KikuchiApprox::CreateFirstListCliquesInter(clique** ListCliques,memberlistclique* actualcliq)   
{   
 KikuchiClique* SolapCliq;   
 memberlistclique *firstpointer, *secondpointer;   
  
 
 if((actualcliq->nextcliq !=(memberlistclique*)0) && (actualcliq->nextcliq ==(memberlistclique*)0))  return 0; //There is only one clique thus no list of intersections   
 else   
 {   
   firstpointer = actualcliq;   
   while (firstpointer!=(memberlistclique*)0) 
   { 
     CheckPresentVars(ListCliques[firstpointer->cliq]);  
     firstpointer = firstpointer->nextcliq; 
   } 
  
 
   firstpointer = actualcliq;   
   while (firstpointer!=(memberlistclique*)0)   
    {      
       secondpointer = firstpointer->nextcliq;   
         while (secondpointer!=(memberlistclique*)0)   
         {   
          SolapCliq = FindIntersection(ListCliques[firstpointer->cliq], ListCliques[secondpointer->cliq]);    
          //if (SolapCliq != (KikuchiClique*)0) SolapCliq->print(); 
          if (SolapCliq != (KikuchiClique*)0)  Insert(0,SolapCliq);             
          secondpointer = secondpointer->nextcliq;   
        }   
       firstpointer = firstpointer->nextcliq;   
    }  
 }   
   return (FirstKCliqLevels[0]!=(memberlistKikuchiClique*)0);  
}   

void  KikuchiApprox::SimplifyClique(KikuchiClique* cliq)  
    {    
	int i,minalpha,alpha;  
        i = 0;  
        minalpha = cliq->count+1; 
        
        
        if(cliq->sign==-1) 
        { 
	    while(i<cliq->NumberVars) 
             { 
               alpha = 1+ cliq->count- count_overl[cliq->vars[i]]; 
	       if(alpha< minalpha) minalpha = alpha; 
	       i++; 
	     }  
        } 
        else 
        { 
	    while(i<cliq->NumberVars) 
             { 
               alpha = -1 + cliq->count + count_overl[cliq->vars[i]]; 
	       if(alpha>=0 && alpha< minalpha) minalpha = alpha; 
	       i++; 
	     }  
        }  
        //cout<<" ( initcount "<< cliq->count;
        if(minalpha>0 && minalpha<=cliq->count) cliq->count -= minalpha; 
        else if(minalpha>cliq->count) cliq->count=0; 
        //cout<<"  sign "<< cliq->sign<<" newcount "<<cliq->count<<" minalpha "<<minalpha<<" ) "<<endl;
        if(cliq->count>0) CheckPresentVars(cliq); 
    }  

 
void  KikuchiApprox::SimplifyAllCliques(int lev) 
{ 

   memberlistKikuchiClique  *firstpointer;   
   
   firstpointer =  FirstKCliqLevels[lev]; 
   //for(i=0;i<MaxVars;i++) cout<<count_overl[i]<<" ";     
   //cout<<" Simplifying  "<<endl; 

   while (firstpointer != (memberlistKikuchiClique*)0) 
     { 
       cout<<"Oldcount" <<  firstpointer->KCliq->count<< "  Newcount "; 
       SimplifyClique(firstpointer->KCliq); 
       cout<<firstpointer->KCliq->count<<"Clique :  "; 
       firstpointer->KCliq->print(); 
      
      firstpointer = firstpointer->nextcliq; 
     } 
   //for(i=0;i<MaxVars;i++) cout<<count_overl[i]<<" ";  
   // cout<<" Simplified  "<<endl; 
} 

void KikuchiApprox::CheckPresentVars(KikuchiClique* Cliq)  
{  
    int i,j;  
   
    if (numberneighbors==0)  
    {  
	for(i=0;i<Cliq->NumberVars;i++)   
         {  
	     if (Cliq->vars[i]!=currentvar) 
	      { 
	        count_overl[Cliq->vars[i]]=  count_overl[Cliq->vars[i]]+Cliq->count*Cliq->sign;  
                neighbors[numberneighbors++] = Cliq->vars[i]; 
              }  
         }           
    }  
    else  
    {  
      for(i=0;i<Cliq->NumberVars;i++)  
      {  
	  j = 0;  
          if (Cliq->vars[i]!=currentvar) 
          { 
            while (j<numberneighbors && neighbors[j] != Cliq->vars[i]) j++;  
            if(j==numberneighbors) 
            { 
             count_overl[Cliq->vars[i]] = Cliq->sign; 
             neighbors[numberneighbors++] = Cliq->vars[i];           
            } 
           else  count_overl[neighbors[j]]= count_overl[neighbors[j]]+ Cliq->count*Cliq->sign;                  
	  }  
      }  
    }  
 
} 
 
 
  
  
void KikuchiApprox::CheckPresentVars(clique* Cliq)  
{  
    int i,j;  
   
    if (numberneighbors==0)  
    {  
	for(i=0;i<Cliq->NumberVars;i++)   
         {  
          if (Cliq->vars[i]!=currentvar) 
	    { 
	       count_overl[Cliq->vars[i]] = 1;  
               neighbors[numberneighbors] = Cliq->vars[i]; 
	       numberneighbors++;
            } 
         }           
    }  
    else  
    {  
      for(i=0;i<Cliq->NumberVars;i++)  
      {  
	  j = 0;  
          if(Cliq->vars[i]!=currentvar) 
          { 
           while (j<numberneighbors && neighbors[j]!=Cliq->vars[i]) j++;  
           if(j==numberneighbors)  
            { 
             count_overl[Cliq->vars[i]] = 1; 
             neighbors[numberneighbors++] = Cliq->vars[i];  
	    } 
           else
              {
               count_overl[neighbors[j]]++;    
              }
          }          
      }  
    }  
   
  
int KikuchiApprox::Subsume(KikuchiClique* Cliq)   
  {   
     int Found; 
     memberlistKikuchiClique* auxKCliq;   
        
     Found = 0;  
     auxKCliq = FirstKCliqLevels[level]; 
   
     while ( (auxKCliq !=(memberlistKikuchiClique*)0) && ! Found)   
      {   
	  Found =  auxKCliq->KCliq->IdemCliques(Cliq);   
        if (!Found) auxKCliq = auxKCliq->nextcliq;   
      }   
  
     if (Found)   
     {  
	 auxKCliq->KCliq->count++;  
       if (Cliq->father->NumberVars < auxKCliq->KCliq->father->NumberVars)// the father with smallest marginal is chosen  
       auxKCliq->KCliq->father=Cliq->father;  
      }   
    return  Found;   
   
  }   
int KikuchiApprox::CreateOtherListCliquesInter(int level)   
{   
 KikuchiClique* SolapCliq;   
  memberlistKikuchiClique  
 *firstpointer, *secondpointer;   
   
   firstpointer =  FirstKCliqLevels[level-1];   
 
    while (firstpointer!=(memberlistKikuchiClique*)0)   
    {       
       if(firstpointer->KCliq->count > 0) 
       {	 
        if(firstpointer->KCliq->count > 0)  
         { 
           SolapCliq = FindSelfIntersection(firstpointer->KCliq); 
           if (SolapCliq != (KikuchiClique*)0) Insert(level,SolapCliq); 
         } 
	 
         secondpointer = firstpointer->nextcliq;   
         while (secondpointer!=(memberlistKikuchiClique*)0)   
          { 
           if(secondpointer->KCliq->count > 0) 
            { 
	     SolapCliq = FindIntersection(firstpointer->KCliq,secondpointer->KCliq);   
	     //if (SolapCliq != (KikuchiClique*)0) SolapCliq->print(); 
             if (SolapCliq != (KikuchiClique*)0) Insert(level,SolapCliq); 
             } 	      
            secondpointer = secondpointer->nextcliq;     
          }       
		 
	  }  
       firstpointer = firstpointer->nextcliq;   
    }    
   
   return (FirstKCliqLevels[level]!=(memberlistKikuchiClique*)0);  
  }   
 
 


*/
  
  
