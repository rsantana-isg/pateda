#include "FDA.h"   
#include "auxfunc.h"   
    
   
using namespace std;

DynFDA::~DynFDA()   
{ int i;  

   
    delete[] AllProb;  
   
    delete[] order;  

    for (i=0;i<length;i++) delete[] Matrix[i];   

    delete[] Matrix;   
 
    if (MChiSquare != (double*)0) delete[] MChiSquare;   

     delete[] auxvectGen;  
 
    delete[] legalconfGen; 
 
    if (CliqWeights != (double*)0)  delete[] CliqWeights;  

    delete[] OrderCliques;    
 
   }   

  
DynFDA::DynFDA(int vars,int CliqMaxLength,int MaxNumCliq, double depthresh, int priors, int LearnT, int Cycles):AbstractProbModel(vars)   
{   
 int i;  
 cycles = Cycles; 
 LearningType = LearnT; 
 CliqMLength= CliqMaxLength;  
 MaxNCliq = MaxNumCliq;  
 threshold = depthresh;  
 NeededOvelapp = 0;
 OrderCliques = new int[MaxNCliq]; 
 CliqWeights = new double[MaxNCliq]; 
 AllProb = new double[length];  
 Matrix = new unsigned* [length];   
 auxvectGen = new int[length]; //Maximo tamanho espacio clique   
 legalconfGen = new int[65536];  
  MChiSquare = new double [length*(length-1)/2];   
  order  = new int[length];    
  Priors = priors;  
  //CliqWeights = (double*)0;  
  for (i=0;i<length;i++) Matrix[i]= new unsigned[length];   
  CondKikuchiApprox = (KikuchiApprox*)0; 
  ListKikuchiCliques = (memberlistKikuchiClique**)0 ; 
  ListOvelap = (clique**)0;

 
}   
   
DynFDA::DynFDA(int vars, Popul* pop,int CliqMaxLength,int MaxNumCliq, double depthresh, int priors, int LearnT, int Cycles):AbstractProbModel(vars)   
{   
 int i;  
 
 cycles = Cycles; 
LearningType = LearnT; 
 CliqMLength= CliqMaxLength;  
 MaxNCliq = MaxNumCliq;   
 OrderCliques = new int[MaxNCliq]; 
 CliqWeights = new double[MaxNCliq];
 threshold = depthresh; 
 NeededOvelapp = 0;
 AllProb = new double[length];  
 Matrix = new unsigned* [length]; 
 auxvectGen = new int[length]; 
 legalconfGen = new int[65536];  
  order  = new int[length];   
  Priors = priors;  
 for (i=0;i<length;i++) Matrix[i] = new unsigned[length];  
  CondKikuchiApprox = (KikuchiApprox*)0; 
  ListKikuchiCliques = (memberlistKikuchiClique**)0 ; 
  ListOvelap = (clique**)0;
 }   
   
DynFDA::DynFDA(int vars,Popul* pop,int CliqMaxLength,int MaxNumCliq,  int* order1, double depthresh, int priors, int LearnT, int Cycles):AbstractProbModel(vars)   
{   
    int i;  
 
 cycles = Cycles; 
  LearningType = LearnT; 
  CliqMLength= CliqMaxLength;   
  MaxNCliq = MaxNumCliq;   
 OrderCliques = new int[MaxNCliq]; 
 CliqWeights = new double[MaxNCliq];
  threshold = depthresh; 
 NeededOvelapp = 0;
 AllProb = new double[length]; 
  Matrix = new unsigned* [length];
  auxvectGen = new int[length]; 
 legalconfGen = new int[65536];  
  order  = new int[length];   
  Priors = priors;  
  for (i=0;i<length;i++) Matrix[i] = new unsigned[length];   
  
  //CliqWeights = (double*)0;  
  CondKikuchiApprox = (KikuchiApprox*)0; 
  ListKikuchiCliques = (memberlistKikuchiClique**)0;
  ListOvelap = (clique**)0;
 }   
 
void DynFDA::SetNPoints(int OrigNPoints,int NPoints, double* pvect)  
{  
     genepoollimit = OrigNPoints;  
     actualpoolsize= NPoints;  
     PopProb = pvect;  
    }  
  
void DynFDA::SetOrderofCliques()  
{  
int i;  
//OrderCliques = new int[SetOfCliques->NumberCliques]; 
 for(i=0; i<SetOfCliques->NumberCliques; i++)  OrderCliques[i] = i;  
 //else   for(i=0; i<SetOfCliques->NumberCliques; i++)  OrderCliques[i] = SetOfCliques->NumberCliques-i-1;  
 // int auxint = randomint(SetOfCliques->NumberCliques);
 //  OrderCliques[0] =   OrderCliques[auxint];
 //  OrderCliques[auxint] = 0;
   //RandomPerm(SetOfCliques->NumberCliques-1,1,OrderCliques);   
}  
  
double DynFDA::CliqWeight(clique* cliq)  
{  
    int i,j,aux;  
    double weight;  
    weight = 0; aux = 0;  
  
    for(i=0;i<cliq->NumberVars-1;i++)  
      for(j=i+1;j<cliq->NumberVars;j++)  
      {  
	  if(cliq->vars[i]<cliq->vars[j])  
               aux = cliq->vars[i]*(2*length-cliq->vars[i]+1)/2 +cliq->vars[j]-2*cliq->vars[i]-1;  
          else aux = cliq->vars[j]*(2*length-cliq->vars[j]+1)/2 +cliq->vars[i]-2*cliq->vars[j]-1;  
       
	weight += MChiSquare[aux];  
       }  
    return weight;  
  
}  
  
void DynFDA::OrderCliquesSizes(int maxway )  
{  
int i,j,aux;  
 SetOrderofCliques();  
 if (maxway)  
 {  
 for(i=0; i<SetOfCliques->NumberCliques-1; i++)   
   for(j=i+1; j<SetOfCliques->NumberCliques; j++)  
    {  
        if (SetOfCliques->CliquesSizes[OrderCliques[j]]>SetOfCliques->CliquesSizes[OrderCliques[i]])  
	{  
	    aux = OrderCliques[j];  
            OrderCliques[j] = OrderCliques[i];  
            OrderCliques[i] = aux;   
        }  
    }  
 }  
 else  
   {  
 for(i=0; i<SetOfCliques->NumberCliques-1; i++)   
   for(j=i+1; j<SetOfCliques->NumberCliques; j++)  
    {  
        if (SetOfCliques->CliquesSizes[OrderCliques[j]]<SetOfCliques->CliquesSizes[OrderCliques[i]])  
	{  
	    aux = OrderCliques[j];  
            OrderCliques[j] = OrderCliques[i];  
            OrderCliques[i] = aux;   
        }  
    }  
 }  
}  
void DynFDA::FindCliquesWeights()  
{  
 int i;  
 //CliqWeights = new double[SetOfCliques->NumberCliques];  
 for(i=0; i<SetOfCliques->NumberCliques; i++) CliqWeights[i] = CliqWeight(SetOfCliques->ListCliques[i]);  
}  

void DynFDA::OrderCliquesForProteins()
{
 SetOrderofCliques();  
}  
  
void DynFDA::OrderCliquesWeights(int maxway )  
{  
int i,j,aux;  
  SetOrderofCliques();  
  FindCliquesWeights(); 
 if (maxway)  
 {  
 for(i=0; i<SetOfCliques->NumberCliques-1; i++)   
   for(j=i+1; j<SetOfCliques->NumberCliques; j++)  
    {  
        if (CliqWeights[OrderCliques[j]]>CliqWeights[OrderCliques[i]])  
	{  
	    aux = OrderCliques[j];  
            OrderCliques[j] = OrderCliques[i];  
            OrderCliques[i] = aux;   
        }  
    }  
 }  
 else  
   {  
 for(i=0; i<SetOfCliques->NumberCliques-1; i++)   
   for(j=i+1; j<SetOfCliques->NumberCliques; j++)  
    {  
        if (CliqWeights[OrderCliques[i]]>CliqWeights[OrderCliques[j]])  
	{  
	    aux = OrderCliques[j];  
            OrderCliques[j] = OrderCliques[i];  
            OrderCliques[i] = aux;   
        }  
    }  
 }  
 
 /* 
for(i=0; i<SetOfCliques->NumberCliques; i++)   
 {  
     cout<<"Cliq "<<i<<" weight "<<CliqWeights[OrderCliques[i]]<< "   ";  
     SetOfCliques->ListCliques[OrderCliques[i]]->print();  
 }  
    
 */
}   


void DynFDA::AncestralOrderingFactOnlyCliq(int maxway )  
{
//Finds those cliques that form a valid factorization
  
 int i,j,k,l,m,auxsolap,maxsolap,maxindex;  
 int *auxlistcliques,*listvars; 
 clique* newcliq; 
 int *auxoverl;

 listvars = new int[length]; 
 auxlistcliques = new int[SetOfCliques->NumberCliques];
   
 OrderCliquesWeights(maxway);  //Cliques are ordered according their weights
 //OrderCliquesForProteins(); // MODIFICATION DONE FOR PROTEINS
 for(i=0; i<length; i++) listvars[i] = 0;  
 NeededCliques = 1;
 NeededOvelapp = 0;
 ListOvelap = new clique*[length];
 auxoverl = new int[length];

 auxlistcliques[0] = OrderCliques[0];  
 for(i=0; i<SetOfCliques->ListCliques[OrderCliques[0]]->NumberVars; i++)  
     listvars[SetOfCliques->ListCliques[OrderCliques[0]]->vars[i]] = 1; 
 OrderCliques[0] =-1; 

 //cout<<"NeedC "<<NeededCliques<<" index "<<0<<" C : ";
 //SetOfCliques->ListCliques[auxlistcliques[0]]->print();
  
for(i=1; i<SetOfCliques->NumberCliques; i++) //Maximum overlapping is found  
  {   
    maxsolap = -1;  
    maxindex = -1;  
    for(j=1; j<SetOfCliques->NumberCliques; j++)  
     {  
      if( OrderCliques[j] != -1)  
       {   
        auxsolap = 0;
	//for(l=0; l<20;l++) cout<<listvars[l]<<" ";
        //cout<<endl;
        for(l=0; l<SetOfCliques->ListCliques[OrderCliques[j]]->NumberVars;l++) 
          if (listvars[SetOfCliques->ListCliques[OrderCliques[j]]->vars[l]]==1)   auxsolap++;  
	//cout<<" i "<<i<<" j "<<j<<" auxsolap "<<auxsolap<<" maxsolap "<<maxsolap<<endl;
        if(auxsolap>maxsolap && auxsolap<SetOfCliques->ListCliques[OrderCliques[j]]->NumberVars)
	{   
         for(k=0; k<NeededCliques; k++)  
         {  
	     //     cout<<" i "<<i<<" j "<<j<<" k "<<k<<" auxsolap "<<auxsolap<<" maxsolap "<<maxsolap<<endl;
          auxsolap = 0;
          for(l=0; l<SetOfCliques->ListCliques[auxlistcliques[k]]->NumberVars;l++) 
	  {
	      if (SetOfCliques->ListCliques[OrderCliques[j]]->VarIsInClique(SetOfCliques->ListCliques[auxlistcliques[k]]->vars[l])==1) auxsolap++;               
          }    
       if (auxsolap>maxsolap)  
	   {  
	    maxsolap = auxsolap;  
            maxindex = j;   
            //cout<<"maxindex "<<maxindex<<"maxsolap "<<maxsolap<<endl;          
           } 
	 } //for k
	} 
       }  
      } 
    //  cout<<"maxindex is "<<maxindex<<endl;     
    if(maxindex != -1) //maxindex keeps the index of the overlapped clique
     {
      m = 0;
     if (maxsolap>0) 
       {
        for(l=0; l<SetOfCliques->ListCliques[OrderCliques[maxindex]]->NumberVars;l++)
        {
           if(listvars[SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l]]==1)
           auxoverl[m++] = SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l];
           listvars[SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l]] = 1;   
        }
        newcliq = new clique(auxoverl[0],maxsolap);  //Creates ovelapped clique 
        for(m=1; m<maxsolap;m++)  newcliq->Add(auxoverl[m]);   
        ListOvelap[NeededOvelapp++] = newcliq;
        //newcliq->print();
       }
       else  
       {
        for(l=0; l<SetOfCliques->ListCliques[OrderCliques[maxindex]]->NumberVars;l++)
        {         
         listvars[SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l]] = 1;   
        }
       }  
        auxlistcliques[NeededCliques++] = OrderCliques[maxindex];  
        //cout<<"NeedC "<<NeededCliques<<" maxsolap "<<maxsolap<<" index "<<maxindex<<" C : ";
        //SetOfCliques->ListCliques[OrderCliques[maxindex]]->print();
        OrderCliques[maxindex] = -1;     
     } 
     }
   j = NeededCliques;
 
 for(i=0; i<SetOfCliques->NumberCliques; i++) 
        if(OrderCliques[i] != -1)  auxlistcliques[j++] = OrderCliques[i];  

 // cout<<" Are Needed "<<NeededCliques<<"  of "<<SetOfCliques->NumberCliques<<endl;  

for(i=0; i<SetOfCliques->NumberCliques; i++) 
 {
  OrderCliques[i] = auxlistcliques[i];  
  //SetOfCliques->ListCliques[OrderCliques[i]]->print(); 
 }

delete[] auxoverl;
delete[] auxlistcliques;  
delete[] listvars;  
}


void DynFDA::AncestralOrderingFact(int maxway )  
{  
int i,j,l,m,auxsolap,maxsolap,maxindex;  
int *auxlistcliques,*listvars;  
 listvars = new int[length]; 
 auxlistcliques = new int[SetOfCliques->NumberCliques];
 clique* newcliq; 
 int *auxoverl;

 OrderCliquesWeights(maxway);  
 for(i=0; i<length; i++) listvars[i] = 0;  
 NeededCliques = 1; 
 NeededOvelapp = 0;
 ListOvelap = new clique*[length];
 auxoverl = new int[length];

     auxlistcliques[0] = OrderCliques[0];  
     for(i=0; i<SetOfCliques->ListCliques[OrderCliques[0]]->NumberVars; i++)  
     listvars[SetOfCliques->ListCliques[OrderCliques[0]]->vars[i]] = 1; 
     OrderCliques[0] =-1; 
     //cout<<"NeedC "<<NeededCliques<<" index "<<0<<" C : ";
     //SetOfCliques->ListCliques[auxlistcliques[0]]->print();
  
for(i=1; i<SetOfCliques->NumberCliques; i++)   
  {   
    maxsolap = -1;  
    maxindex = -1;  
    //while (maxindex<SetOfCliques->NumberCliques &&  OrderCliques[maxindex]==-1) maxindex++;  
    for(j=1; j<SetOfCliques->NumberCliques; j++)  
     {  
      if( OrderCliques[j] != -1)  
       {  
        auxsolap = 0;
        for(l=0; l<SetOfCliques->ListCliques[OrderCliques[j]]->NumberVars;l++) 
	    if (listvars[SetOfCliques->ListCliques[OrderCliques[j]]->vars[l]]==1) auxsolap++;
        if(auxsolap>maxsolap &&  auxsolap<SetOfCliques->ListCliques[OrderCliques[j]]->NumberVars)  
	   {  
	    maxsolap = auxsolap;  
            maxindex = j;             
           }          
       }  
     }  
     
    if(maxindex != -1)
       {  
       m = 0;
       if (maxsolap>0) 
       {
          for(l=0; l<SetOfCliques->ListCliques[OrderCliques[maxindex]]->NumberVars;l++)
        {
           if(listvars[SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l]]==1)
           auxoverl[m++] = SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l];
           listvars[SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l]] = 1;   
        }
        newcliq = new clique(auxoverl[0],maxsolap);  //Creates ovelapped clique 
        for(m=1; m<maxsolap;m++)  newcliq->Add(auxoverl[m]);   
        ListOvelap[NeededOvelapp++] = newcliq;
        //newcliq->print();
       }
       else  
      {
        for(l=0; l<SetOfCliques->ListCliques[OrderCliques[maxindex]]->NumberVars;l++)
        {         
         listvars[SetOfCliques->ListCliques[OrderCliques[maxindex]]->vars[l]] = 1;   
        }
      }  
        auxlistcliques[NeededCliques++] = OrderCliques[maxindex];         
      	//cout<<"NeedC "<<NeededCliques<<" maxsolap "<<maxsolap<<" index "<<maxindex<<" C : ";
        //SetOfCliques->ListCliques[OrderCliques[maxindex]]->print();
        OrderCliques[maxindex] = -1;     
       }
    }
  
  j = NeededCliques;
 for(i=0; i<SetOfCliques->NumberCliques; i++) 
        if(OrderCliques[i] != -1)  auxlistcliques[j++] = OrderCliques[i];  


 //cout<<" Are Needed "<<NeededCliques<<"  of "<<SetOfCliques->NumberCliques<<endl;  

for(i=0; i<SetOfCliques->NumberCliques; i++) 
 {
  OrderCliques[i] = auxlistcliques[i];  
  //cout<<"i= "<<i<<" order "<<OrderCliques[i]<<endl;
 }
delete[] auxoverl;
delete[] auxlistcliques;  
delete[] listvars;  
}  


double DynFDA::ProbKikuchi(unsigned* vector) 
{
    double prob,valprob;
    int i,l;
    memberlistKikuchiClique *actualKcliq;

    prob = 1.0;
  
     for(i=0;i<SetOfCliques->NumberCliques; i++) 
     {
      prob *= SetOfCliques->ListCliques[OrderCliques[i]]->Prob(vector);
      //cout<<"i "<<i<<" p "<<prob<<endl;
     }   

     for(i=0;i<=CondKikuchiApprox->level; i++)  
       {  
         actualKcliq = CondKikuchiApprox->FirstKCliqLevels[i]; 
         while (actualKcliq!=(memberlistKikuchiClique*)0)  
          { 
	      //actualKcliq->KCliq->current->print();
           valprob = actualKcliq->KCliq->current->Prob(vector); 
           for(l=0;l<abs(actualKcliq->KCliq->count);l++) 
            { 
	     if (actualKcliq->KCliq->sign==1) 
	           prob  *= valprob;    
             else if(valprob>0)
                   prob  /= valprob; 
             else 
		 cout<<"Something happened here"<<endl;
            } 
           //cout<<"i "<<i<<" p "<<prob<<endl;
	   actualKcliq=actualKcliq->nextcliq; 
          }  
       }
 return prob;
}
  





double DynFDA::Prob(unsigned* vector) 
{
    double prob;
    int i;
    prob = 1.0;
  
 if (LearningType==1  ||  LearningType==3 || LearningType == 5) 
  {
  for(i=0;i<NeededCliques; i++) 
     prob *= SetOfCliques->ListCliques[OrderCliques[i]]->Prob(vector);
  for (i=0;i<NeededOvelapp; i++)
    prob /=  ListOvelap[i]->Prob(vector);  
  }
 else prob = ProbKikuchi(vector);
 return prob;
}
  

double DynFDA::Prob(unsigned* vector, int pri) 
{
    return  Prob(vector); 
}
  
void DynFDA::SimpleOrdering()   
 {  
 int i;  
 for (i=0;i<length;i++) order[i] = i;  
 }  

void DynFDA::CreateGraphCliques()   
{  
  SetOfCliques = new maximalsubgraph(length,Matrix,CliqMLength,MaxNCliq,order);   
  SetOfCliques->CreateGraphCliques();  
}  


void DynFDA::CreateGraphCliquesProtein(int typemodel, int sizecliqp)   
{  
  SetOfCliques = new maximalsubgraph(length,(unsigned int**)0,CliqMLength,MaxNCliq,order);   
  SetOfCliques->CreateGraphCliquesProtein(typemodel,sizecliqp);  
}  


void DynFDA::CreateGraphCliquesProteinMPM(int typemodel, int ncliques, unsigned long** listclusters) // MPM, used for AffEDA     
{  
  SetOfCliques = new maximalsubgraph(length,(unsigned int**)0,CliqMLength+1,MaxNCliq,order);
  SetOfCliques->CreateGraphCliquesProteinMPM(typemodel,ncliques,listclusters);
  
}  

void DynFDA::DestroyKikuchi()   
{  
    int i;
   memberlistKikuchiClique* auxmemberKikCliq; 
   if (CondKikuchiApprox != (KikuchiApprox*)0) delete CondKikuchiApprox; 
  
    if (ListKikuchiCliques != (memberlistKikuchiClique**)0)  
     { 
      for(i=0;i<length;i++)    
	{  
	    while(ListKikuchiCliques[i] != (memberlistKikuchiClique*)0) 
            { 
	      auxmemberKikCliq = ListKikuchiCliques[i]; 
              ListKikuchiCliques[i] = ListKikuchiCliques[i]->nextcliq;                 
	      delete  auxmemberKikCliq;   
            }  		   
	}   
      delete[] ListKikuchiCliques;  
     } 
    
}  
  

void DynFDA::Destroy()   
{  
   if (LearningType==2) DestroyKikuchi(); 
   else if (LearningType==1  || LearningType==3 || LearningType==5) DestroyFDAOverl();
   else if (LearningType==0  || LearningType==4)
    {
      DestroyFDAOverl();
      DestroyKikuchi(); 
    } 
 
   delete SetOfCliques;
 
 }

 
void DynFDA::DestroyFDAOverl()   
{  
   int i;
   
   if (ListOvelap != (clique**)0) 
    {
      for(i=0;i< NeededOvelapp;i++)    delete ListOvelap[i];   
     
      delete[] ListOvelap;
   
    }     
}

void DynFDA::SetEdgesPerm(clique* cliq)  
{   
    int i,j;  
    for(i=0; i<cliq->NumberVars;i++)   
     for(j=i+1; j<cliq->NumberVars;j++)  
     {  
	 Matrix[cliq->vars[i]][cliq->vars[j]] = 2;   
         Matrix[cliq->vars[j]][cliq->vars[i]] = 2;  
     }   
     
}  
/*   
void DynFDA::DepSecOrder(int order) // it is assumed order=2  
{   
  //This procedure receives a list of cliques after zero and one order  
  //dep tests  
    cliques** newlist;  
    newlist = new clique*[SetOfCliques->MaxNumberOfCliquesInTheGraph];  
    int i, newlistncliqs;  
    newlistncliqs=0;  
  
    OrderCliques = new int[SetOfCliques->NumberCliques];  
    OrderCliquesSizes(0);  
    for(i=0;i<SetOfCliques->NumberCliques; i++ )   
    {  
      if (SetOfCliques->CliquesSizes[OrderCliques[i]]<=order)  
      {  
      if (SetOfCliques->CliquesSizes[OrderCliques[i]]>=2)  
	  SetEdgesPerm(SetOfCliques->ListCliques[OrderCliques[i]]);  
      newlistncliqs  
      }   
    }	   
      
    while(i<)  
     
  
 delete[] OrderCliques;  
 delete[] newlist  
}  
  
void DynFDA::DepSecOrder(int order, double valchi) // it is assumed order=2  
{   
  //This procedure receives a list of cliques after zero and one order  
  //dep tests  
    int i,j,k,l,aux;  
    int auxvars[4], *indexpairs;   
    double dtest;  
  
     indexpairs = new int[actualpoolsize];    
      Pot[0]=8; Pot[1]=4; Pot[2]=2; Pot[3]=1;  
  
          for(j=0; j<length-1; j++)  
	  {     
	  	for(k=j+1; k<length; k++)   
		{		              
		  aux = j*(2*length-j+1)/2 +k-2*j-1;   
		  Matrix[j][k] = (MChiSquare[aux] > 0);  
                  Matrix[k][j] =  Matrix[j][k];  
		  // MChiSquare[aux] = 0;  
                }  
          }   
	  
     for(i=0; i<length-1; i++)   // these are the conditionant vars  
       for(j=i+1; j<length; j++)  
       {  
	   if(Matrix[i][j]==1)  
	    {  
		for(k=0; k<length-1; k++)  // for these the test is done  
                for(l=k+1; l<length; l++)  
                 {  
                   aux = k*(2*length-k+1)/2 +l-2*k-1;   
                   //cout<<" k "<<k<<" l "<<l<<"i "<<i<<" j "<<j<<endl;  
                   if( !(k==i || k==j || l==i ||  l==j) && Matrix[k][l]==1 && Matrix[i][l]>0 && Matrix[k][j]>0 && Matrix[i][j]>0 && Matrix[k][l]>0  )   
		   {  
		       auxvars[0] = k; auxvars[1] = l;  
                       auxvars[2] = i; auxvars[3] = j;  
                        
                       dtest = DepTest(4,16,&auxvars[0],indexpairs);  
                       //cout<<" k "<<k<<" l "<<l<<" i "<<i<<" j "<<j<<" dep "<<dtest<<endl;  
                       if (dtest<valchi)  
                        {  
			    Matrix[k][l] = 0; //now independent  
                            Matrix[l][k] = 0;   
                            Matrix[i][j] = 2; // now permanent  
                            Matrix[j][i] = 2;    
                            MChiSquare[aux] = 0;                         
                        }  
                       //else   MChiSquare[aux] = dtest;   
                     
                   }   
                    
                 }  
  
            }  
          }    
     //cout<<"Last matrix"<<endl;   
     //PrintMatrix(MChiSquare);  
  
     delete[] indexpairs;  
     
}  
  
*/  
  
void DynFDA::DepSecOrder(int order, double valchi) // it is assumed order=2  
{   
  //This procedure receives a list of cliques after zero and one order  
  //dep tests  
    int i,j,k,l,aux,a;  
    int auxvars[4], *indexpairs, *neighbor;   
    double dtest;  
  
      indexpairs = new int[actualpoolsize];    
      neighbor = new int[length];    
  
      Pot[0]=8; Pot[1]=4; Pot[2]=2; Pot[3]=1;  
  
          for(j=0; j<length-1; j++)  
	  {     
	  	for(k=j+1; k<length; k++)   
		{		              
		  aux = j*(2*length-j+1)/2 +k-2*j-1;   
		  Matrix[j][k] = (MChiSquare[aux] > 0);  
                  Matrix[k][j] =  Matrix[j][k];  
		  // MChiSquare[aux] = 0;  
                }  
          }   
	  
     for(i=0; i<length-1; i++)   // these are the conditionant vars  
       for(j=i+1; j<length; j++)  
        {  
  
          aux = i*(2*length-j+1)/2 +j-2*i-1;   
	   if(Matrix[i][j]==1)  
	    {  
                a=0;  
		for(k=0; k<length; k++)   
		    if( k!=i && k!=j && (Matrix[i][k]>0 || Matrix[j][k]>0)) neighbor[a++] = k;  
                if(a>1)  
		{     
                 for(k=0; k<a-1; k++)   
                  for(l=k+1; l<a; l++)  
                  {  
                    
                   //cout<<" k "<<k<<" l "<<l<<"i "<<i<<" j "<<j<<endl;  
                   if( Matrix[i][j]==1 &&  ((Matrix[i][neighbor[k]]>0 && Matrix[j][neighbor[l]]>0) || (Matrix[i][neighbor[l]]>0 && Matrix[j][neighbor[k]]>0))  )   
		   {  
                       auxvars[0] = i; auxvars[1] = j;  
		       auxvars[2] = neighbor[k]; auxvars[3] = neighbor[l];                                                            dtest = DepTest(4,16,&auxvars[0],indexpairs);  
                       //cout<<" k "<<k<<" l "<<l<<" i "<<i<<" j "<<j<<" dep "<<dtest<<endl;  
                       if (dtest<valchi)  
                        {  
			    Matrix[i][j] = 0; // now permanent  
                            Matrix[j][i] = 0;    
                            MChiSquare[aux] = 0;                         
                        }  
                     
                   }   
                    
                 }  
  
                }  
              }    
    //cout<<"Last matrix"<<endl;   
     //PrintMatrix(MChiSquare);  
     }  
     delete[] indexpairs;  
     delete[] neighbor;  
}  
  
  
double DynFDA::DepTest(int nvars, int dimension, int* depvars, int* indexpairs)   
 {   
   int i,k,j,auxind, remaining_configurations;  
   unsigned current_pair[10]; //maximum number of dep test  
   double OrvBivProb,ExpBivProb,sqval;  
    
	for(i=0; i<actualpoolsize; i++) indexpairs[i] = i;   
        remaining_configurations = actualpoolsize;   
 	sqval  = 0;  
	for(i=0;i<dimension;i++) freqxyz[i] = 0;  
  
        while(remaining_configurations > 0)   
          {   
	   for(i=0; i<nvars; i++) current_pair[i] = Pop->P[indexpairs[0]][depvars[i]];  
           i = 0;   
           OrvBivProb=0;  
	   while(i< remaining_configurations)   
	    {    
	       j=0;  
               while (j<nvars && (current_pair[j] == Pop->P[indexpairs[i]][depvars[j]])) j++;  
               if(j==nvars)  
               {  
                 OrvBivProb += PopProb[indexpairs[i]];   
		 indexpairs[i] = indexpairs[remaining_configurations-1];   
		 remaining_configurations--;   
	       }			    
	       else i++;   
	    } 			    
	   auxind = 0;	  
           for(j=0;j<nvars;j++)  auxind += Pot[j]*current_pair[j];  
           freqxyz[auxind] += OrvBivProb;  
	 }  
	  
	 for(i=0;i<dimension;i++)  
          {  
	       
            freqxyz[i] =  (freqxyz[i]*genepoollimit+1.0)/(genepoollimit+dimension);   
            if (i<(dimension/4)) freqz[i]=0;  
            if (i<(dimension/2))  
	    {  
             freqxz[i] = 0;  
             freqyz[i] = 0;  
            }  
          
          }  
	       
	 for(i=0;i<(dimension/2);i++) //Marginal freq are calculated  
		{   
                  freqyz[i] += (freqxyz[i]+ freqxyz[i+dimension/2]);  
                   
                  if (i<(dimension/4))  
		      freqxz[i] +=  (freqxyz[i]+ freqxyz[i+dimension/4]);  
                  else  
                      freqxz[i] +=  (freqxyz[i+dimension/4]+ freqxyz[i+dimension/2]);  
                  if (i<(dimension/4))  
                  {  
                   freqz[i] += (freqyz[i]+ freqxyz[i+dimension/4]+freqxyz[i+3*dimension/4]);   
                   
                  }  
                }  
          
	 for(i=0;i<=Pot[0];i+=(Pot[0])) //value sqval of chiquad is found  
         for(j=0;j<=Pot[1];j+=(Pot[1]))  
          for(k=1;k<=dimension/4;k++)  
           {  
                 OrvBivProb = freqxyz[i+j+k-1];   
                 ExpBivProb =  (freqxz[i/2+k-1]* freqyz[j+k-1])/freqz[k-1];  
                 //sqval += ((OrvBivProb-ExpBivProb)*(OrvBivProb-ExpBivProb))/ExpBivProb;      
		 sqval += (OrvBivProb)*log(ExpBivProb/OrvBivProb);           
           }  
	  
         //return sqval*genepoollimit;     
         return -2*sqval*genepoollimit;                   
			  
  }  
  
   
  
void DynFDA::FindChiSquareMat(double valchi)   
 {   
  // The Chi square Matrix is constructed, it contains how significan is the relationship  
   
   int i,k,j;   
   int aux;   
   unsigned current_pair[2];  
   double bivfreq[4];   
   double univfreq1,univfreq2,x;  
   int *indexpairs;   
   int remaining_configurations;   
   double OrvBivProb,ExpBivProb ;  //Observed and expected biv prob. for chi square  
      
	indexpairs = new int[actualpoolsize];          
  
 	 for(j=0; j<length-1; j++)         // For all possible pairs of variables   
	  	for(k=j+1 ; k<length; k++)   
		{   
                 aux = j*(2*length-j+1)/2 +k-2*j-1;   
		 x=0;   
		 for(i=0;i<4;i++) bivfreq[i] = 0; //introduce priors  
      	    
	     for(i=0; i<actualpoolsize; i++) indexpairs[i] = i;   
	     remaining_configurations = actualpoolsize;   
 	    
        while(remaining_configurations > 0)   
          {   
			    OrvBivProb = 0;                  
				current_pair[0] = Pop->P[indexpairs[0]][j];   
				current_pair[1] = Pop->P[indexpairs[0]][k];   
   
			    i = 0;   
			    while(i< remaining_configurations)   
				{    
                 if( current_pair[0] == Pop->P[indexpairs[i]][j] &&    
				   current_pair[1] == Pop->P[indexpairs[i]][k] )    
				 {   
                                  OrvBivProb += PopProb[indexpairs[i]];   
				  indexpairs[i] = indexpairs[remaining_configurations-1];   
				  remaining_configurations--;   
				 }   
			     else i++;   
				} 			    
			    bivfreq[2*current_pair[0]+current_pair[1]] += OrvBivProb;  
	   }  
	for(i=0;i<4;i++)   
          {  
            bivfreq[i] = (bivfreq[i]*genepoollimit+1.0)/(genepoollimit+4.0); //normalize prob with priors   
	    //   cout<<bivfreq[i]<<" ";  
            }  
	  
	                    univfreq1 = bivfreq[2]+bivfreq[3];  
                            univfreq2 = bivfreq[1]+bivfreq[3];  
			      
/*                          if(univfreq1!=1 && univfreq2!=1)  
                           {  
                            OrvBivProb = bivfreq[0];   
                            ExpBivProb = (1-univfreq1)* (1-univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb; // ((O - E)2/E)       
                            }  
  
                            //cout<<j<<k<<"  Real "<<OrvBivProb<<" Expected "<<ExpBivProb<<" Chi "<<x<<endl;  
  
                          if(univfreq1!=1 && univfreq2!=0)  
                           {  
                            OrvBivProb = bivfreq[1];   
                            ExpBivProb = (1-univfreq1)* (univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb;  
                           }  
			      
                       if(univfreq1!=0 && univfreq2!=1)  
                           {  
                            OrvBivProb = bivfreq[2];   
                            ExpBivProb = (univfreq1)* (1-univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb;   
                           }   
                      if(univfreq1!=0 && univfreq2!=0)  
                           {      
                            OrvBivProb = bivfreq[3];   
                            ExpBivProb = (univfreq1)* (univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb;   
                           }  
  
 			    x*= genepoollimit;  
*/  
			      
              if(univfreq1!=1 && univfreq2!=1)  
               {  
                            OrvBivProb = bivfreq[0];   
                            ExpBivProb = (1-univfreq1)* (1-univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
               }             	      
              if(univfreq1!=1 && univfreq2!=0)  
               {       
  
                            OrvBivProb = bivfreq[1];   
                            ExpBivProb = (1-univfreq1)* (univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
               }             	      
              if(univfreq1!=0 && univfreq2!=1)  
               {   
                            OrvBivProb = bivfreq[2];   
                            ExpBivProb = (univfreq1)* (1-univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
              }             	      
              if(univfreq1!=0 && univfreq2!=0)  
               {   
                            OrvBivProb = bivfreq[3];   
                            ExpBivProb = (univfreq1)* (univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
               }  
                            x*= -2*genepoollimit; //Important to determine if is actualpoolsize  
                   
			    //    cout<<"j "<<j<<" k "<< k<<" Chi "<<x<<" valchi "<< valchi<<endl;      
		     if(x>valchi)  
		    {     
                     MChiSquare[aux] = x;   
                     Matrix[j][k] = 1;  
                     Matrix[k][j] = 1;  
                    }  
		    
		     //  Matrix[j][k] = 1;  
		     //MChiSquare[aux] = x;      
		}  
	 //cout<<"Previous matrix"<<endl;   
	  
	 //PrintMatrix(MChiSquare);  
         //PrintSimMatrix();  
        delete[] indexpairs;   
  }	           			   
    

  
void DynFDA::FindChiSquareMatBiv(double valchi)   
 {   
  // The Chi square Matrix is constructed, it contains how significan is the relationship  
   
int i,k,j,z,aux, remaining_configurations;  
   unsigned current_pair[3];  
   double freqxyz[8],freqxz[4],freqyz[4],freqz;  
   int *indexpairs;   
   double OrvBivProb,ExpBivProb,sqval,x ;  //Observed and expected biv prob. for chi square  
   
	indexpairs = new int[actualpoolsize];          
	 for(j=0; j<length-1; j++)         // For all possible pairs of variables   
	  	for(k=j+1 ; k<length; k++)   
		{		              
                  aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  MChiSquare[aux] = 0;  
                }  
  
       for(z=0; z<length; z++)   
	 for(j=0; j<length-1; j++)         // For all possible pairs of variables   
	  	for(k=j+1 ; k<length; k++)   
		{   
                    
                  aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  sqval=0;   
                 
		 if (z!=j && z!=k && Matrix[j][k]==1)  
		 {                      
		    for(i=0;i<8;i++) freqxyz[i] = 0.0; //1.0/(genepoollimit+8); //introduce priors  
      	    
	            for(i=0; i<actualpoolsize; i++) indexpairs[i] = i;   
	            remaining_configurations = actualpoolsize;   
 	    
                    while(remaining_configurations > 0)   
                      {   
			        OrvBivProb = 0;                  
				current_pair[0] = Pop->P[indexpairs[0]][j];   
				current_pair[1] = Pop->P[indexpairs[0]][k];  
                                current_pair[2] = Pop->P[indexpairs[0]][z];   
   
			    i = 0;   
			    while(i< remaining_configurations)   
				{    
                                if( current_pair[0] == Pop->P[indexpairs[i]][j] &&    
				   current_pair[1] == Pop->P[indexpairs[i]][k]  
                                  && current_pair[2] == Pop->P[indexpairs[i]][z] )    
				 {   
                                  OrvBivProb += PopProb[indexpairs[i]];   
				  indexpairs[i] = indexpairs[remaining_configurations-1];   
				  remaining_configurations--;   
				 }   
			        else i++;   
				} 			    
			    freqxyz[4*current_pair[0]+2*current_pair[1]+current_pair[2]] += OrvBivProb;  
	                 }  
		    
		           for(i=0;i<8;i++)   
                    {  
			freqxyz[i] = (freqxyz[i]*genepoollimit+1.0)/(genepoollimit+8.0); //normalize prob with priors  
			//cout<< freqxyz[i]<<" ";  
			}    
		      
		  //cout<<endl;  
	          freqz = freqxyz[1] + freqxyz[3] + freqxyz[5] + freqxyz[7]; // z=1  
                  freqxz[0] = freqxyz[0] + freqxyz[2]; freqxz[1] = freqxyz[1] + freqxyz[3];  
                  freqxz[2] = freqxyz[4] + freqxyz[6]; freqxz[3] = freqxyz[5] + freqxyz[7];  
                  freqyz[0] = freqxyz[0] + freqxyz[4]; freqyz[1] = freqxyz[1] + freqxyz[5];  
                  freqyz[2] = freqxyz[2] + freqxyz[6]; freqyz[3] = freqxyz[3] + freqxyz[7];  
/*		  	   
		 if(freqz !=1)  
		 {  
                 OrvBivProb = freqxyz[0];   
                 ExpBivProb =  (freqxz[0]* freqyz[0])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb)*(OrvBivProb-ExpBivProb))/ExpBivProb;   
                 }  
                 if(freqz !=0)  
		 {  
                 OrvBivProb = freqxyz[1];   
                 ExpBivProb =  (freqxz[1]* freqyz[1])/(freqz);  
                 sqval += ((OrvBivProb-ExpBivProb)*(OrvBivProb-ExpBivProb))/ExpBivProb;   
                 }  
                 if(freqz !=1)  
		 {  
                 OrvBivProb = freqxyz[2];   
                 ExpBivProb =  (freqxz[0]* freqyz[2])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 }  
                 if(freqz !=0)  
		 {  
                 OrvBivProb = freqxyz[3];   
		 ExpBivProb =  (freqxz[1]* freqyz[3])/(freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
		 }  
                 if(freqz !=1)  
                 {  
                 OrvBivProb = freqxyz[4];   
                 ExpBivProb =  (freqxz[2]* freqyz[0])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;  
                 }  
                 if(freqz !=0)  
                 {  
                 OrvBivProb = freqxyz[5];   
                 ExpBivProb =  (freqxz[3]* freqyz[1])/(freqz);   
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                  }  
                 if(freqz !=1)  
                 {  
                 OrvBivProb = freqxyz[6];   
                 ExpBivProb =  (freqxz[2]* freqyz[2])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 }  
                 if(freqz !=0)  
                 {  
                 OrvBivProb = freqxyz[7];   
                 ExpBivProb =  (freqxz[3]* freqyz[3])/(freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 }  
                 x= sqval*genepoollimit;   
*/		   
		 if(freqz !=1)  
		 {   
                 OrvBivProb = freqxyz[0];   
                 ExpBivProb =  (freqxz[0]* freqyz[0])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 }  
                 if(freqz !=0)  
		 {   
                 OrvBivProb = freqxyz[1];   
                 ExpBivProb =  (freqxz[1]* freqyz[1])/(freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 }  
                 if(freqz !=1)  
		 {   
                 OrvBivProb = freqxyz[2];   
                 ExpBivProb =  (freqxz[0]* freqyz[2])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 }  
                 if(freqz !=0)  
		 {   
                 OrvBivProb = freqxyz[3];   
                 ExpBivProb =  (freqxz[1]* freqyz[3])/(freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 }  
                 if(freqz !=1)  
		 {   
                 OrvBivProb = freqxyz[4];   
                 ExpBivProb =  (freqxz[2]* freqyz[0])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 }  
                 if(freqz !=0)  
		 {   
                 OrvBivProb = freqxyz[5];   
                 ExpBivProb =  (freqxz[3]* freqyz[1])/(freqz);   
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 }  
                 if(freqz !=1)  
		 {   
                 OrvBivProb = freqxyz[6];   
                 ExpBivProb =  (freqxz[2]* freqyz[2])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);    
                 }  
                 if(freqz !=0)  
		 {   
                 OrvBivProb = freqxyz[7];   
                 ExpBivProb =  (freqxz[3]* freqyz[3])/(freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);   
                 }              
		 x = -2*sqval*genepoollimit; //Important to determine if is actualpoolsize  
		 		   
		   
		 if (x<valchi)  
                  {  
                    MChiSquare[aux] = 0;  
                    Matrix[j][k] = 0;  
                    Matrix[k][j] = 0;  
                    //cout<<"Indep   j "<<j<<" k "<<k <<" z "<<z<<" x "<<x<<endl;   
                  }  
                  else  
		   if(MChiSquare[aux]==0 || x<MChiSquare[aux])  
                  {    
		      //cout<<" j "<<j<<" k "<<k <<" z "<<z<<" x "<<x<<endl;    
                     MChiSquare[aux] = x;     
                  }  
		   
		 }       
                                          
		}   
                    
       //cout<<"Second matrix"<<actualpoolsize<<" "<<genepoollimit<<endl;   
       //PrintMatrix(MChiSquare);  
	   //PrintSimMatrix();   
         
		delete[] indexpairs;   
  }	           			   
    
	  
void DynFDA::kmeans(int iters, double distclasses, int *nclasses, double *depmean, double *indepmean)   
 {   
     int i,j,k,countindep,countdep,aux;  
     double sumindep,sumdep;  
       
     i=0;  
     *depmean = 1;  
     *indepmean = 0;  
     *nclasses =2;  
     while (i<iters &&  *nclasses==2)  
     {  
     countdep =0; countindep =0;  
     sumdep = 0; sumindep = 0;  
        for(j=0; j<length-1; j++)         // For all possible pairs of variables   
	    for(k=j+1 ; k<length; k++)   
		{   
		if(Matrix[j][k]>0)  
                {  
                 aux = j*(2*length-j+1)/2 +k-2*j-1;   
		 if (fabs(MChiSquare[aux]-*depmean) <=  
fabs(MChiSquare[aux]-*indepmean))  
                 {  
		     sumdep += fabs(MChiSquare[aux]);  
                     countdep++;  
   
                 }  
                 else  
                 {  
		     sumindep += fabs(MChiSquare[aux]);  
                     countindep++;  
                 }  
                }  
                }  
         
	if (countdep>0)  
	{  
	    *depmean = sumdep/countdep;  
        }              
        else *nclasses=1;  
          
        if (countindep>0)  
	{  
	    *indepmean = sumindep/countindep;   
        }else *nclasses=1;  
          
      if(*nclasses==1 && i==0)  
        {  
	    *depmean = (sumdep+sumindep+1)/(countdep+countindep);   
            *indepmean = (sumdep+sumindep-1)/(countdep+countindep);  
            *nclasses=2;  
        }      
        
      cout<<"iter "<<i<<" nclasses "<<*nclasses<<" depmean "<<*depmean<< "  indepmean "<<*indepmean<<" countindep "<<countindep<<" countdep "<<countdep<<endl;  
        i++;  
     }  
     
   if(*depmean<=(*indepmean)+1) *nclasses = 1;  
 }  
  
void DynFDA::SetMatrix(int nclasses, double depmean, double indmean)  
{  
    int j,k,aux,count;  
    double thr;  
    count = 0;  
    thr = indmean + 6.0*(depmean-indmean)/10.0;  
  
    //cout<<"nclasee "<<nclasses<<" depmean "<<depmean<<" indmean "<<indmean<<endl;  
  
        for(j=0; j<length; j++)         // For all possible pairs of variables   
	    for(k=0; k<length; k++)   
		{   
		    if(k==j) Matrix[j][k]=1;  
                    else if(j<k)  
		    {  
                      aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  
		      if(nclasses==1 || MChiSquare[aux]==0)  
		      {  
		       Matrix[j][k] = 0;  
                       Matrix[k][j] = 0;  
                       MChiSquare[aux] = 0;  
                      }  
                      else     
                      {  
			  /*      MChiSquare[aux]=(MChiSquare[aux]>thr)*MChiSquare[aux];  
		      Matrix[j][k] =(MChiSquare[aux]>thr);  
		      Matrix[k][j] =  Matrix[j][k];   
			  */  
		      Matrix[j][k] =(fabs(MChiSquare[aux]-depmean) <fabs(MChiSquare[aux]-indmean));  
		      Matrix[k][j] =  Matrix[j][k];  
		      MChiSquare[aux] *= ((fabs(MChiSquare[aux]-depmean) <fabs(MChiSquare[aux]-indmean)));  
                  count += Matrix[j][k];  
                    }  
		   }  
                }  
	//cout<<"Number of dep "<<count<<endl;     
}  
  
  


  
 void DynFDA::CorrectChiSquareMat()   
 {   
    
   int k,j;   
   int aux,numedges;   
    
      
	 for(j=0; j<length; j++)    
	 {    
             numedges = 0;         
	  	for(k=0 ; k<length; k++)   
		{   
		 if(j!=k)  
		 {  
                   if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1;   
                   else   aux = k*(2*length-k+1)/2 +j-2*k-1;  
                   numedges += (MChiSquare[aux]>threshold);  
                 }  
                }   
		  
                  
		 if (numedges>=CliqMLength)   
                  {  
		      //    cout<<"This case "<<"j="<<j<<" numedges="<<numedges<<" CliqMLength "<<CliqMLength<<endl;   
                      TruncChiSquareMatrix(j,numedges);  
                  }  
                		   
	}  
 }  
  




void DynFDA::CorrectChiSquareMatGTest(unsigned int* Cardinalities, double cthreshold)   
 {   
    int which,df,j,k,aux,numedges;  
    double  threshchival;  
    which = 2; //Calculate the chi for a given prob  
    df = Cardinalities[0]*Cardinalities[1]-1;
    threshchival = (FindChiVal(cthreshold,which,df))/(2*genepoollimit); // The G statistics is G = 2*N*MI(j,k) and follows chi-square                   
       
         
	 for(j=0; j<length; j++)    
	 {    
             numedges = 0;         
	  	for(k=0 ; k<length; k++)   
		{                  
         	 if(j!=k)  
		 {                                
                   if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1;   
                   else   aux = k*(2*length-k+1)/2 +j-2*k-1;  
                   numedges += (MChiSquare[aux]>threshchival);  
                   //cout<<j<<" "<<k<<" "<<df<<" "<<threshchival<<" "<<MChiSquare[aux]<<endl;
                 }  
                }   
		  
                  
		 if (numedges>=CliqMLength)   
                  {  
		      //    cout<<"This case "<<"j="<<j<<" numedges="<<numedges<<" CliqMLength "<<CliqMLength<<endl;   
                      TruncChiSquareMatrix(j,numedges);  
                  }  
                		   
	}  
 }  




  
void DynFDA::TruncChiSquareMatrix(int row, int numedges)  
{  
  int k,j,aux1,aux2;   
    int* auxvect;  
    int aux,s;  
  
  auxvect = new int[length];  
  s = 0;  
  
  for(j=0; j<length; j++)  if(j!=row) auxvect[s++] = j;  
  RandomPerm(length-1,length-1,auxvect);   
		  
	 for(j=0; j<numedges; j++)          
	   for(k=j+1 ; k<length-1; k++)   
		{   
                   if (row<auxvect[j]) aux1 = row*(2*length-row+1)/2 +auxvect[j]-2*row-1;   
                   else   aux1 = auxvect[j]*(2*length-auxvect[j]+1)/2 +row-2*auxvect[j]-1;  
                   if (row<auxvect[k]) aux2 = row*(2*length-row+1)/2 +auxvect[k]-2*row-1;   
                   else   aux2 = auxvect[k]*(2*length-auxvect[k]+1)/2 +row-2*auxvect[k]-1;   
		   
                  if (MChiSquare[aux1]<MChiSquare[aux2])  
		  {  
                   aux = auxvect[j];  
		   auxvect[j] = auxvect[k];   
                   auxvect[k] = aux;                      
                  }  
		}  
                  
                 for(j=CliqMLength-1; j<numedges; j++)  
		 {  
                  
                   if (row<auxvect[j]) aux1 = row*(2*length-row+1)/2 +auxvect[j]-2*row-1;   
                   else   aux1 = auxvect[j]*(2*length-auxvect[j]+1)/2 +row-2*auxvect[j]-1;  
		  MChiSquare[aux1] = 0;  
                 }  
  		 delete[] auxvect;  
}  
  


void DynFDA::FindNeighbors(long unsigned** listclusters, int nneighbors,double thresh)  
{  
  int k,j,aux1,aux2;   
    int* auxvect;  
    int aux,s,row;  
    double val_mut_inf, tot_mut_inf;

  auxvect = new int[length];  
  s = 0;
  tot_mut_inf = 0;  
  
  	 for(j=0; j<length-1; j++)          
	   for(k=j+1 ; k<length; k++)   
		{   
                   aux = j*(2*length-j+1)/2 +k-2*j-1; 
                   if(MChiSquare[aux]>0) tot_mut_inf += MChiSquare[aux];
                   s++;
		   //cout<<j<<" "<<k<<" "<<MChiSquare[aux]<<" "<<tot_mut_inf<<endl;
                   
		}

	 // val_mut_inf = thresh * 2 * (tot_mut_inf/(length*(length-1)));
        val_mut_inf = thresh *  (tot_mut_inf/s);

	//cout<<val_mut_inf <<" "<<tot_mut_inf<<endl;

	 for (row=0;row<length;row++)        
	   {       
             s = 0;  
             for(j=0; j<length; j++) 
	       {
		 if(j<row)   aux = j*(2*length-j+1)/2 +row-2*j-1;
                 else if (j>row) aux = row*(2*length-row+1)/2 +j-2*row-1;
                 if(j!=row && MChiSquare[aux]>val_mut_inf) auxvect[s++] = j; 
               }

          if (s>nneighbors)
	   {  
            
             RandomPerm(s,s ,auxvect);   // A random permutation of the row to avoid ordering bias
	     for(j=0; j<s; j++)          
	      for(k=j+1 ; k<s; k++)   
		{   
		 
                   if (row<auxvect[j]) aux1 = row*(2*length-row+1)/2 +auxvect[j]-2*row-1;   
                   else   aux1 = auxvect[j]*(2*length-auxvect[j]+1)/2 +row-2*auxvect[j]-1;  
                   if (row<auxvect[k]) aux2 = row*(2*length-row+1)/2 +auxvect[k]-2*row-1;   
                   else   aux2 = auxvect[k]*(2*length-auxvect[k]+1)/2 +row-2*auxvect[k]-1;   
		   
		   //cout<<"row "<<row<<" "<<s<<" "<<" j "<<j<<" "<<aux1<<" "<<aux2<<endl; 
                  if (MChiSquare[aux1]<MChiSquare[aux2])  
		  {  
                   aux = auxvect[j];  
		   auxvect[j] = auxvect[k];   
                   auxvect[k] = aux;                      
                  }  
		}  
	  
                
	         listclusters[length-row-1] = new unsigned long[nneighbors+2];
	         listclusters[length-row-1][0] = nneighbors+1;
                 listclusters[length-row-1][1] = row;
                 for(j=2; j<nneighbors+2; j++)   listclusters[length-row-1][j] = auxvect[j-2];
	   }
	  else
	    {
              listclusters[length-row-1] = new unsigned long[s+2];
              listclusters[length-row-1][0] = s+1;
              listclusters[length-row-1][1] = row;
              for(j=2; j<s+2; j++)   listclusters[length-row-1][j] = auxvect[j-2];
            }
	   }
         
  delete[] auxvect;  
}  

 




void DynFDA::FindStrongNeighbors(unsigned int* Cardinalities, double cthreshold,long unsigned** listclusters, int nneighbors,double thresh)  
{  
  int k,j,aux1,aux2;   
    int* auxvect;  
    int aux,s,row;  
    double val_mut_inf, tot_mut_inf;
    int which,df,numedges;  
    double  threshchival;  
   
  auxvect = new int[length];  
  s = 0;
  which = 2; //Calculate the chi for a given prob  
  df = Cardinalities[0]*Cardinalities[1]-1;
  threshchival = (FindChiVal(cthreshold,which,df))/(2*genepoollimit); // The G statistics is G = 2*N*MI(j,k) and follows chi-square                   
       
         
	 for(j=0; j<length; j++)    
	 {    
             numedges = 0;         
	  	for(k=0 ; k<length; k++)   
		{                  
         	 if(j!=k)  
		 {                                
                   if (j<k) aux = j*(2*length-j+1)/2 +k-2*j-1;   
                   else   aux = k*(2*length-k+1)/2 +j-2*k-1;  
                   if(MChiSquare[aux]<threshchival) MChiSquare[aux]=0;  
                   //cout<<j<<" "<<k<<" "<<df<<" "<<threshchival<<" "<<MChiSquare[aux]<<endl;
                 }  
                }   		   
	}  

	 val_mut_inf = 0;
	 for (row=0;row<length;row++)        
	   {       
             s = 0;  
             for(j=0; j<length; j++) 
	       {
		 if(j<row)   aux = j*(2*length-j+1)/2 +row-2*j-1;
                 else if (j>row) aux = row*(2*length-row+1)/2 +j-2*row-1;
                 if(j!=row && MChiSquare[aux]>val_mut_inf) auxvect[s++] = j; 
               }

          if (s>nneighbors)
	   {  
            
             RandomPerm(s,s ,auxvect);   // A random permutation of the row to avoid ordering bias
	     for(j=0; j<s; j++)          
	      for(k=j+1 ; k<s; k++)   
		{   
		 
                   if (row<auxvect[j]) aux1 = row*(2*length-row+1)/2 +auxvect[j]-2*row-1;   
                   else   aux1 = auxvect[j]*(2*length-auxvect[j]+1)/2 +row-2*auxvect[j]-1;  
                   if (row<auxvect[k]) aux2 = row*(2*length-row+1)/2 +auxvect[k]-2*row-1;   
                   else   aux2 = auxvect[k]*(2*length-auxvect[k]+1)/2 +row-2*auxvect[k]-1;   
		   
		   //cout<<"row "<<row<<" "<<s<<" "<<" j "<<j<<" "<<aux1<<" "<<aux2<<endl; 
                  if (MChiSquare[aux1]<MChiSquare[aux2])  
		  {  
                   aux = auxvect[j];  
		   auxvect[j] = auxvect[k];   
                   auxvect[k] = aux;                      
                  }  
		}  
	  
                
	         listclusters[length-row-1] = new unsigned long[nneighbors+2];
	         listclusters[length-row-1][0] = nneighbors+1;
                 listclusters[length-row-1][1] = row;
                 for(j=2; j<nneighbors+2; j++)   listclusters[length-row-1][j] = auxvect[j-2];
	   }
	  else
	    {
              listclusters[length-row-1] = new unsigned long[s+2];
              listclusters[length-row-1][0] = s+1;
              listclusters[length-row-1][1] = row;
              for(j=2; j<s+2; j++)   listclusters[length-row-1][j] = auxvect[j-2];
            }
	   }
         
  delete[] auxvect;  
}  



    
void DynFDA::SetNPoints(int NPoints, double* pvect)  
{  
     actualpoolsize = NPoints;  
     PopProb = pvect;  
}  
   


void DynFDA::LearnMatrix(int** lattice)  
{ 
    int i,j,k;

  for(j=0; j<length; j++)         // For all possible pairs of variables   
      for(k=0; k<length; k++)   Matrix[j][k] = 0;

    for (i=0; i<length; i++) 
    { 
	Matrix[i][i] = 1;
	for (j=1; j<lattice[i][0]+1; j++) Matrix[i][lattice[i][j]] = 1;  
    }
}

void DynFDA::LearnMatrixSAT(int** lattice)  
{ 
    int j,k;

  for(j=0; j<length; j++)         // For all possible pairs of variables   
      for(k=0; k<length; k++) 
         Matrix[j][k] = lattice[j][k];

  for(j=0; j<length; j++)  Matrix[j][j] = 1;
}



/*
void DynFDA::LearnMatrix(int** lattice)  
{ 
    int i,j,k;

  for(j=0; j<length; j++)         // For all possible pairs of variables   
   (  for(k=0; k<length; k++)   Matrix[j][k] = 0;

    for (i=0; i<length; i++) 
    { 
	Matrix[i][i] = 1;
	for (j=1; j<lattice[i][0]; j++) 
	{  
           Matrix[i][lattice[i][j]] = 1;
           for (k=j+1; k<lattice[i][0]+1; k++) 
                Matrix[lattice[i][j]][lattice[i][k]] = 1;  
        }
    }
}
*/


void DynFDA::LearnMatrixFromOther(int** OtherMatrix)  
{  
  int j,k;          
      	 for(j=0; j<length; j++)         // For all possible pairs of variables              
           {  
              Matrix[j][j] = 1;  
	  	for(k=j+1; k<length; k++)   
		{   
                
                  Matrix[j][k] = OtherMatrix[j][k];
     	          Matrix[k][j] = OtherMatrix[k][j];
                }  
         }  
  
	  
}  


void DynFDA::LearnMatrixGTest(unsigned int* Cardinalities, double cthreshold)  
{  
  int j,k,aux,df,which;  
    double threshchival;
    which  = 2;
    df = Cardinalities[0]*Cardinalities[1]-1;
    threshchival = (FindChiVal(cthreshold,which,df))/(2*genepoollimit); // The G statistics is G = 2*N*MI(j,k) and follows chi-square                   
  
     CorrectChiSquareMatGTest(Cardinalities,cthreshold);
      
	 for(j=0; j<length; j++)         // For all possible pairs of variables              
           {  
              Matrix[j][j] = 1;  
	  	for(k=j+1; k<length; k++)   
		{   
                  aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  Matrix[j][k] = (MChiSquare[aux]>threshchival);  
                  Matrix[k][j] = Matrix[j][k];  
                }  
         }  
  
	  
}  


void DynFDA::LearnMatrix()  
{  
    int which,df,j,k,aux;  
    double  threshchival;  
    //int  nclasses;
    //double depmean,indmean;
    which = 2; //Calculate the chi for a given prob  
    df = 1;  
    threshchival = FindChiVal(threshold,which,df);  
    FindChiSquareMat(threshchival);  
    df = 2;  
    threshchival = FindChiVal(threshold,which,df);  
    FindChiSquareMatBiv(threshchival);  
    
    //PrintSimMatrix();   
    //df = 4;  
    //threshchival = FindChiVal(threshold,which,df);  
    //DepSecOrder(2,threshchival);  

    //PrintSimMatrix();  
    //for(j=0; j<length; j++)  Matrix[j][j] = 1;  
    //SetMatrix(nclasses,depmean,indmean);  
    CorrectChiSquareMat();   
      
      
	 for(j=0; j<length; j++)         // For all possible pairs of variables              
           {  
              Matrix[j][j] = 1;  
	  	for(k=j+1; k<length; k++)   
		{   
                  aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  Matrix[j][k] = (MChiSquare[aux]>threshold);  
                  Matrix[k][j] = Matrix[j][k];  
                }  
         }  
  
	  
}  
  
void DynFDA::FindUnivProb()
{ 
    int i,j; 
	// Univariate probabilities are calculated 
  for(j=0; j<length; j++) AllProb[j]=0;       

    for(j=0; j<length; j++)
    {
	 for(i=0; i<actualpoolsize; i++) 
	 { 
  
		if (Pop->P[i][j]==1) AllProb[j] += PopProb[i]; 
	 }   
        AllProb[j] = (AllProb[j]*genepoollimit+1.0)/(genepoollimit+2);   
    }
} 

void DynFDA::UpdateModelSAT(int** lattice)   
{   
      int which,df,j,k,aux;  
    double  threshchival;  
    
    which = 2; //Calculate the chi for a given prob  
   
    LearnMatrixSAT(lattice);
    
    //df = 2;  
    //threshchival = FindChiVal(threshold,which,df);  
    //FindChiSquareMatBiv(threshchival);  
    
    // PrintSimMatrix();   
    //df = 4;  
    //threshchival = FindChiVal(threshold,which,df);  
    //DepSecOrder(2,threshchival);  

    /*
    CorrectChiSquareMat();   
      
      
	 for(j=0; j<length; j++)         // For all possible pairs of variables              
           {  
              Matrix[j][j] = 1;  
	  	for(k=j+1; k<length; k++)   
		{   
                  aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  Matrix[j][k] = (MChiSquare[aux]>threshold);  
                  Matrix[k][j] = Matrix[j][k];  
                }  
         }  
    */
    FindUnivProb();   
    //PrintSimMatrix();  
    CreateGraphCliques();   
    CallProb();     

}


void DynFDA::UpdateModel(int** lattice)   
{   
    LearnMatrix(lattice);
     FindUnivProb();   
    //PrintSimMatrix();  
    CreateGraphCliques();   
    CallProb();     
} 


void DynFDA::UpdateModelProtein(int typemodel, int sizecliq)   
{   
  
    CreateGraphCliquesProtein(typemodel,sizecliq);   
    CallProb();     
} 


void DynFDA::UpdateModelProteinMPM(int typemodel, int ncliques, unsigned long** listclusters) // MPM, used for AffEDA  
{   
    CreateGraphCliquesProteinMPM(typemodel,ncliques,listclusters);   
    CallProb();    
} 


void DynFDA::InitTree(int CNumberPoints, double* pvect, Popul* pop, int NumberPoints)
{
 SetNPoints(NumberPoints,CNumberPoints,pvect); 
 SetPop(pop); 
 UpdateModel();
 // if (LearningType==1  ||  LearningType==3) Priors = 0; // To guarantee exact learning during the mixture phase
}


void DynFDA::UpdateModel(double *vector,int howmany,Popul* EPop) 
{   
    //cout<<"howmany is"<<howmany<<endl;
    //for (int i=0;i<howmany;i++) cout<< vector[i]<<" ";
    //cout<<endl;
  Destroy();
  SetNPoints(genepoollimit,howmany,vector); 
  SetPop(EPop); 
  UpdateModel();
} 


void DynFDA::UpdateModel()   
{   
    LearnMatrix();
    FindUnivProb();  
    //cout<<"FInal Mat"<<endl;  
    //PrintSimMatrix();  
    //FindRandomSimMatrix();   
    CreateGraphCliques();   
    CallProb();     
}   


void DynFDA::UpdateModelGTest(unsigned int* Cardinalities,double threshchival)   
{   
    LearnMatrixGTest(Cardinalities,threshchival);
    FindUnivProb();  
    //cout<<"FInal Mat"<<endl;  
    //PrintSimMatrix();      
    CreateGraphCliques();   
    FGDACallProb();
}   
  

void DynFDA::UpdateModelFromMatrix(int** OtherMatrix)   
{   
    LearnMatrixFromOther(OtherMatrix);
    //cout<<"FInal Mat"<<endl;  
    //PrintSimMatrix();      
    //FindUnivProb();  
    CreateGraphCliques();   
    //FGDACallProb();
}   



  
void DynFDA::PrintSimMatrix()  
{  
 int i,j;   
       
	for (i=0;i<length;i++)  
	{   
	  for (j=0;j<length;j++)    
	  {  
	    cout<<Matrix[i][j]<<" ";  
          }  
          cout<<endl;  
         }   
   cout<<endl;  
}  
  
void DynFDA::FindRandomSimMatrix()  
{  
 int i,j;   
       
	for (i=0;i<length;i++)  
	{   
	  for (j=0;j<length;j++)    
	  {  
           if(j==i+1)   
              {  
		  Matrix[i][j] = 1;  
               Matrix[j][i]  = Matrix[i][j];  
              }  
              else if(i==j) Matrix[i][j]=1;  
	      /*  if(i<j)   
              {  
               Matrix[i][j] =  myrand()>0.5;//  
               Matrix[j][i]  = Matrix[i][j];  
              }  
              else if(i==j) Matrix[i][j]=1;  
	      */  
	      //cout<<Matrix[i][j]<<" ";  
          }  
          
         }  
}  
  
void DynFDA::printmarg()  
{ int i;  
   for (i=0;i<SetOfCliques->NumberCliques; i++)   
	{   
          cout<<"Cliq "<<i+1<<"- ";  
          SetOfCliques->ListCliques[i]->printmarg();  
	 }  
}  
 
void DynFDA::CallProb()   
{   
  //cout<<" LEARNING TYPE "<<LearningType<<endl;

    if (LearningType==0 ||  LearningType==2 ||  LearningType==4) MarkovCallProb(); 
    else if (LearningType==1  ||  LearningType==3 || LearningType==5 || LearningType==6  || LearningType==7  || LearningType==8 || LearningType==9) FDACallProb();	
     
}   
   
void DynFDA::CreateMarg(int ncliques) 
{ 
 int i;   
  for(i=0;i<ncliques; i++ )  SetOfCliques->ListCliques[OrderCliques[i]]->CreateMarg(Pop->dim);   

 if (LearningType != 2 ) 
  {
      for (i=0;i<NeededOvelapp; i++)   ListOvelap[i]->CreateMarg(Pop->dim);  
  }
   
} 
 
void DynFDA::ComputeMarg(int ncliques) 
{   //Debe incluirse orden 
    int i,l; 
   
   if(actualpoolsize>0)   
	 for(l=0; l<actualpoolsize; l++)   
	   for (i=0;i<ncliques; i++)   SetOfCliques->ListCliques[OrderCliques[i]]->Compute(Pop->P[l],PopProb[l]);  


  if (LearningType != 2 ) 
  {
   for(l=0; l<actualpoolsize; l++)   
	   for (i=0;i<NeededOvelapp; i++)  ListOvelap[i]->Compute(Pop->P[l],PopProb[l]);  
  }

} 
 
void DynFDA::SetPriors()
{
    Priors = 1;
}

void DynFDA::Normalize(int ncliques) 
{  
    int i; 
   
    //cout<<"genepoollimit "<<genepoollimit<<" "<<actualpoolsize<<endl;
 if (Priors) 
 {

   if (LearningType != 6 && LearningType != 7) 
   for (i=0;i<ncliques; i++)   
     {      
         SetOfCliques->ListCliques[OrderCliques[i]]->Normalize(genepoollimit); //(actualpoolsize); 
     } 

   if (LearningType != 2 && LearningType != 6 && LearningType != 7 )  
     {
      for (i=0;i<NeededOvelapp; i++)   ListOvelap[i]->Normalize(genepoollimit);      
     }
  
   if (LearningType == 6 || LearningType == 7)  
    {
     // cout<<"CR is "<<0.5*current_gen<<"  "<<ncliques<<endl;
     // for (i=0;i<ncliques; i++)   SetOfCliques->ListCliques[i]->printmarg();   
     for (i=0;i<ncliques; i++)   SetOfCliques->ListCliques[i]->NormalizeBoltzmann(1/(0.5*current_gen));    
   
  }

 }
 
}       
 


 
void DynFDA::MarkovCallProb()   
{   
    int i; 
      SetOrderofCliques();
      if(LearningType==0)  AncestralOrderingFact(1); // MN-FDA illegal factorizations
      else if(LearningType==4) AncestralOrderingFactOnlyCliq(1); // MN-FDA legal factorization     
  
        CreateMarg(SetOfCliques->NumberCliques); 
  	ComputeMarg(SetOfCliques->NumberCliques); 
	Normalize(SetOfCliques->NumberCliques); 
 
        CondKikuchiApprox = new KikuchiApprox(CliqMLength,SetOfCliques->NumberNodes,-1);  
	 CondKikuchiApprox->FindKikuchiApproximation(SetOfCliques->ListCliques, SetOfCliques->NumberCliques);  
	 //CondKikuchiApprox->FindOneLevelKikuchiApproximation(SetOfCliques->ListCliques, SetOfCliques->NumberCliques);
	 CondKikuchiApprox->CreateMarg(Pop->dim); 

        ListKikuchiCliques = new memberlistKikuchiClique*[SetOfCliques->NumberNodes]; 
        for (i=0;i<SetOfCliques->NumberNodes; i++) ListKikuchiCliques[i] = (memberlistKikuchiClique*)0; 
        CondKikuchiApprox->FillListKikuchiCliques(ListKikuchiCliques); 
	  
	//printmarg(); 	  
}   
   
void DynFDA::FDACallProb()   
{   
   
        SetOrderofCliques();  
        if(LearningType==1)  AncestralOrderingFact(1); // MN-FDA illegal factorizations
        else if(LearningType==3 || LearningType==5 )  AncestralOrderingFactOnlyCliq(1);  // MN-FDA legal factorization  
        else if(LearningType==6 || LearningType == 7 || LearningType == 8 || LearningType == 9) NeededCliques = length;
        CreateMarg(NeededCliques); 
  	ComputeMarg(NeededCliques); 
       	Normalize(NeededCliques);          
} 
 

void DynFDA::FGDACallProb()   
{      
        SetOrderofCliques();  
        if(LearningType==1)  AncestralOrderingFact(1); // MN-FDA illegal factorizations
        else if(LearningType==3 || LearningType==5 )  AncestralOrderingFactOnlyCliq(1);  // MN-FDA legal factorization  
        else if(LearningType==6 || LearningType == 7 || LearningType == 8 || LearningType == 9) NeededCliques = length;
        CreateMarg(SetOfCliques->NumberCliques); 
  	ComputeMarg(SetOfCliques->NumberCliques); 
	Normalize(SetOfCliques->NumberCliques);          
} 


void DynFDA::SetFactorizationFromMatrix(int** OtherMatrix)   
{
  UpdateModelFromMatrix(OtherMatrix);    
  SetOrderofCliques();  
  if(LearningType==1)  AncestralOrderingFact(1); // MN-FDA illegal factorizations
  else if(LearningType==3 || LearningType==5 )  AncestralOrderingFactOnlyCliq(1);  // MN-FDA legal factorization  
  else if(LearningType==6 || LearningType == 7 || LearningType == 8 || LearningType == 9) NeededCliques = length;
}

void DynFDA::FDACallProbFromMatrix()   
{
        FindUnivProb();        
        CreateMarg(SetOfCliques->NumberCliques); 
  	ComputeMarg(SetOfCliques->NumberCliques); 
	Normalize(SetOfCliques->NumberCliques);          
} 


void  DynFDA::SetProtein(HPProtein* HPProt)
{
  FoldingProtein = HPProt;
}



void DynFDA::GenStructuredCrossPop(int From,Popul* NewPop, Popul* SelPop)   
{    
  int i,parent1,parent2;   
       for(i=From; i<NewPop->psize; i++)   
	 {   
	   parent1 = randomint(SelPop->psize);
           parent2 = randomint(SelPop->psize);	  
           GenStructuredCrossInd(NewPop,i, SelPop->P[parent1],SelPop->P[parent2]);           
        }
}     


void DynFDA::GenCrossPop(int From,Popul* NewPop, Popul* SelPop)   
{    
  int i,parent1,parent2;   
       for(i=From; i<NewPop->psize; i++)   
	 {   
	   parent1 = randomint(SelPop->psize);
           parent2 = randomint(SelPop->psize);	  
           GenOnePointCXInd(NewPop,i, SelPop->P[parent1],SelPop->P[parent2]);           
        }
}   


void DynFDA::GenPop(int From,int To,  Popul* NewPop)   
{    
   int i,j;   
      if (LearningType == 5)  GenPopMPM(From,To,NewPop); //Marginal Product Model (learned using affinity propagation)   
      else
	{
         for(i=From; i<To; i++)   
	 {   
	    if (LearningType==0 || LearningType==4) //MN-EDA FDA init
                  {
                    GenIndividualMNFDA(NewPop,i,auxvectGen,legalconfGen); 
                    GenIndividualMRF(NewPop,i,auxvectGen,legalconfGen); 
                  } 
            else if (LearningType==1  || LearningType==3)  GenIndividualMNFDA(NewPop,i,auxvectGen,legalconfGen);  // MN-FDA (illegal and legal factorizations)
            else if(LearningType==2) // MN-EDA random init
               {
                for (j=0;j<length; j++)  auxvectGen[j] = (myrand()>0.5);  
                GenIndividualMRF(NewPop,i,auxvectGen,legalconfGen); 
               }   
          }               
        }
}     
  

void DynFDA::GenBiasedPop(int From, Popul* NewPop, unsigned int* MPCConf, double chunkprob)   
{ 
  int i;
  GenPop(From,NewPop);
  for(i=From; i<NewPop->psize; i++)   
   {
     GenBiasedIndividual(NewPop,i,MPCConf,chunkprob);  
   }

}



void DynFDA::GenPop(int From, Popul* NewPop)   
{    
   int i,j;   
      if (LearningType == 5)  GenPopMPM(From,NewPop->psize,NewPop); //Marginal Product Model (learned using affinity propagation)   
      else
	{
         for(i=From; i<NewPop->psize; i++)   
	 {   
	    if (LearningType==0 || LearningType==4) //MN-EDA FDA init
                  {
                    GenIndividualMNFDA(NewPop,i,auxvectGen,legalconfGen); 
                    GenIndividualMRF(NewPop,i,auxvectGen,legalconfGen); 
                  } 
            else if (LearningType==1  || LearningType==3)  GenIndividualMNFDA(NewPop,i,auxvectGen,legalconfGen);  // MN-FDA (illegal and legal factorizations)
            else if(LearningType==2) // MN-EDA random init
               {
                for (j=0;j<length; j++)  auxvectGen[j] = (myrand()>0.5);  
                GenIndividualMRF(NewPop,i,auxvectGen,legalconfGen); 
               }  
              else if(LearningType==6 || LearningType==8) // MOA
               {
                   for (j=0;j<length; j++)  auxvectGen[j] = (myrand()>0.5);  
		   GenIndividualMOA(NewPop,i,auxvectGen,legalconfGen,int(log(length)*cycles));
                    
               }    
              else if(LearningType==7 || LearningType==9) // MOA
               {
                 if(From==0)
		  { 
                   for (j=0;j<length; j++)  auxvectGen[j] = (myrand()>0.5);  
		  }
                 else
		   {
                    for (j=0;j<length; j++)  auxvectGen[j] = NewPop->P[i][j];  
                    GenIndividualMOA(NewPop,i,auxvectGen,legalconfGen,int(log(length)*cycles));
                   } 
               }    
          }               
        }
}     



        void DynFDA::GenPopMPM(int From, int To, Popul* NewPop)   
        {    
          int i,j;   
          int *auxPopIndex, *auxPopValues;
          auxPopIndex = new int[NewPop->psize - From];
          auxPopValues = new int[NewPop->psize - From];

          for (i=From; i<To; i++) auxPopIndex[i-From] = i;
 
         
          for (i=0;i<NeededCliques; i++)   
             {  	          
  	             RandomPerm(NewPop->psize - From, NewPop->psize - From,auxPopIndex); 
                     SUS(SetOfCliques->ListCliques[OrderCliques[i]]->r_NumberCases, SetOfCliques->ListCliques[OrderCliques[i]]->marg, NewPop->psize - From, auxPopValues);
		     //cout<<" SUS was finished "<<endl;
                    for (j=From;j<NewPop->psize; j++)  
		      {
			// cout<<j<<" "<<auxPopValues[j]<<" "<<auxPopIndex[j]<<endl;
                        SetOfCliques->ListCliques[OrderCliques[i]]->PutCase(auxPopValues[j-From],NewPop->P[auxPopIndex[j-From]]);      
                      }             
           }   

	  delete[] auxPopIndex;
          delete[] auxPopValues;
 
        } 


void DynFDA::GenPopProtein(int From, Popul* NewPop, double GenPrior)   
{    
   int i,j;   
   
  //for (i=0;i<NeededCliques; i++)   cout<<OrderCliques[i]<< "  "<<i<<endl;
//SetOfCliques->ListCliques[OrderCliques[i]]->Normalize(genepoollimit,GenPrior); //Modificacion Protein

     for(i=From; i<NewPop->psize; i++)   
       {
	 GenIndividualMNFDAProtein(NewPop,i,auxvectGen,legalconfGen);            
         //NewGenIndividualMNFDAProtein(NewPop,i,auxvectGen,legalconfGen);       
   	} 

       //	for (i=0;i<length; i++)  cout<<FoldingProtein->statvector[i]<<" ";
       // cout<<endl;
 
}     



void DynFDA::GenIndividualMNFDAProtein(Popul* NewPop, int pos, int* auxvect, int* legalconf)   
{   
   
  // The vector in position pos is generated   
 int i,j,enn;   
 int numberinst;  
 int totinst;
 totinst = 0;
 
    for (i=0;i<length; i++)    
          {
            auxvect[i] = -1;  
            FoldingProtein->statvector[i] = 0;
          }
     
      for (i=0;i<NeededCliques; i++)   
             {  	          
  	       numberinst = SetOfCliques->ListCliques[OrderCliques[i]]->NumberInstantiated(auxvect);  	       
               if (numberinst == 0) 
                   {
                      SetOfCliques->ListCliques[OrderCliques[i]]->Instantiate(auxvect);  
                      totinst +=  SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars;
                    }                 
	       else if(numberinst <  SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars)
		 {   
		       SetOfCliques->ListCliques[OrderCliques[i]]->PartialInstantiate(auxvect,legalconf);        
                       totinst +=  (SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars - numberinst); 
                 } 
	       //FoldingProtein->CallRepair(auxvect,totinst);    
             
           }   

     /*
     if(myrand()> 0.5)
           {   
              for (i=0;i<length; i++)  
                 {     
                   if(myrand()> 0.9)
                        {
                          if (auxvect[i] == 0)  auxvect[i] = randomint(2)+1;
			  else if (auxvect[i] == 1) auxvect[i] = 2*randomint(2);
                          else auxvect[i] = randomint(2);
                        }   
                 }  
           } 
     */
	
     

    totinst = length;

    // FoldingProtein->CallRepair(auxvect,totinst);
    
                for (i=0;i<length; i++)  
                 {  
                      NewPop->P[pos][i] = (auxvect[i]);   
                 }  
	
    }   
  








void DynFDA::NewGenIndividualMNFDAProtein(Popul* NewPop, int pos, int* auxvect, int* legalconf)   // Combines the generation with a local opt. method
{   

  // The vector in position pos is generated   
 int i,j;   
 int numberinst;  
 int totinst;
 int* uniquepos;
 int  nmoves;
 double legprob[5],cutoff,tot;
 tot = 0;
 cutoff = 0;
 totinst = 0;

     for (i=0;i<length; i++)  
          {
            auxvect[i] = -1;  
            FoldingProtein->statvector[i] = 0;
          }

      for (i=0;i<NeededCliques; i++)   
         {  	     
                
               numberinst = SetOfCliques->ListCliques[OrderCliques[i]]->NumberInstantiated(auxvect);  	    
               //cout<<i<<" "<<numberinst<<" "<<totinst<<endl;   
               if (numberinst == 0)                                                         // No variables have been instantiated
                   {
		     SetOfCliques->ListCliques[OrderCliques[i]]->Instantiate(auxvect);    //They are instantiated
                      totinst +=  SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars;
                      for (j=0;j<totinst;j++) FoldingProtein->SetPosInGrid(auxvect[j], j);      // The grid is updated
                    }                 
	       else if(numberinst <  SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars)  // Only one variable will be instantiated
		 {   
		   nmoves = FoldingProtein->FindGridProbabilities(totinst);  //nmoves correspond to number of legal movements
                   //cout<<"nmoves "<<nmoves<<endl;
		   //for (j=0;j<3;j++)  cout<< FoldingProtein->moveprob[j]<<" ";
                   //cout<<endl;
		   if (nmoves==0)  //  There are not legal moves 
                        {                        
			 auxvect[totinst] = randomint(3); // A random move is chosen and Repair is called
                         //FoldingProtein->CallRepair(auxvect,totinst);
                         FoldingProtein->SetPosInGrid(auxvect[totinst],totinst); // The grid is updated
                         totinst++;
                        }  
		   else if (nmoves==1) // Only one legal move is possible  
                        {
                          auxvect[totinst] =  FoldingProtein->theone;  // Assignment is direct
                          FoldingProtein->SetPosInGrid(auxvect[totinst],totinst); // The grid is updated
                          totinst++;
                        }   
		   else   //There are many possible movements, probabilities have to be used
                      {                         
			SetOfCliques->ListCliques[OrderCliques[i]]->GiveProb(auxvect,legprob); // EDA prob are stored in legprob  
                        tot = 0;
                        for (j=0;j<3;j++)  
                          {
                            if(FoldingProtein->moveprob[j]==0) legprob[j] = 0;
                            else tot += legprob[j];			                                  
                          }

			for (j=0;j<3;j++)  
                             { 
			       //cout<<"alpha"<<FoldingProtein->alpha<<endl;
                               //legprob[j] = FoldingProtein->alpha*FoldingProtein->moveprob[j] + (1.0-FoldingProtein->alpha)*(legprob[j]/tot);  // alpha weighting                      
			        legprob[j] = (legprob[j]/tot); 
                               //legprob[j] = FoldingProtein->moveprob[j];  // alpha weighting 
                               // legprob[j] = 0.2*FoldingProtein->moveprob[j] + (0.8)*(legprob[j]/tot);  // alpha
                               //     cout<<j<<" "<<legprob[j]<<endl;
                             } 
                      
		       cutoff = myrand(); //Maximum number the cutoff can take	   
                       j = 0;   
	               tot = legprob[0];   
	               while ( (tot<cutoff) && (j<3) )   
	                {    
		          j++;   
		          tot += legprob[j];   
			  // cout<<"j "<<j<<" tot "<<tot<<" cut "<<cutoff<<endl;   
                        }           
	                // If the following happens there's a mistake   
	                 if (j>=3)   
	                  {  
                            cout<<"Algo justo aqui"<<endl;   
       	                    j = randomint(3);   
	                  }
                         auxvect[totinst] =  j; 
                         //cout<<"j is "<<j<<endl;
                         FoldingProtein->SetPosInGrid(auxvect[totinst],totinst);
                         totinst +=  (SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars - numberinst);                      
                      } 
                     }      
              } 
	       //FoldingProtein->CallRepair(auxvect,totinst);    
  
       FoldingProtein->CleanGridFromPos(length); //18-4-2005 
                for (i=0;i<length; i++)  
                 {  
                      NewPop->P[pos][i] = (auxvect[i]);   
                 }  
	
    }   
  




void DynFDA::GenIndividual(Popul* NewPop,int pos, int extra)   
{    
   int j;   
  
	    if (LearningType==0 || LearningType==4) //MN-EDA FDA init
                  {
                    GenIndividualMNFDA(NewPop,pos,auxvectGen,legalconfGen); 
                    GenIndividualMRF(NewPop,pos,auxvectGen,legalconfGen); 
                  } 
            else if (LearningType==1  || LearningType==3)  
                    GenIndividualMNFDA(NewPop,pos,auxvectGen,legalconfGen);  // MN-FDA (illegal and legal factorizations)
            else if(LearningType==2) // MN-EDA random init
               {
                for (j=0;j<length; j++)  auxvectGen[j] = (myrand()>0.5);  
                GenIndividualMRF(NewPop,pos,auxvectGen,legalconfGen); 
               }     
  
             
 }     






void DynFDA::GenStructuredCrossInd(Popul* NewPop, int pos, unsigned int* parind1, unsigned int* parind2)   
{   
   
  // The vector in position pos is generated   
  int i,thevar;   
 int numberinst;  

     for (i=0;i<SetOfCliques->NumberCliques; i++)   
		 {  		       
		   if(myrand()<0.5)
                     {		      
		       for(int j=0;j<SetOfCliques->ListCliques[i]->NumberVars;j++)
			 {
                           thevar = SetOfCliques->ListCliques[i]->vars[j];                         
                           NewPop->P[pos][thevar] = parind1[thevar]; 
                         }
                     }    
		   else 
                     {		      
		       for(int j=0;j<SetOfCliques->ListCliques[i]->NumberVars;j++)
			 {
                           thevar = SetOfCliques->ListCliques[i]->vars[j];  
                           NewPop->P[pos][thevar] = parind2[thevar];  
                         }
                     }    
                 
                 }  

    }   
  

// This function modifies an individual by copying some chunks (corresponding to cliques) of another solution (MPC or other)
// the chunks are copied according to some defined probability. It is similar to a clique-based crossover operator
void DynFDA::GenBiasedIndividual(Popul* NewPop, int pos,  unsigned int* MPCConf, double chunkprob)   
{      
  // The vector in position pos is generated   
  int i,j,thevar;   
  //cout<<SetOfCliques->NumberCliques<<endl;
          for (i=0;i<SetOfCliques->NumberCliques; i++)   
		 {  		       
		   if(myrand()<chunkprob)
                     {
		       //cout<<"pos "<<pos<<"  Cliq "<<i<<" nvars "<<SetOfCliques->ListCliques[i]->NumberVars<<endl;
		       for(int j=0;j<SetOfCliques->ListCliques[i]->NumberVars;j++)
			 {
                           thevar = SetOfCliques->ListCliques[i]->vars[j];
                           //cout<<j<<"  "<<SetOfCliques->ListCliques[i]->NumberVars<<"  "<<thevar<<endl;
                           NewPop->P[pos][thevar] = MPCConf[thevar]; 
                         }
                     }                     
                 }  
}   

	   
void DynFDA::GenIndividualMNFDA(Popul* NewPop, int pos, int* auxvect, int* legalconf)   
{   
   
  // The vector in position pos is generated   
 int i;   
 int numberinst;  

     for (i=0;i<length; i++)  auxvect[i] = -1;  
     for (i=0;i<NeededCliques; i++)   
		 {  		       
		     numberinst = SetOfCliques->ListCliques[OrderCliques[i]]->NumberInstantiated(auxvect);  	       // if(pos==1) SetOfCliques->ListCliques[OrderCliques[i]]->print(); 

        if (numberinst == 0)  
                      SetOfCliques->ListCliques[OrderCliques[i]]->Instantiate(auxvect);  		                 
        else if(numberinst <  SetOfCliques->ListCliques[OrderCliques[i]]->NumberVars)  
		       SetOfCliques->ListCliques[OrderCliques[i]]->PartialInstantiate(auxvect,legalconf);                                     
                 }  
                for (i=0;i<length; i++)  
                 {     
                   NewPop->P[pos][i] = (auxvect[i]);   
                 }  

  }   
  


void DynFDA::GenIndividualMOA(Popul* NewPop, int pos, int* auxvect, int* legalconf, int iters)   
{   
   
  // The vector in position pos is generated   
  int i,j,k,thevar,auxcliq;   

  

int numberinst;  

  
//cout<<"ITERS " <<iters<<"  "<<NeededCliques<<endl;
 
   for(j=0;j<iters;j++)
     {   
       for (i=0;i<NeededCliques; i++)         // For MOA NeededCliques and legnth should be equal (one clique per var)
		 {  		       
                   auxcliq = randomint(NeededCliques);
		   //cout<<"clique number "<<auxcliq<<endl;

		     if(SetOfCliques->ListCliques[auxcliq]->NumberVars==1)  
		      SetOfCliques->ListCliques[auxcliq]->Instantiate(auxvect);
                     else
                     {
		      thevar =  SetOfCliques->ListCliques[auxcliq]->vars[0]; //The var to be updated is the first in each clique
		   // for(k=1;k<SetOfCliques->ListCliques[auxcliq]->NumberVars;k++) auxvect[SetOfCliques->ListCliques[auxcliq]->vars[k]] = -1;
		     auxvect[thevar] = -1;                     
		     SetOfCliques->ListCliques[auxcliq]->PartialInstantiate(auxvect,legalconf);
		     // SetOfCliques->ListCliques[auxcliq]->Instantiate(auxvect);
		     //if(j==0) 
		     // {
		       // cout<<"clique number "<<auxcliq<<endl;
		       //  SetOfCliques->ListCliques[auxcliq]->print();   //printmarg();
		       // SetOfCliques->ListCliques[auxcliq]->printmarg();
		     // }
		     }
                 }
             
       //for (i=0;i<length; i++) cout<<auxvect[i]<<" ";
       // cout<<endl;
		
     }  
                 for (i=0;i<length; i++)  
                 {     
                   NewPop->P[pos][i] = (auxvect[i]);   
                 }  
}
     
  



void DynFDA::GenOnePointCXInd(Popul* NewPop, int pos, unsigned int* parind1, unsigned int* parind2)   
{   
   
  // The vector in position pos is generated   
  int i,thevar;   
  int numberinst;  
  int CXPoint;
  CXPoint = randomint(length-1)+1;
  for (i=0;i<CXPoint; i++)  NewPop->P[pos][i] =  parind1[i]; 
  for (i=CXPoint;i<length; i++)  NewPop->P[pos][i] =  parind2[i];
}   


 
 int DynFDA::SamplingVarKikuchi(int* conf,int varpos)  
  {  
      int j,l,varposfreq;  
      double prob[2], finprob[2];  
      memberlistclique* actualcliq;  
      memberlistKikuchiClique* actualKcliq; 
      finprob[0] = 1.0;  finprob[1] = 1.0;  
      int* auxconf;  
      varposfreq = 0;
      actualcliq = SetOfCliques->CliquesPerNodes[varpos];  //Original cliques for each variable
        
      while (actualcliq!=(memberlistclique*)0)  
      {  
	      SetOfCliques->ListCliques[actualcliq->cliq]->GetValProb(conf,varpos,prob);  
	      finprob[0] *= (prob[0]);    
              finprob[1] *= (prob[1]);   
              actualcliq = actualcliq->nextcliq;
              varposfreq++;
      }  
  
      auxconf = new int[SetOfCliques->NumberNodes];   
      for(j=0;j<SetOfCliques->NumberNodes; j++) auxconf[j] = conf[j];  
  
      actualKcliq = ListKikuchiCliques[varpos];                  
        while (actualKcliq!=(memberlistKikuchiClique*)0)  
         {  
	     // for(j=0;j<actualKcliq->KCliq->NumberVars; j++) auxconf[actualKcliq->KCliq->vars[j]] = -1;   
       	  for(j=0;j<actualKcliq->KCliq->father->NumberVars; j++) auxconf[actualKcliq->KCliq->father->vars[j]] = -1;    
 for(j=0;j<actualKcliq->KCliq->NumberVars; j++) auxconf[actualKcliq->KCliq->vars[j]] = conf[actualKcliq->KCliq->vars[j]];   
            actualKcliq->KCliq->father->GetValProb(auxconf,varpos,prob);  
	    // for(j=0;j<actualKcliq->KCliq->NumberVars; j++) auxconf[actualKcliq->KCliq->vars[j]] = conf[actualKcliq->KCliq->vars[j]];  
     for(j=0;j<actualKcliq->KCliq->father->NumberVars; j++) auxconf[actualKcliq->KCliq->father->vars[j]] = conf[actualKcliq->KCliq->father->vars[j]];
         
           varposfreq += actualKcliq->KCliq->count;

            for(l=0;l<abs(actualKcliq->KCliq->count);l++) 
            { 
	     if (actualKcliq->KCliq->sign==1) 
	     {    
	      finprob[0] *= (prob[0]);    
              finprob[1] *= (prob[1]);   
             } 
	     else if ( prob[0]!=0 &&  prob[1]!=0) 
             {    
	      finprob[0] /= (prob[0]);    
              finprob[1] /= (prob[1]);   
             } 
            } 
            actualKcliq = actualKcliq->nextcliq;  
         }  
      //cout<< "finpron "<<finprob[0]<< " "<<finprob[1]<<endl; 
      delete[] auxconf;  

      if(AllProb[varpos] != 0 && AllProb[varpos] != 1)
      {
       if (varposfreq>1) 
        for(l=1;l<varposfreq;l++) 
         {
          finprob[0] /= (1-AllProb[varpos]);    
          finprob[1] /= (AllProb[varpos]);  
         }
       else if (varposfreq<1) 
        for(l=varposfreq;l<1;l++) 
         {
          finprob[0] *= (1-AllProb[varpos]);    
          finprob[1] *= (AllProb[varpos]);  
         }
       }
      if( finprob[0]+ finprob[1] > 0)
      {
       finprob[0] =  finprob[0]/(finprob[0]+finprob[1]);  
       finprob[1] = 1.0- finprob[0];  
       if(myrand()<finprob[1]) return 1;  
       return 0;  
      }
      for(l=0;l<length;l++) cout<<AllProb[l]<<" ";
      cout<<endl;
      cout<< "It entered here with finprob[0]="<<finprob[0]<<"  finprob[1]="<<finprob[1]<<endl;
      return (myrand()>0.5);
   }  
  
  
  
void DynFDA::GenIndividualMRF (Popul* NewPop, int pos, int* auxvect, int* legalconf)   
{   
   
  // The vector in position pos is generated using GS from the   
  // marginals of all the cliques  
 int i,j;   
  
 //if(pos==1) 
 // for (i=0;i<length; i++)  auxvect[i] = (myrand()>0.5);  
 //  else  for (i=0;i<length; i++)  auxvect[i] = NewPop->P[pos-1][i];  
   
 
/*  
cout<<" pos="<<pos<<"  ";  
for (int ii=0;ii<length; ii++) cout<<auxvect[ii]<<" ";  
*/   
 int a = 1;  
                   
 for(j=0;j<cycles;j++)  
 {      
    for (i=0;i<length; i++)   
	 {   
             a = randomint(length);  
	     auxvect[a] = -1;    
             auxvect[a] = SamplingVarKikuchi(auxvect,a); //KikuchiFindMargForGS(auxvect,a); 	         
         }                    
  }  
  
 /* 
 cout<<endl<<" ->  ";  
 for (int ii=0;ii<length; ii++) cout<<auxvect[ii]<<" ";  
 cout<<endl;  
 */ 
  
   for (i=0;i<length; i++)  
      {     
        NewPop->P[pos][i] = (auxvect[i]);   
      }    
		  
 }   
   
  
  
void DynFDA::PrintMatrix(double* Matrix)   
{   
  int aux,j,k;   
   
	for(j=0; j<length; j++)   
	{   
	  	for(k=0 ; k<length; k++)   
		{   
                if (k==j)   
		  {   
		       
			printf("%2.2f ",0.00);  	          
			     
		  }	     
		   else if(j<k)   
		   {    
  			aux = j*(2*length-j+1)/2 +k-2*j-1;   
			printf("%2.2f ",Matrix[aux]);   
		   }   
		   else   
			{    
			aux = k*(2*length-k+1)/2 +j-2*k-1;   
                        printf("%2.2f ",Matrix[aux]);   
			  
			  
		   }   
		}	   
		  
		printf("\n");   
	}   
	  
}   
   
   
  
/////////////////////////////////////////////////////  
  
  
  
  
double DynFDA::FindChiSquarelimit()  
{  
    double chival;  
  int i,k,j;   
   int aux;   
   unsigned current_pair[2];  
   double bivfreq[4];   
   double univfreq1,univfreq2,x;  
   int *indexpairs;   
   int remaining_configurations;   
   double OrvBivProb,ExpBivProb ;  //Observed and expected biv prob. for chi square  
     
    chival = 0;  
      
  
   indexpairs = new int[actualpoolsize];          
  
 	 for(j=0; j<length-1; j++)         // For all possible pairs of variables   
	  {  
	      k= length-1;   
                 aux = j*(2*length-j+1)/2 +k-2*j-1;   
		 x=0;   
		 for(i=0;i<4;i++) bivfreq[i] = 0; //introduce priors  
      	    
	     for(i=0; i<actualpoolsize; i++) indexpairs[i] = i;   
	     remaining_configurations = actualpoolsize;   
 	    
        while(remaining_configurations > 0)   
          {   
			    OrvBivProb = 0;                  
				current_pair[0] = Pop->P[indexpairs[0]][j];   
				current_pair[1] = Pop->P[indexpairs[0]][k];   
   
			    i = 0;   
			    while(i< remaining_configurations)   
				{    
                 if( current_pair[0] == Pop->P[indexpairs[i]][j] &&    
				   current_pair[1] == Pop->P[indexpairs[i]][k] )    
				 {   
                                  OrvBivProb += PopProb[indexpairs[i]];   
				  indexpairs[i] = indexpairs[remaining_configurations-1];   
				  remaining_configurations--;   
				 }   
			     else i++;   
				} 			    
			    bivfreq[2*current_pair[0]+current_pair[1]] += OrvBivProb;  
	   }  
	for(i=0;i<4;i++)   
          {  
            bivfreq[i] = (bivfreq[i]*genepoollimit+1.0)/(genepoollimit+4.0); //normalize prob with priors   
	    //  cout<<bivfreq[i]<<" ";  
            }  
	//cout<<endl;  
	                    univfreq1 = bivfreq[2]+bivfreq[3];  
                            univfreq2 = bivfreq[1]+bivfreq[3];  
			     
                            OrvBivProb = bivfreq[0];   
                            ExpBivProb = (1-univfreq1)* (1-univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb; // ((O - E)2/E)  
                            //cout<<j<<k<<"  Real "<<OrvBivProb<<" Expected "<<ExpBivProb<<" Chi "<<x<<endl;  
  
                            OrvBivProb = bivfreq[1];   
                            ExpBivProb = (1-univfreq1)* (univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb;  
			      
  
                            OrvBivProb = bivfreq[2];   
                            ExpBivProb = (univfreq1)* (1-univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb;    
                            
                            OrvBivProb = bivfreq[3];   
                            ExpBivProb = (univfreq1)* (univfreq2);  
                            x += ((OrvBivProb - ExpBivProb)*(OrvBivProb - ExpBivProb))/ExpBivProb;   
 			    x*= actualpoolsize; //genepoollimit;  
  
     
/*			      
  
                            OrvBivProb = bivfreq[0];   
                            ExpBivProb = (1-univfreq1)* (1-univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
                            //cout<<j<<k<<"  Real "<<OrvBivProb<<" Expected "<<ExpBivProb<<" Chi "<<x<<endl;  
  
                            OrvBivProb = bivfreq[1];   
                            ExpBivProb = (1-univfreq1)* (univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
                            OrvBivProb = bivfreq[2];   
                            ExpBivProb = (univfreq1)* (1-univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
                            OrvBivProb = bivfreq[3];   
                            ExpBivProb = (univfreq1)* (univfreq2);  
                            x += OrvBivProb* log(ExpBivProb/ OrvBivProb);  
                            x*= -2*genepoollimit; //Important to determine if is actualpoolsize  
                 */  
			    //    cout<<"j "<<j<<" k "<< k<<" Chi "<<x<<" valchi "<< valchi<<endl;      
			    if(x>chival) chival = x;  
		            //chival += x;  
		}  
	  
		delete[] indexpairs;   
                //chival /= (length-1);  
                cout<<"valchi= "<<chival<<endl;  
               	return chival;  
}  
  
double  DynFDA::FindBivlimit()   
 {   
    double valchi;  
    int count;  
int i,k,j,z,aux, remaining_configurations;  
   unsigned current_pair[3];  
   double freqxyz[8],freqxz[4],freqyz[4],freqz;  
   int *indexpairs;   
   double OrvBivProb,ExpBivProb,sqval,x ;  //Observed and expected biv prob. for chi square  
   
	indexpairs = new int[actualpoolsize];          
	valchi = 0;  
        count =0;  
         z =  randomint(length)-1;   
	 cout<<"z= "<<z<<endl;  
	 j = length-1;  
	  	for(k=0 ; k<length-1; k++)   
		{   
                  
                  aux = j*(2*length-j+1)/2 +k-2*j-1;   
                  sqval=0;   
                 
		 if (z!=j && z!=k )  
		 {                      
		    for(i=0;i<8;i++) freqxyz[i] = 0.0; //1.0/(genepoollimit+8); //introduce priors  
      	            count++;  
	            for(i=0; i<actualpoolsize; i++) indexpairs[i] = i;   
	            remaining_configurations = actualpoolsize;   
 	    
                    while(remaining_configurations > 0)   
                      {   
			        OrvBivProb = 0;                  
				current_pair[0] = Pop->P[indexpairs[0]][j];   
				current_pair[1] = Pop->P[indexpairs[0]][k];  
                                current_pair[2] = Pop->P[indexpairs[0]][z];   
   
			    i = 0;   
			    while(i< remaining_configurations)   
				{    
                                if( current_pair[0] == Pop->P[indexpairs[i]][j] &&    
				   current_pair[1] == Pop->P[indexpairs[i]][k]  
                                  && current_pair[2] == Pop->P[indexpairs[i]][z] )    
				 {   
                                  OrvBivProb += PopProb[indexpairs[i]];   
				  indexpairs[i] = indexpairs[remaining_configurations-1];   
				  remaining_configurations--;   
				 }   
			        else i++;   
				} 			    
			    freqxyz[4*current_pair[0]+2*current_pair[1]+current_pair[2]] += OrvBivProb;  
	                 }  
		    //cout<<genepoollimit<<" "<<j <<","<<k<<","<<z<<" ";  
	          for(i=0;i<8;i++)   
                    {  
			freqxyz[i] = (freqxyz[i]*genepoollimit+1.0)/(genepoollimit+8.0); //normalize prob with priors  
			//cout<< freqxyz[i]<<" ";  
                    }     
		  //cout<<endl;  
	          freqz = freqxyz[1] + freqxyz[3] + freqxyz[5] + freqxyz[7]; // z=1  
                  freqxz[0] = freqxyz[0] + freqxyz[2]; freqxz[1] = freqxyz[1] + freqxyz[3];  
                  freqxz[2] = freqxyz[4] + freqxyz[6]; freqxz[3] = freqxyz[5] + freqxyz[7];  
                  freqyz[0] = freqxyz[0] + freqxyz[4]; freqyz[1] = freqxyz[1] + freqxyz[5];  
                  freqyz[2] = freqxyz[2] + freqxyz[6]; freqyz[3] = freqxyz[3] + freqxyz[7];  
		   
		    
                 OrvBivProb = freqxyz[0];   
                 ExpBivProb =  (freqxz[0]* freqyz[0])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb)*(OrvBivProb-ExpBivProb))/ExpBivProb;   
                 OrvBivProb = freqxyz[1];   
                 ExpBivProb =  (freqxz[1]* freqyz[1])/(freqz);  
                 sqval += ((OrvBivProb-ExpBivProb)*(OrvBivProb-ExpBivProb))/ExpBivProb;   
                 OrvBivProb = freqxyz[2];   
                 ExpBivProb =  (freqxz[0]* freqyz[2])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 OrvBivProb = freqxyz[3];   
                  ExpBivProb =  (freqxz[1]* freqyz[3])/(freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 OrvBivProb = freqxyz[4];   
                 ExpBivProb =  (freqxz[2]* freqyz[0])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 OrvBivProb = freqxyz[5];   
                 ExpBivProb =  (freqxz[3]* freqyz[1])/(freqz);   
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 OrvBivProb = freqxyz[6];   
                 ExpBivProb =  (freqxz[2]* freqyz[2])/(1- freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                 OrvBivProb = freqxyz[7];   
                 ExpBivProb =  (freqxz[3]* freqyz[3])/(freqz);  
                 sqval += ((OrvBivProb-ExpBivProb )*(OrvBivProb-ExpBivProb ))/ExpBivProb;   
                   
                 x= sqval*actualpoolsize; //genepoollimit;   
		   
		 /*  
                  OrvBivProb = freqxyz[0];   
                 ExpBivProb =  (freqxz[0]* freqyz[0])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 OrvBivProb = freqxyz[1];   
                 ExpBivProb =  (freqxz[1]* freqyz[1])/(freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 OrvBivProb = freqxyz[2];   
                 ExpBivProb =  (freqxz[0]* freqyz[2])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 OrvBivProb = freqxyz[3];   
                 ExpBivProb =  (freqxz[1]* freqyz[3])/(freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 OrvBivProb = freqxyz[4];   
                 ExpBivProb =  (freqxz[2]* freqyz[0])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 OrvBivProb = freqxyz[5];   
                 ExpBivProb =  (freqxz[3]* freqyz[1])/(freqz);   
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);  
                 OrvBivProb = freqxyz[6];   
                 ExpBivProb =  (freqxz[2]* freqyz[2])/(1- freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);    
                 OrvBivProb = freqxyz[7];   
                 ExpBivProb =  (freqxz[3]* freqyz[3])/(freqz);  
                 sqval += OrvBivProb*log(ExpBivProb/OrvBivProb);   
                               
		 x = -2*sqval*genepoollimit; //Important to determine if is actualpoolsize  
		 */   
		 if (x>valchi) valchi = x;  
                 //valchi +=x;  
		 }       
                                          
		}   
            
		delete[] indexpairs;  
                //valchi /= count;  
                cout<<"valchibiv= "<<valchi<<endl;  
                return valchi;  
  }	           			   
   
  
/*

 int DynFDA::KikuchiFindMargForGS(int* conf,int varpos)  
  {  
      int i,j,l;  
      double prob[2], finprob[2];  
      memberlistclique* actualcliq;  
      memberlistKikuchiClique* actualKcliq; 
      finprob[0] = 1.0;  finprob[1] = 1.0;  
      KikuchiApprox* CondKikuchiApprox;  
      int* auxconf;  
  
      actualcliq = SetOfCliques->CliquesPerNodes[varpos];  
        
      while (actualcliq!=(memberlistclique*)0)  
      {  
	  //SetOfCliques->ListCliques[actualcliq->cliq]->print(); 
	      SetOfCliques->ListCliques[actualcliq->cliq]->GetValProb(conf, varpos,prob);  
	      finprob[0] *= (prob[0]);    
              finprob[1] *= (prob[1]);   
              actualcliq = actualcliq->nextcliq;  
      }  
  
 
      CondKikuchiApprox = new KikuchiApprox(CliqMLength,SetOfCliques->NumberNodes,varpos);  
      CondKikuchiApprox->FindKikuchiApproximation(SetOfCliques->ListCliques, SetOfCliques->CliquesPerNodes[varpos]);  
  
      auxconf = new int[SetOfCliques->NumberNodes];   
      for(j=0;j<SetOfCliques->NumberNodes; j++) auxconf[j] = conf[j];  
  
      for (i=0;i<=CondKikuchiApprox->level; i++)  
       {    
	    
	   //cout<<"level "<<i<<endl; 
       actualKcliq = CondKikuchiApprox->FirstKCliqLevels[i]; 
        
       while (actualKcliq!=(memberlistKikuchiClique*)0)  
        { 
	    //cout<<"sign "<<actualKcliq->KCliq->sign<< "  count = "<<actualKcliq->KCliq->count<<"  "; 
	    //actualKcliq->KCliq->print(); 
         actualKcliq=actualKcliq->nextcliq; 
        } 
         
        actualKcliq = CondKikuchiApprox->FirstKCliqLevels[i];              
        while (actualKcliq!=(memberlistKikuchiClique*)0)  
         { //The following step has to be optimized  
	     for(j=0;j<actualKcliq->KCliq->NumberVars; j++) auxconf[actualKcliq->KCliq->vars[j]] = -1;          //actualKcliq->KCliq->print(); 
	      actualKcliq->KCliq->father->GetValProb(auxconf,varpos,prob);  
            for(j=0;j<actualKcliq->KCliq->NumberVars; j++) auxconf[actualKcliq->KCliq->vars[j]] = conf[actualKcliq->KCliq->vars[j]];  
            for(l=0;l<actualKcliq->KCliq->count;l++) 
            { 
	     if (actualKcliq->KCliq->sign==1) 
	     {    
	      finprob[0] *= (prob[0]);    
              finprob[1] *= (prob[1]);   
             } 
	     else if ( prob[0]!=0 &&  prob[1]!=0) 
             {    
	      finprob[0] /= (prob[0]);    
              finprob[1] /= (prob[1]);   
             } 
            } 
            actualKcliq = actualKcliq->nextcliq;  
         }  
      }  
      //cout<< "finpron "<<finprob[0]<< " "<<finprob[1]<<endl; 
      delete[] auxconf;  
      delete CondKikuchiApprox;  
      finprob[0] =  finprob[0]/(finprob[0]+finprob[1]);  
      finprob[1] = 1.0- finprob[0];  
      if(myrand()<finprob[1]) return 1;  
      return 0;  
   }  


  auxsolap  = 0;  
         for(l=0; l<SetOfCliques->ListCliques[auxlistcliques[k]]->NumberVars; l++)  
         auxsolap += (SetOfCliques->ListCliques[OrderCliques[j]]->VarIsInClique(SetOfCliques->ListCliques[auxlistcliques[k]]->vars[l]));  
         //cout<<"i "<<i<<" j "<<j<<" auxlistcliq[k] "<<auxlistcliques[k]<<" auxsolap "<<auxsolap<<endl; 
        

*/


