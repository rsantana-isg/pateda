#include <stdio.h> 
#include <memory.h> 
#include <math.h> 
#include <iostream> 
#include <fstream>

#include "RotamerClass.h" 
#include "auxfunc.h" 



void RotProtein::EmptyPsi()  
{ 
  int i,j;
  Psi = new double**[numberresidues];

  for(i=0;i<numberresidues;i++)
    {
      Psi[i] = new double*[numberresidues]; 
      for(j=0;j<numberresidues;j++)  Psi[i][j] = (double*)0;
    }

}




void RotProtein::PrintAdjacencyMatrix()  
{
  int i,j;
  
  for(i=0;i<numberresidues;i++) 
  {
    for(j=0;j<numberresidues;j++) 
      {     
	cout<<Matrix[i][j]<<" ";
      }
   
  }
}


void RotProtein::FillAdjacencyMatrix()  
{
  int j,k;
 
   
  order  = new int[numberresidues]; 
  Matrix = new unsigned int*[numberresidues];

      for(j=0;j<numberresidues;j++) 
      {
 	  order[j] = j+1;
          Matrix[j] =  new unsigned int[numberresidues];
          for(k=0;k<numberresidues;k++)  Matrix[j][k] = 0;
          Matrix[j][j] = 1;
      }      

  for(j=0;j<ncells;j++)
   {
     Matrix[ProteinContacts[0][j]][ProteinContacts[1][j]] = 1;
     Matrix[ProteinContacts[1][j]][ProteinContacts[0][j]] = 1;
   }     
}


RotProtein::RotProtein(FILE * stream)  
{  
  int j,k,l;

  Matrix = (unsigned int**) 0;
  order = (int*) 0;

  fscanf(stream, "%d ", &numberresidues);

 
  //cout<<numberresidues<<endl;
  

  RotNumber = new int[numberresidues];

  for(j=0;j<numberresidues;j++) 
  {
    fscanf(stream, "%d ", &RotNumber[j]);     
    //cout<<j<<" "<<RotNumber[j]<<endl;
  }

  // cout<<endl;

 Ls = new double*[numberresidues];

 for(j=0;j<numberresidues;j++)
   {
    Ls[j] = new double[RotNumber[j]];
    for(k=0;k<RotNumber[j];k++)
      {  
	 fscanf(stream, "%lf", &Ls[j][k]);  //Energy between backbone and residue for every rotamer
	 //cout<<j<<" "<<k<<" "<<Ls[j][k]<<endl;
      }
    
   }                                
 //cout<<endl;
   



 fscanf(stream, "%d ", &ncells);  // Number of cells  

 //cout<<ncells<<endl;

 ProteinContacts = new int*[2];
 ProteinContacts[0] = new int[ncells];
 ProteinContacts[1] = new int[ncells];

  
 for(j=0;j<ncells;j++)
   {
    fscanf(stream, "%d ", &ProteinContacts[0][j]); 
    fscanf(stream, "%d ", &ProteinContacts[1][j]); 
    ProteinContacts[0][j]--;
    ProteinContacts[1][j]--;
   
    //cout<<ProteinContacts[0][j]<<" "<<ProteinContacts[1][j]<<endl;
   }                                
      

  EmptyPsi();  

  for(j=0;j<ncells;j++)
   {
     // cout<<j<<" "<<ProteinContacts[0][j]<<" "<<ProteinContacts[1][j]<<" "<<RotNumber[ProteinContacts[0][j]]*RotNumber[ProteinContacts[1][j]]<<endl;
     Psi[ProteinContacts[0][j]][ProteinContacts[1][j]] = new double[RotNumber[ProteinContacts[0][j]]*RotNumber[ProteinContacts[1][j]]];
  

     for(k=0;k<RotNumber[ProteinContacts[0][j]];k++)
       for(l=0;l<RotNumber[ProteinContacts[1][j]];l++)
         {
           fscanf(stream, "%lf", &Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][k*RotNumber[ProteinContacts[1][j]]+l]); 

	   // cout<<j<<" "<<k<<" "<<l<<" "<<ProteinContacts[0][j]<<" "<<ProteinContacts[1][j]<<" "<<RotNumber[ProteinContacts[0][j]]<<" "<<RotNumber[ProteinContacts[1][j]]<<" "<<k*RotNumber[ProteinContacts[1][j]]+l<<endl;
         }  

        Psi[ProteinContacts[1][j]][ProteinContacts[0][j]] =  Psi[ProteinContacts[0][j]][ProteinContacts[1][j]]; //Pointing to the same place
	// cout<<j<<" "<<k<<" "<<l<<" "<<ProteinContacts[0][j]<<" "<<ProteinContacts[1][j]<<" "<<Psi[ProteinContacts[1][j]][ProteinContacts[0][j]]<<endl;
   }

  FinalRot  =  new int*[numberresidues];
  Fixed     =   new int[numberresidues];
  NFinalRot =   new int[numberresidues];
  RemainResPositions  =   new int[numberresidues];
  AcceptRes = new int*[numberresidues];
  for(j=0;j<numberresidues;j++)
   {
    AcceptRes[j] = new int[RotNumber[j]];
    for(k=0;k<RotNumber[j];k++) AcceptRes[j][k] = 1;
   }      
  VarPos = new int[numberresidues];
  NResiduesAfterDEE = 0;
  DissComponents = (int*) 0;
  ActiveContacts = (int*) 0;
}




RotProtein::RotProtein(FILE * stream, int ftype)  //Constructor for reading from 
{  
  int j,k,l,m;
  int auxint1,auxint2;
  char auxfilename[200]; 
  double auxdouble;
  int nrotamers,edge;
  int *indexnodes;


  Matrix = (unsigned int**) 0;
  order = (int*) 0;
 
  auxfilename[0] = '#';

  while(auxfilename[0]=='#')
  {
    fgets(auxfilename, 200,stream);
    cout<<auxfilename<<endl;    
  }
  // The last string read is the total number of rotamers
  
  fscanf(stream, "%d ", &numberresidues); //Number of residues
  //  cout<<"number of residues "<<numberresidues<<endl;
  

  RotNumber = new int[numberresidues];

  for(j=0;j<numberresidues;j++) 
  {
   fscanf(stream, "%d", &RotNumber[j]); 
   fscanf(stream, "%s", &auxfilename); 
  }

  nrotamers = RotNumber[numberresidues-1];  //The last value corresponds to the number of rotamers
  
   for(j=numberresidues-1;j>0;j--) 
    {   
     RotNumber[j] = RotNumber[j] - RotNumber[j-1]; //Relative number of rotamers
     //  cout<<j<<" -- "<<RotNumber[j]<<endl;
    }
   // cout<<endl;

 indexnodes = new int[2*nrotamers];
 m = 0;
 for(j=0;j<numberresidues;j++) //An index for all the nodes is constructed
  {
     for(k=0;k<RotNumber[j];k++) 
       {
        indexnodes[m] =  j;
        indexnodes[m+1] =  k;
        m+=2;
       }    
  }



 fscanf(stream, "%d ", &ncells);  // Number of cells  
 ncells = ncells - nrotamers; //The self energies will not be counted as energy
 //cout<<"Number of cells is"<<ncells<<endl;

//Energy between backbone and residue for every rotamer is read

 Ls = new double*[numberresidues];
 for(j=0;j<numberresidues;j++)
   {
    Ls[j] = new double[RotNumber[j]];
    for(k=0;k<RotNumber[j];k++)
      {  
        fscanf(stream, "%d %d %lg",&auxint1,&auxint2,&Ls[j][k]);  	
	//	    cout<<j<<" "<<k<<" "<<Ls[j][k]<<endl;
      }    
   }                                
 //cout<<endl;
   
// The pairwise interactions are read. At the same time the number of interacting residues
// is determined

 int MAXMEM = min(500,numberresidues*(numberresidues+1)/2);
 //cout<<"MAXMEN "<<MAXMEM<<endl;
 ProteinContacts = new int*[2];
 ProteinContacts[0] = new int[MAXMEM];
 ProteinContacts[1] = new int[MAXMEM];

 
 EmptyPsi();  

 edge = -1;
  for(j=0;j<ncells;j++)
   {
    fscanf(stream, "%d", &auxint1);
    fscanf(stream, "%d", &auxint2);
    fscanf(stream, "%lg",&auxdouble);
    //cout<<auxint1<<"---  "<<auxint2<<"---  "<<auxdouble<<endl; 
    k = indexnodes[2*auxint1-1];
    l = indexnodes[2*auxint2-1];

    if(k==0 && l==0) //First contact between rotamers
     {
       edge++;
       if(edge==MAXMEM) //To enlarge memory because the number of edges is not known in advance
        {

         MAXMEM += MAXMEM;          
         int* auxptr = new int[MAXMEM];
         for (m=0;m<edge;m++) auxptr[m] = ProteinContacts[0][m];
         delete[] ProteinContacts[0];
         ProteinContacts[0] = auxptr;
         auxptr = new int[MAXMEM];
         for (m=0;m<edge;m++) auxptr[m] = ProteinContacts[1][m];
         delete[] ProteinContacts[1];
         ProteinContacts[1] = auxptr;
        }
       ProteinContacts[0][edge] = indexnodes[2*auxint1-2];
       ProteinContacts[1][edge] = indexnodes[2*auxint2-2];
       Psi[ProteinContacts[0][edge]][ProteinContacts[1][edge]] = new double[RotNumber[ProteinContacts[0][edge]]*RotNumber[ProteinContacts[1][edge]]];      
       Psi[ProteinContacts[1][edge]][ProteinContacts[0][edge]] =  Psi[ProteinContacts[0][edge]][ProteinContacts[1][edge]];      
     }
       Psi[ProteinContacts[0][edge]][ProteinContacts[1][edge]][k*RotNumber[ProteinContacts[1][edge]]+l] = auxdouble; 

       cout<<edge<<" "<<k<<" "<<l<<" "<<ProteinContacts[0][edge]<<" "<<ProteinContacts[1][edge]<<" "<<RotNumber[ProteinContacts[0][edge]]<<" "<<RotNumber[ProteinContacts[1][edge]]<<" "<<k*RotNumber[ProteinContacts[1][edge]]+l<<" "<<  Psi[ProteinContacts[0][edge]][ProteinContacts[1][edge]][k*RotNumber[ProteinContacts[1][edge]]+l]<<endl;
     
}                                
  
  delete[] indexnodes;    
  FinalRot  =  new int*[numberresidues];
  Fixed     =   new int[numberresidues];
  NFinalRot =   new int[numberresidues];
  RemainResPositions  =   new int[numberresidues];
  AcceptRes = new int*[numberresidues];
  for(j=0;j<numberresidues;j++)
   {
    AcceptRes[j] = new int[RotNumber[j]];
    for(k=0;k<RotNumber[j];k++) AcceptRes[j][k] = 1;
   }      
  VarPos = new int[numberresidues];
  NResiduesAfterDEE = 0;
  DissComponents = (int*) 0;
  ActiveContacts = (int*) 0;
}


RotProtein::~RotProtein()  
{  
  int j;
  delete[]  RotNumber;

 if(order != (int*) 0) delete[] order;

  for(j=0;j<ncells;j++) delete[] Psi[ProteinContacts[0][j]][ProteinContacts[1][j]]; 


  for(j=0;j<numberresidues;j++) 
    {    
      delete[] Psi[j];   
      if (j<NResiduesAfterDEE) delete[] FinalRot[j];
      if ( Matrix != (unsigned int**) 0 && Matrix[j] != (unsigned int*)0)   delete[] Matrix[j]; 
      delete[] Ls[j];     
     }
  
  delete[] Psi;   
  if (Matrix != (unsigned int**)0)  delete[] Matrix;
 
  delete[] Ls;
  delete[] FinalRot;
 

  delete[] ProteinContacts[0];
  delete[] ProteinContacts[1];
  delete[] ProteinContacts;
  delete[] Fixed;
  delete[] NFinalRot;
  delete[] RemainResPositions;
  for(j=0;j<numberresidues;j++) delete[]  AcceptRes[j];
  delete[]  AcceptRes;
  delete[] VarPos;

  if(DissComponents != (int *)0) delete[] DissComponents;
  if(ActiveContacts != (int *)0) delete[] ActiveContacts;
 
} 

unsigned* RotProtein::FindBestSol(InferenceClass* RotamerInference)
{
 unsigned* BestSol;
 int i;

 BestSol = new unsigned[RotamerInference->num_nodes];
 for (i=0;i<RotamerInference->num_nodes;i++) BestSol[i] = 0;
 RotamerInference->FindBest(BestSol);
 for (i=0;i<RotamerInference->num_nodes;i++) cout<<BestSol[i]<<" ";
 cout<<endl;
 return BestSol;

}


void RotProtein::BestReducedLoopySolution(unsigned int *Card,double temperature,int MaxIter)
{
  double** Lambda = 0; //This value is used only by the POTT model. 
  unsigned *LargeSol,*BestSol;
  LargeSol = new unsigned[numberresidues];
  
  SimplifyProteinFunction();
  InferenceClass RotamerInference(AT_LOOPY,GENERAL,NResiduesAfterDEE,RedMatrix,Card,RedLs,RedPsi,Lambda,temperature);
  RotamerInference.SetLoopy(MaxIter,MAX,SEQUENTIAL);
  RotamerInference.CreateAlgorithm();
  RotamerInference.MakeInferenceAlgorithm();  
  BestSol = FindBestSol(&RotamerInference);
  FindEnlargedCodification(LargeSol,BestSol);
  DeleteSimplified();

  delete[] LargeSol;
  delete[] BestSol;
 }



void RotProtein::BestReducedLoopyMaxConfigurations(unsigned int *Card,double temperature,int MaxIter, unsigned** bestconf, double* energies, int numberconf, int* finalnumberconf)
{
  double** Lambda = 0; //This value is used only by the POTT model. 
  

  SimplifyProteinFunction();
  InferenceClass* RotamerInference = new  InferenceClass(AT_LOOPY,GENERAL,NResiduesAfterDEE,RedMatrix,Card,RedLs,RedPsi,Lambda,temperature);
  //RotamerInference->LoopyPropagation(MaxIter,MAX,SEQUENTIAL);
  int aux  =  RotamerInference->MaxConfigurationsLoopy(bestconf, energies, RotamerInference, numberconf,MaxIter);
  *finalnumberconf = aux; 
  //FindEnlargedCodification(LargeSol,BestSol);
 
  DeleteSimplified();
  delete  RotamerInference;
 }


void RotProtein::BestLoopySolutionMaxConfigurations(unsigned int *Card,double temperature,int MaxIter, unsigned** bestconf, double* energies, int numberconf, int* finalnumberconf)
{
  double** Lambda = 0; //This value is used only by the POTT model. 
  FillAdjacencyMatrix(); //The contact matrix is filled from the pairs of neighbors proteins
  //PrintAdjacencyMatrix();

  InferenceClass*  RotamerInference  = new  InferenceClass(AT_LOOPY,GENERAL, numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);
  //RotamerInference = new InferenceClass(AT_LOOPY,GENERAL, numberresidues ,Matrix,Card,Ls,Psi,Lambda,temperature);
  int aux  =  RotamerInference->MaxConfigurationsLoopy(bestconf, energies, RotamerInference, 10,MaxIter);
  *finalnumberconf = aux; 
  delete RotamerInference;
 }



void RotProtein::BestLoopySolution(unsigned int *Card,double temperature,int MaxIter)
{
  double** Lambda = 0; //This value is used only by the POTT model. 
  unsigned *BestSol;

 
  FillAdjacencyMatrix(); //The contact matrix is filled from the pairs of neighbors proteins
  //PrintAdjacencyMatrix();

  InferenceClass RotamerInference(AT_LOOPY,GENERAL, numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);
  //RotamerInference = new InferenceClass(AT_LOOPY,GENERAL, numberresidues ,Matrix,Card,Ls,Psi,Lambda,temperature);
  RotamerInference.SetLoopy(MaxIter,MAX,SEQUENTIAL);
  RotamerInference.CreateAlgorithm();
  RotamerInference.MakeInferenceAlgorithm();

  BestSol = FindBestSol(&RotamerInference);
  delete[] BestSol;
  //delete RotamerInference;
 }


void RotProtein::BestGBPSolution(unsigned int *Card,double temperature,int MaxIter)
{
  int i,j,k,auxpos,numRegs;
  int* regionsizes;
  int** AllInitRegions;
  unsigned *BestSol;
 

  double** Lambda = 0; //This value is used only by the POTT model.
  FillAdjacencyMatrix();

  InferenceClass RotamerInference(AT_GBP,GENERAL,numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);

  cout<<"Inference initialized "<<endl;

  //RotamerInference = new InferenceClass(AT_GBP,GENERAL,numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);

  MaxSubgraph = new maximalsubgraph(numberresidues,Matrix,15,5000,order);
  MaxSubgraph->FindAllCliques();
    
  numRegs = MaxSubgraph->NumberCliques;
  regionsizes = new int[numRegs];
  AllInitRegions = new int*[numRegs];

  cout<<"Cliques found "<<numRegs<<endl;

  for(i=0;i<numRegs;i++)   //The set of regions is constructed
   {
     regionsizes[i] =  MaxSubgraph->CliquesSizes[i];
     AllInitRegions[i] = new int[regionsizes[i]];
     for(j=0;j<regionsizes[i];j++) AllInitRegions[i][j] = MaxSubgraph->ListCliques[i]->vars[j];
     
     for(j=0;j<regionsizes[i]-1;j++)  //The cliques are ordered, this is needed by inference
       for(k=j+1;k<regionsizes[i];k++)
	 { 
           if(AllInitRegions[i][j]>AllInitRegions[i][k])
             {
	       auxpos = AllInitRegions[i][j];
               AllInitRegions[i][j] = AllInitRegions[i][k];
               AllInitRegions[i][k] = auxpos;
             }
         }
     
   } 
 
  RotamerInference.SetGBP(MaxIter,MAX,0.5,1,numRegs,regionsizes,AllInitRegions);
  RotamerInference.CreateAlgorithm();
  RotamerInference.MakeInferenceAlgorithm();
 
  BestSol  = FindBestSol(&RotamerInference); 

  delete[] BestSol;
  delete MaxSubgraph;
  delete[] regionsizes;
  for(i=0;i<numRegs;i++)  delete[] AllInitRegions;
  //delete RotamerInference;
 }




void RotProtein::BestReducedGBPSolution(unsigned int *Card,double temperature,int MaxIter)
{
  int i,j,k,auxpos,numRegs;
  int* regionsizes;
  int** AllInitRegions;
  double** Lambda = 0; //This value is used only by the POTT model.
 

  unsigned *LargeSol,*BestSol;
  LargeSol = new unsigned[NResiduesAfterDEE];

 
  SimplifyProteinFunction();

  InferenceClass RotamerInference(AT_GBP,GENERAL,NResiduesAfterDEE,RedMatrix,Card,RedLs,RedPsi,Lambda,temperature);

  //cout<<"Inference initialized "<<endl;

  //RotamerInference = new InferenceClass(AT_GBP,GENERAL,numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);

  MaxSubgraph = new maximalsubgraph(NResiduesAfterDEE,RedMatrix,15,5000,Redorder);
  MaxSubgraph->FindAllCliques();
  numRegs = MaxSubgraph->NumberCliques;
  regionsizes = new int[numRegs];
  AllInitRegions = new int*[numRegs];

  cout<<"Cliques found "<<numRegs<<endl;

  for(i=0;i<numRegs;i++)   //The set of regions is constructed
   {
     regionsizes[i] =  MaxSubgraph->CliquesSizes[i];
     AllInitRegions[i] = new int[regionsizes[i]];
    for(j=0;j<regionsizes[i];j++) AllInitRegions[i][j] = MaxSubgraph->ListCliques[i]->vars[j];
     
     for(j=0;j<regionsizes[i]-1;j++)  //The cliques are ordered, this is needed by inference
       for(k=j+1;k<regionsizes[i];k++)
	 { 
           if(AllInitRegions[i][j]>AllInitRegions[i][k])
             {
	       auxpos = AllInitRegions[i][j];
               AllInitRegions[i][j] = AllInitRegions[i][k];
               AllInitRegions[i][k] = auxpos;
             }
         }
     
   } 
 
  RotamerInference.SetGBP(MaxIter,MAX,0.5,1,numRegs,regionsizes,AllInitRegions);
  RotamerInference.CreateAlgorithm();
  RotamerInference.MakeInferenceAlgorithm();
  BestSol = FindBestSol(&RotamerInference);
  CalculateReducedEnergy(BestSol);

  FindEnlargedCodification(LargeSol,BestSol);
  DeleteSimplified();

  delete[] LargeSol;
  delete[] BestSol;
  delete MaxSubgraph;
  delete[] regionsizes; 
  for(i=0;i<numRegs;i++)  delete[] AllInitRegions;
  
 }



void RotProtein::FindDisconnectedComponents()
{
  int j,k,BothFixed;
  int valmin,valmax;

   
 DissComponents = new int[numberresidues];
 for(k=0;k<numberresidues;k++) DissComponents[k] = -1;

 ncomp = 0;
 for(j=0;j<ncells;j++)
   {
     //  cout<<"Before "<<j<<"  "<<ProteinContacts[0][j]<<"  "<<ProteinContacts[1][j]<<"  "<<Fixed[ProteinContacts[0][j]]<<"  "<<Fixed[ProteinContacts[1][j]]<<"  "<<DissComponents[ProteinContacts[0][j]]<<"  "<<DissComponents[ProteinContacts[1][j]]<<endl;  

    BothFixed =  (Fixed[ProteinContacts[0][j]] != -1 && Fixed[ProteinContacts[1][j]] != -1);
 
    if(!BothFixed)
     {
     if (DissComponents[ProteinContacts[0][j]] == -1 &&   DissComponents[ProteinContacts[1][j]] == -1)
       {
         if(Fixed[ProteinContacts[0][j]] == -1) DissComponents[ProteinContacts[0][j]] = ncomp;
         if(Fixed[ProteinContacts[1][j]] == -1) DissComponents[ProteinContacts[1][j]] = ncomp;
         ncomp++;
        }
     else if (DissComponents[ProteinContacts[0][j]] == -1 &&  (Fixed[ProteinContacts[0][j]] == -1) &&  DissComponents[ProteinContacts[1][j]] != -1)
       {
         DissComponents[ProteinContacts[0][j]] = DissComponents[ProteinContacts[1][j]];
       }
     else if (DissComponents[ProteinContacts[0][j]] != -1 &&  (Fixed[ProteinContacts[1][j]] == -1) &&  DissComponents[ProteinContacts[1][j]] == -1)
       {
         DissComponents[ProteinContacts[1][j]] = DissComponents[ProteinContacts[0][j]];
       }
     else if (DissComponents[ProteinContacts[0][j]] != -1 &&   DissComponents[ProteinContacts[1][j]] != -1  &&   (DissComponents[ProteinContacts[0][j]] !=   DissComponents[ProteinContacts[1][j]]) )
	{
          if(DissComponents[ProteinContacts[0][j]] <  DissComponents[ProteinContacts[1][j]])
           {
             valmin = DissComponents[ProteinContacts[0][j]];
             valmax = DissComponents[ProteinContacts[1][j]];
           }
          else
           {
             valmin = DissComponents[ProteinContacts[1][j]];
             valmax = DissComponents[ProteinContacts[0][j]];
           }
          for(k=0;k<numberresidues;k++) if(DissComponents[k]==valmax) DissComponents[k]=valmin;
	        
	 
   }

   }
    //cout<<"After "<<j<<"  "<<ProteinContacts[0][j]<<"  "<<ProteinContacts[1][j]<<"  "<<Fixed[ProteinContacts[0][j]]<<"  "<<Fixed[ProteinContacts[1][j]]<<"  "<<DissComponents[ProteinContacts[0][j]]<<"  "<<DissComponents[ProteinContacts[1][j]]<<endl;  
 }          
                
 // for(k=0;k<numberresidues;k++) cout<<DissComponents[k]<<" ";
 //cout<<endl;               
      
}




double RotProtein::CalculateReducedEnergy(unsigned* assign)
  {
   double epsilon = 1e-200;
   double E;
   int j,k;

   E = FixedValue;
   //cout<<"Initial Value "<<E<<endl;

   for(j=0;j<NResiduesAfterDEE;j++)
    {
     E = E - log(RedLs[j][assign[j]] + epsilon);
     //cout<<j<<" val "<<RedLs[j][assign[j]]<<"  "<<" 1-   "<<E<<endl;
     for(k=j+1;k<NResiduesAfterDEE;k++)
      {
     if(RedPsi[j][k] != (double*)0)  
        {
	  // cout<<"-------------------- "<<k<<"  "<<RedPsi[j][k][RotNumber[k]*assign[j]+assign[k]]<<endl;
         E = E - log(RedPsi[j][k][RotNumber[k]*assign[j]+assign[k]]+epsilon);
          
        }  
      }
     //cout<<" 2-  "<<E<<endl;


    }    
   return E;
  }





double RotProtein::CalculateEnergy(unsigned* assign)
  {
   double epsilon = 1e-200;
   double E;
   int j,k;

   E = 0;
  
   for(j=0;j<numberresidues;j++)
    {
     E = E - log(Ls[j][assign[j]] + epsilon);
     cout<<j<<" val "<<Ls[j][assign[j]]<<"  "<<" 1-   "<<E<<endl;
     for(k=j+1;k<numberresidues;k++)
      {
     if(Psi[j][k] != (double*)0)  
        {
	  cout<<"-------------------- "<<k<<"  "<<Psi[j][k][RotNumber[k]*assign[j]+assign[k]]<<endl;
         E = E - log(Psi[j][k][RotNumber[k]*assign[j]+assign[k]]+epsilon);
          
        }  
      }
     cout<<" 2-  "<<E<<endl;


    }    
   return E;
  }

double RotProtein::CalculateEnergySCP(unsigned* assign)
  {   
   double E;
   int j,k;

   E = 0;
  
   for(j=0;j<numberresidues;j++)
    {
     E = E + Ls[j][assign[j]];
     //cout<<j<<" val "<<Ls[j][assign[j]]<<"  "<<" 1-   "<<E<<endl;
     for(k=j+1;k<numberresidues;k++)
      {
     if(Psi[j][k] != (double*)0)  
        {
	  // cout<<"-------------------- "<<k<<"  "<<Psi[j][k][RotNumber[k]*assign[j]+assign[k]]<<endl;
         E = E + Psi[j][k][RotNumber[k]*assign[j]+assign[k]];
          
        }  
      }
     //cout<<" 2-  "<<E<<endl;


    }    
   return E;
  }


void  RotProtein::FillFixed(unsigned* assign)
  {
    int j;
    for(j=0;j<NResiduesAfterDEE;j++) Fixed[RemainResPositions[j]] = FinalRot[j][assign[j]];
  }



void  RotProtein::FillFixedComp(unsigned* assign, int numbercomp, int* indexcomponents)
{
    int j;  for(j=0;j<numbercomp;j++) Fixed[RemainResPositions[indexcomponents[j]]] = FinalRot[indexcomponents[j]][assign[j]];
}  



void  RotProtein::FindEnlargedCodification(unsigned int* largeassign, unsigned int* smallassign)
{
    int j;

    //for(j=0;j<NResiduesAfterDEE;j++)  cout<<" "<<RemainResPositions[j];
    //cout<<endl; 
   for(j=0;j<NResiduesAfterDEE;j++) Fixed[RemainResPositions[j]] = FinalRot[j][smallassign[j]];
    for(j=0;j<numberresidues;j++) 
      {
         largeassign[j] =  Fixed[j];
	 // cout<<" "<<largeassign[j];
      }
    //cout<<endl;

}  



void  RotProtein::FindReducedCodification(unsigned* largeassign, unsigned* assign)
{
  int j,k;

  for(j=0;j<numberresidues;j++) Fixed[j] = largeassign[j];
  for(j=0;j<NResiduesAfterDEE;j++) 
   { 
     //cout<<NFinalRot[j]<<" ";
    for(k=0;k<NFinalRot[j];k++) 
       { 
         //cout<<"pos "<<j<<" val "<<k<<" "<<RemainResPositions[j]<<"  "<<FinalRot[j][k]<<endl;
         if(FinalRot[j][k] == Fixed[RemainResPositions[j]]) 
            {
             assign[j] = k;
             //cout<<"pos "<<j<<" val "<<k<<endl;
            }
       }       
   }
  //cout<<endl;
}  

int RotProtein::Best1Neighborhood(unsigned* assign, int* nval) //Search in the neighborhood of assign the best solution
  {
   double BestE,NewE;
   int j,k,BestRes;
   int BestVal,OldVal;

   (*nval) = 0;
   BestRes = -1;
   BestVal = -1;

   BestE = CalculateEnergyWithDEE(assign);
   //cout<<"InitVal "<<BestE<<endl;
   //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
   //cout<<endl;
    for(j=0;j<NResiduesAfterDEE;j++) 
      for(k=0;k<NFinalRot[j];k++) 
        {
          OldVal  =   Fixed[RemainResPositions[j]];
          if (FinalRot[j][k] != OldVal)
	    {
              Fixed[RemainResPositions[j]] =  FinalRot[j][k];
              //for(int l=0;l<numberresidues;l++) cout<<Fixed[l]<<" ";
              NewE =  EvaluateFromFixed();
              (*nval)++;    
	      //cout<<" "<<NewE<<endl;
              if(NewE<BestE)
	       {
	        BestRes = j;
                BestVal = k;
                BestE = NewE;
               } 
              Fixed[RemainResPositions[j]] = OldVal;      
             }             
	}
    if(BestRes>-1)  
     {
       assign[BestRes] = BestVal;
       //cout<<BestE<<" "<<BestRes<<" "<<BestVal<<endl;
     } 
    //for(j=0;j<NResiduesAfterDEE;j++) cout<<assign[j]<<" ";
    //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
    //cout<<endl;
    return (BestRes>-1);       
  }




int RotProtein::Best2Neighborhood(unsigned* assign, int* nval) //Search in the neighborhood of assign the best solution
  {
   double BestE,NewE;
   int j,k,l,BestRes,var1,var2;
   int BestVal1,BestVal2,OldVal1,OldVal2;
   (*nval) = 0;
   BestRes = -1;
 

   BestE = CalculateEnergyWithDEE(assign);
   //cout<<"InitVal at two "<<BestE<<endl;

   //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
   //cout<<endl;
    for(j=0;j<NActiveContacts;j++) 
     {
       var1 = VarPos[ProteinContacts[0][ActiveContacts[j]]];
       var2 = VarPos[ProteinContacts[1][ActiveContacts[j]]];
       for(k=0;k<NFinalRot[var1];k++)
         for(l=0;l<NFinalRot[var2];l++)
          {
            OldVal1  =   Fixed[RemainResPositions[var1]];
            OldVal2  =   Fixed[RemainResPositions[var2]];
          
            if (FinalRot[var1][k] != OldVal1 && FinalRot[var2][l] != OldVal2)
	     {
              Fixed[RemainResPositions[var1]] =  FinalRot[var1][k];
              Fixed[RemainResPositions[var2]] =  FinalRot[var2][l];
              //for(int l=0;l<numberresidues;l++) cout<<Fixed[l]<<" ";
              NewE =  EvaluateFromFixed();    
              (*nval)++;
	      //cout<<" "<<NewE<<endl;
              if(NewE<BestE)
	       {
	        BestRes = j;
                BestVal1 = k;
                BestVal2 = l;
                BestE = NewE;
               } 
              Fixed[RemainResPositions[var1]] = OldVal1;
              Fixed[RemainResPositions[var2]] = OldVal2;
             }             
  	}
     }
      if(BestRes>-1)
       {
        var1 = VarPos[ProteinContacts[0][ActiveContacts[BestRes]]];
        var2 = VarPos[ProteinContacts[1][ActiveContacts[BestRes]]];  
        assign[var1] = BestVal1;
        assign[var2] = BestVal2;
        //cout<<BestE<<" "<<BestVal1<<" "<<BestVal2<<endl; 
      }   
    //for(j=0;j<NResiduesAfterDEE;j++) cout<<assign[j]<<" ";
    //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
    //cout<<endl;
    return (BestRes>-1);       
  }






int RotProtein::Best3Neighborhood(unsigned* assign, int* nval) //Search in the neighborhood of assign the best solution
  {
   double BestE,NewE;
   int i,j,k,l,m,BestRes,var1,var2,var3,var4;
   int BestVal1,BestVal2,BestVal3,OldVal1,OldVal2,OldVal3,Bestvar1,Bestvar2,Bestvar3;

   (*nval) = 0;
   BestRes = -1;
 

   BestE = CalculateEnergyWithDEE(assign);
   //cout<<"InitVal at three "<<BestE<<endl;

   //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
   //cout<<endl;
    for(i=0;i<NActiveContacts-1;i++) 
     for(j=i+1;j<NActiveContacts;j++) 
     {
       var1 = VarPos[ProteinContacts[0][ActiveContacts[j]]];
       var2 = VarPos[ProteinContacts[1][ActiveContacts[j]]];
       var3 = VarPos[ProteinContacts[0][ActiveContacts[i]]];
       var4 = VarPos[ProteinContacts[1][ActiveContacts[i]]]; 
       //cout<<" var 1: "<<var1<<" var 2: "<<var2<<" var 3: "<<var3<<" var 4: "<<var4<<endl;
       if(var1==var3 || var2==var3 || var1==var4 || var2==var4 ) 
	 {   
          if(var2==var3 || var1==var3) var3 = var4;
          for(k=0;k<NFinalRot[var1];k++)
           for(l=0;l<NFinalRot[var2];l++)
            for(m=0;m<NFinalRot[var3];m++)
          {
            OldVal1  =   Fixed[RemainResPositions[var1]];
            OldVal2  =   Fixed[RemainResPositions[var2]];
            OldVal3  =   Fixed[RemainResPositions[var3]];
          
            if (FinalRot[var1][k] != OldVal1 && FinalRot[var2][l] != OldVal2 && FinalRot[var3][m] != OldVal3)
	     {
              Fixed[RemainResPositions[var1]] =  FinalRot[var1][k];
              Fixed[RemainResPositions[var2]] =  FinalRot[var2][l];
              Fixed[RemainResPositions[var3]] =  FinalRot[var3][m];
              //for(int l=0;l<numberresidues;l++) cout<<Fixed[l]<<" ";
              NewE =  EvaluateFromFixed();   
              (*nval)++; 
	      //cout<<" "<<NewE<<endl;
              if(NewE<BestE)
	       {
		BestRes = 1;
	        Bestvar1 = var1;
                Bestvar2 = var2;
                Bestvar3 = var3;
                BestVal1 = k;
                BestVal2 = l;
                BestVal3 = m;
                BestE = NewE;
               } 
              Fixed[RemainResPositions[var1]] = OldVal1;
              Fixed[RemainResPositions[var2]] = OldVal2;
              Fixed[RemainResPositions[var3]] = OldVal3;
             }             
  	}
     }
    } 
      if(BestRes>-1)
       {
        assign[Bestvar1] = BestVal1;
        assign[Bestvar2] = BestVal2;
        assign[Bestvar3] = BestVal3;
        //cout<<BestE<<" "<<BestVal1<<" "<<BestVal2<<" "<<BestVal3<<endl; 
      }   
    //for(j=0;j<NResiduesAfterDEE;j++) cout<<assign[j]<<" ";
    //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
    //cout<<endl;
    return (BestRes>-1);       
  }




int RotProtein::RandomBest1Neighborhood(unsigned* assign, int numbertries, int* nval) //Search in the neighborhood of assign the best solution
  {
   double BestE,NewE;
   int i,j,k,BestRes;
   int BestVal,OldVal;
   time_t ltime;
   struct tm *gmt;
  
  (*nval) = 0;
   
   BestRes = -1;
   BestVal = -1;

   BestE = CalculateEnergyWithDEE(assign);
   //cout<<"InitVal "<<BestE<<endl;
   //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
   //cout<<endl;

    for(i=0;i<numbertries;i++)
      {
         j = randomint(NResiduesAfterDEE);
         k = randomint(NFinalRot[j]);
         {
          OldVal  =   Fixed[RemainResPositions[j]];
          if (FinalRot[j][k] != OldVal)
	    {
              Fixed[RemainResPositions[j]] =  FinalRot[j][k];
              NewE =  EvaluateFromFixed();
              (*nval)++;
    
	      if(NewE<BestE)
	       {
	        BestRes = j;
                BestVal = k;
                BestE = NewE;
                assign[BestRes] = BestVal;
                //cout<<BestE<<" 1  "<<BestRes<<" "<<BestVal<<endl;
                time(&ltime);
                gmt = localtime(&ltime);
 	        cout<<" 1  "<<(*nval)<<" "<<BestE<<" "<<gmt->tm_mday<<" "<<gmt->tm_hour<<" "<<gmt->tm_min<<" "<<gmt->tm_sec<<endl;
               } 
              else Fixed[RemainResPositions[j]] = OldVal;      
             }             
	 }
      }
 
     //for(j=0;j<NResiduesAfterDEE;j++) cout<<assign[j]<<" ";
    //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
    //cout<<endl;
    return (BestRes>-1);       
  }




int RotProtein::RandomBest2Neighborhood(unsigned* assign, int numbertries, int* nval) //Search in the neighborhood of assign the best solution
  {
   double BestE,NewE;
   int i,j,k,l,BestRes,var1,var2;
   int BestVal1,BestVal2,OldVal1,OldVal2;

   time_t ltime;
   struct tm *gmt;

   (*nval) = 0;

   BestRes = -1;
 

   BestE = CalculateEnergyWithDEE(assign);    
    
    for(i=0;i<numbertries;i++) 
     {
       j= randomint(NActiveContacts);
       var1 = VarPos[ProteinContacts[0][ActiveContacts[j]]];
       var2 = VarPos[ProteinContacts[1][ActiveContacts[j]]];
       k = randomint(NFinalRot[var1]);
       l = randomint(NFinalRot[var2]);
          {
            OldVal1  =   Fixed[RemainResPositions[var1]];
            OldVal2  =   Fixed[RemainResPositions[var2]];
          
            if (FinalRot[var1][k] != OldVal1 && FinalRot[var2][l] != OldVal2)
	     {
              Fixed[RemainResPositions[var1]] =  FinalRot[var1][k];
              Fixed[RemainResPositions[var2]] =  FinalRot[var2][l];           
              NewE =  EvaluateFromFixed();    	
              (*nval)++; 
              if(NewE<BestE)
	       {
	        BestRes = j;
                BestVal1 = k;
                BestVal2 = l;
                BestE = NewE;
                var1 = VarPos[ProteinContacts[0][ActiveContacts[BestRes]]];
                var2 = VarPos[ProteinContacts[1][ActiveContacts[BestRes]]];  
                assign[var1] = BestVal1;
                assign[var2] = BestVal2;
                time(&ltime);
                gmt = localtime(&ltime);
 	        cout<<" 2  "<<(*nval)<<" "<<BestE<<" "<<gmt->tm_mday<<" "<<gmt->tm_hour<<" "<<gmt->tm_min<<" "<<gmt->tm_sec<<endl;
		//cout<<BestE<<" 2  "<<BestRes<<endl;
               } 
              else 
               {
                Fixed[RemainResPositions[var1]] = OldVal1;
                Fixed[RemainResPositions[var2]] = OldVal2;
               }
             }             
  	}
     }
       
    return (BestRes>-1);       
  }






int RotProtein::RandomBest3Neighborhood(unsigned* assign, int numbertries, int* nval) //Search in the neighborhood of assign the best solution
  {
   double BestE,NewE;
   int ii,i,j,k,l,m,BestRes,var1,var2,var3,var4;
   int BestVal1,BestVal2,BestVal3,OldVal1,OldVal2,OldVal3,Bestvar1,Bestvar2,Bestvar3;

   time_t ltime;
   struct tm *gmt;

   (*nval) = 0;
 
   BestRes = -1;
 

   BestE = CalculateEnergyWithDEE(assign);
 
   cout<<BestE<<"   "<<NActiveContacts<<" In here "<<endl;
   if(NActiveContacts>1)  // This most be indeed the case
   {
   for(ii=0;ii<numbertries;ii++) 
   {
    i = randomint(NActiveContacts);
    j = randomint(NActiveContacts);
    while(i==j)
      {
        i = randomint(NActiveContacts);
        j = randomint(NActiveContacts);
      } 
   
     {
       var1 = VarPos[ProteinContacts[0][ActiveContacts[j]]];
       var2 = VarPos[ProteinContacts[1][ActiveContacts[j]]];
       var3 = VarPos[ProteinContacts[0][ActiveContacts[i]]];
       var4 = VarPos[ProteinContacts[1][ActiveContacts[i]]]; 
     
       if(var1==var3 || var2==var3 || var1==var4 || var2==var4 ) 
	 {   
          if(var2==var3 || var1==var3) var3 = var4;
          k = randomint(NFinalRot[var1]);
          l = randomint(NFinalRot[var2]);
          m = randomint(NFinalRot[var3]);
          {
            OldVal1  =   Fixed[RemainResPositions[var1]];
            OldVal2  =   Fixed[RemainResPositions[var2]];
            OldVal3  =   Fixed[RemainResPositions[var3]];
          
            if (FinalRot[var1][k] != OldVal1 && FinalRot[var2][l] != OldVal2 && FinalRot[var3][m] != OldVal3)
	     {
              Fixed[RemainResPositions[var1]] =  FinalRot[var1][k];
              Fixed[RemainResPositions[var2]] =  FinalRot[var2][l];
              Fixed[RemainResPositions[var3]] =  FinalRot[var3][m];
              NewE =  EvaluateFromFixed();
              (*nval)++;    
              if(NewE<BestE)
	       {
		BestRes = 1;
	        Bestvar1 = var1;
                Bestvar2 = var2;
                Bestvar3 = var3;
                BestVal1 = k;
                BestVal2 = l;
                BestVal3 = m;
                BestE = NewE;
                assign[Bestvar1] = BestVal1;
                assign[Bestvar2] = BestVal2;
                assign[Bestvar3] = BestVal3;
                //cout<<BestE<<" 3  "<<BestRes<<endl;
                time(&ltime);
                gmt = localtime(&ltime);
 	        cout<<" 3  "<<(*nval)<<" "<<BestE<<" "<<gmt->tm_mday<<" "<<gmt->tm_hour<<" "<<gmt->tm_min<<" "<<gmt->tm_sec<<endl;
               } 
              else
	       {
                Fixed[RemainResPositions[var1]] = OldVal1;
                Fixed[RemainResPositions[var2]] = OldVal2;
                Fixed[RemainResPositions[var3]] = OldVal3;
               }
             }             
  	}
     }
    }     
   }
   }    
    return (BestRes>-1);       
  }


int RotProtein::SearchNeighborhood(int maxmoves,unsigned* assign, int* nval)
{
  int i,a,improved,GoOn;
  int auxnval;
  *nval = 0;


  i = 0;
  GoOn = 0;
  a = 0;
  
 while(a==0 || (GoOn && i<maxmoves))
  {
   GoOn = 0;
   improved = 1;
   a++;
  
   while(i<maxmoves && improved)
    {
      improved = Best1Neighborhood(assign,&auxnval);
      (*nval) = (*nval) +  (auxnval);
      // cout<<"Pass 1 "<<i<<*nval<<" -- > "<<auxnval<<endl;
      if (improved) 
       {
         GoOn = 1;
         i++;
	 //    cout<<"At step "<<a<<" 1 Neigh improved"<<endl;
       }  
    }
  
   /* 
  improved = 1;
   while(i<maxmoves && improved)
    {
      improved = Best2Neighborhood(assign,&auxnval);
       (*nval) = (*nval) +  (auxnval);
       cout<<"Pass 2 "<<i<<*nval<<" -- > "<<auxnval<<endl;
      if (improved) 
       {
         GoOn = 1;
	 // cout<<"At step "<<a<<" 2 Neigh improved"<<endl;
         i++;
       } 
    }
   // Only for instances small enough
   improved = 1;

   while(i<maxmoves && improved)
    {
      improved = Best3Neighborhood(assign,&auxnval);
       (*nval) = (*nval) +  (auxnval);      
       cout<<"Pass 3 "<<i<<*nval<<" -- > "<<auxnval<<endl;
      if (improved) 
       {
         GoOn = 1;
         //cout<<"At step "<<a<<" 2 Neigh improved"<<endl;
         i++;
       } 
    }
   */
  
  }  
  return i;
}



int RotProtein::SearchRandomNeighborhood(int maxmoves,unsigned* assign, int numbertries, int* nval)
{
  int i,a,improved,GoOn;
  i = 0;
  GoOn = 0;
  a = 0;
  *nval = 0;
  int auxnval;

  
 while(a==0 || (GoOn && i<maxmoves))
  {
   GoOn = 0;
   improved = 1;
   a++;
   /*
   while(i<maxmoves && improved)
    {
      improved = RandomBest1Neighborhood(assign,numbertries,&auxnval);
       (*nval) = (*nval) +  (auxnval);
       // cout<<i<<" "<<*nval<<endl;    
      if (improved) 
       {
         GoOn = 1;
         i++;
	 //   cout<<"At step "<<a<<" 1 Neigh improved"<<endl;
       }  
    }
  
  
  improved = 1;
   while(i<maxmoves && improved)
    {
      improved = RandomBest2Neighborhood(assign,numbertries,&auxnval);
       (*nval) = (*nval) +  (auxnval);
      if (improved) 
       {
         GoOn = 1;
	 // cout<<"At step "<<a<<" 2 Neigh improved"<<endl;
         i++;
       } 
    }
   // Only for instances small enough
   
  */
  improved = 1;

   while(i<maxmoves && improved)
    {
      improved = RandomBest3Neighborhood(assign,numbertries,&auxnval);
       (*nval) = (*nval) +  (auxnval);
      if (improved) 
       {
         GoOn = 1;
         //cout<<"At step "<<a<<" 2 Neigh improved"<<endl;
         i++;
       } 
    }
  
   
  }  
  return i;
}




// This function finds the reduced Matrix (RedMatrix), the Reduced Psi (RedPsi) and the Reduced Ls vector(RedLs);
void RotProtein::SimplifyProteinFunction()
{
  int i,j,k,l,res_0,res_1; 
 double  epsilon = 1e-200;
 // unsigned testvector[54] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  FixedValue  = 0;
  
  RedLs = new double*[NResiduesAfterDEE];
  RedPsi = new double**[NResiduesAfterDEE];
  RedMatrix = new unsigned int*[NResiduesAfterDEE];
  Redorder = new int[NResiduesAfterDEE];

  for(j=0;j<numberresidues;j++)  
   {
     if(Fixed[j] != -1) FixedValue -= log(Ls[j][Fixed[j]] + epsilon); 
   }  

  for(i=0;i<NResiduesAfterDEE;i++)  
   {
     Redorder[i] = i+1;
     RedLs[i] =  new double[NFinalRot[i]]; 
     //cout<<i<<" "<<RemainResPositions[i]<<" "<<RemainResPositions[i]<<endl;
     for(j=0;j<NFinalRot[i];j++)  
      {
          RedLs[i][j] = Ls[RemainResPositions[i]][FinalRot[i][j]]; 
	  //   cout<<j<<" "<<RemainResPositions[i]<<" "<<FinalRot[i][j]<<"  "<<RedLs[i][j]<<endl;
      }
        
     RedPsi[i] =  new double*[NResiduesAfterDEE];
     RedMatrix[i] = new unsigned int[NResiduesAfterDEE];
     for(j=0;j<NResiduesAfterDEE;j++)  
       { 
        if(i==j) RedMatrix[i][j] = 1;
         else RedMatrix[i][j] = 0;
	RedPsi[i][j] = (double*)0;
       }
   }


   for(j=0;j<ncells;j++)
    {
     res_0 = VarPos[ProteinContacts[0][j]];
     res_1 = VarPos[ProteinContacts[1][j]];
     //cout<<j<<" "<<ProteinContacts[0][j]<<" "<<ProteinContacts[1][j]<<" "<<Fixed[ProteinContacts[0][j]]<<" "<<Fixed[ProteinContacts[1][j]]<<"  "<<res_0<<"  "<<res_1<<" "<< RotNumber[ProteinContacts[0][j]]<<" "<< RotNumber[ProteinContacts[1][j]]<<endl;

     if( res_0 != -1 &&  res_1 != -1)
      {       
        RedMatrix[res_0][res_1] = 1;
        RedMatrix[res_1][res_0] = 1;
	//cout<<j<<" "<<ProteinContacts[0][j]<<" "<<ProteinContacts[1][j]<<" "<<Fixed[ProteinContacts[0][j]]<<" "<<Fixed[ProteinContacts[1][j]]<<"  "<<res_0<<"  "<<res_1<<" "<< RotNumber[ProteinContacts[0][j]]<<" "<< RotNumber[ProteinContacts[1][j]]<<endl;
	RedPsi[res_0][res_1] = new double[NFinalRot[res_0]*NFinalRot[res_1]];
       
        for(k=0;k<NFinalRot[res_0];k++)
           for(l=0;l<NFinalRot[res_1];l++)
	     { 
               // cout<<"l "<<l<<" k*NFinalRot[res_1]+l "<<  k*NFinalRot[res_1]+l<<"  "<<FinalRot[res_0][k]<<"  "<<FinalRot[res_1][l]<< " Psi "<<FinalRot[res_0][k]* RotNumber[ProteinContacts[1][j]]+ FinalRot[res_1][l]<<" "<< Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][RotNumber[ProteinContacts[1][j]]*FinalRot[res_0][k] + FinalRot[res_1][l]]<<endl;
           
              RedPsi[res_0][res_1][k*NFinalRot[res_1]+l] =  Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][FinalRot[res_0][k]* RotNumber[ProteinContacts[1][j]]+ FinalRot[res_1][l]];            
             }       
       }
     else if( res_0 == -1 &&  res_1 != -1)
       {     
          for(k=0;k<NFinalRot[res_1];k++)  RedLs[res_1][k] *= Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][Fixed[ProteinContacts[0][j]]* RotNumber[ProteinContacts[1][j]]+ FinalRot[res_1][k]]; 

       }
     else if( res_0 != -1 &&  res_1 == -1)
       {     
          for(k=0;k<NFinalRot[res_0];k++)  RedLs[res_0][k] *= Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][FinalRot[res_0][k]* RotNumber[ProteinContacts[1][j]]+ Fixed[ProteinContacts[1][j]]]; 

       }
      else if( res_0 == -1 &&  res_1 == -1)
       { 
         FixedValue -= log(Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][Fixed[ProteinContacts[0][j]]*RotNumber[ProteinContacts[1][j]]+ Fixed[ProteinContacts[1][j]]] + epsilon); 
       }

    }

   
    //CalculateReducedEnergy((unsigned*)testvector);


   /*
   CalculateReducedEnergy((unsigned*)testvector);
   FindEnlargedCodification((unsigned*)testvector,(unsigned*)testvector);
   cout<<"Test Vector "<<endl;
   for(j=0;j<numberresidues;j++)  cout<<testvector[j]<<" ";
   cout<<endl;   
   CalculateEnergy((unsigned*)testvector);
   */  
 
}


void RotProtein::DeleteSimplified()
{
  int i,j, res_0,res_1;

 delete[] Redorder;
   
  
  for(j=0;j<ncells;j++)
    {
    
     res_0 = VarPos[ProteinContacts[0][j]];
     res_1 = VarPos[ProteinContacts[1][j]];
   
     // cout<<j<<" "<<res_0<<"  "<<res_1<<" "<<endl;
   if(res_0 != -1 && res_1 != -1) 
     {
       //cout<<j<<" "<<res_0<<"  "<<res_1<<" "<<RedPsi[res_0][res_1]<<endl;  
       delete[] RedPsi[res_0][res_1];
     }

    }
 
  for(i=0;i<NResiduesAfterDEE;i++)  
   {
     delete[] RedLs[i]; 
     delete[] RedPsi[i];
     delete[] RedMatrix[i];
   } 

  delete[]  RedLs;
  delete[]  RedPsi;
  delete[]  RedMatrix; 

   
}




void RotProtein::FindActiveContacts()
{
  int j;
  ActiveContacts = new int[ncells];

  NActiveContacts = 0;
  for(j=0;j<ncells;j++) 
    if( VarPos[ProteinContacts[0][j]] != -1 &&  VarPos[ProteinContacts[1][j]] != -1)
      { 
        ActiveContacts[NActiveContacts] = j;
        NActiveContacts++; 
      }
}


double RotProtein::EvaluateFromFixed()
  {
   double epsilon = 1e-200;
   double E;
   int j,k;

      E = 0;
  
   for(j=0;j<numberresidues;j++)
    {
     E = E - log(Ls[j][Fixed[j]] + epsilon);
     //cout<<j<<" val "<<Ls[j][assign[j]]<<"  "<<" 1-   "<<E<<" ";
     for(k=j+1;k<numberresidues;k++)
      {
     if(Psi[j][k] != (double*)0)  
        {
	  E = E - log(Psi[j][k][RotNumber[k]*Fixed[j]+Fixed[k]]+epsilon);
          
        }  
      }
     //cout<<" 2-  "<<E<<endl;

    }  
   return E;


  }

double RotProtein::CalculateEnergyWithDEE(unsigned* assign)
  {
   double E;
   int j;

   for(j=0;j<NResiduesAfterDEE;j++) Fixed[RemainResPositions[j]] = FinalRot[j][assign[j]];
  
   
   //for(j=0;j<numberresidues;j++) cout<<Fixed[j]<<" ";
   //cout<<endl; 
   
   E = EvaluateFromFixed();
  
  
   return E;
  }


double RotProtein::CalculateEnergyDECOMP(unsigned* assign, int numbercomp, int* indexcomponents)
  {
   double epsilon = 1e-200;
   double E;
   int j,k;



    for(j=0;j<numbercomp;j++) Fixed[RemainResPositions[indexcomponents[j]]] = FinalRot[indexcomponents[j]][assign[j]];
  

   E = 0;
   for(j=0;j<numberresidues;j++)
    {
     if(Fixed[j]>-1)
      {
       E = E - log(Ls[j][Fixed[j]] + epsilon);
       //cout<<j<<" val "<<Ls[j][assign[j]]<<"  "<<" 1-   "<<E<<" ";
       for(k=j+1;k<numberresidues;k++)
        {
        if(Fixed[k]>-1 && Psi[j][k] != (double*)0)  
         {
	   E = E - log(Psi[j][k][RotNumber[k]*Fixed[j]+Fixed[k]]+epsilon);        
        }  
       }
      }    
    }    
   return E;
  }






void RotProtein::DEE(double alpha)
  {

   int i,j,k,l,m,n,totrem;
   totrem  = 0;

   double diff_energy, val, minval;
    double epsilon = 1e-200;
    i = 0;
    totrem = 1;
while(totrem>0 && i<100)
   {
    totrem  = 0;
    for(j=0;j<numberresidues;j++)
    {
      for(k=0;k<RotNumber[j];k++) 
       {
         l = 0;
	 while(l<RotNumber[j]) 
	   {
	   if (l != k  && AcceptRes[j][k] == 1  && AcceptRes[j][l] == 1)
	    {
             diff_energy = - log(Ls[j][k] + epsilon) + log(Ls[j][l] + epsilon);

	     //cout<<" Init diff_energy ="<<diff_energy<<endl;
    	      for(m=0;m<numberresidues;m++)
	       {
               
		 if( (j != m) && (Psi[j][m] != (double*)0) ) 
		   {
                       minval = 1000000.0;
                       for(n=0;n<RotNumber[m];n++)
		       {
                       
			  if(AcceptRes[m][n] == 1)
			   {
                             if(j<m)  val = - log(Psi[j][m][RotNumber[m]*k+n] + epsilon) + log(Psi[j][m][RotNumber[m]*l+n] + epsilon);
                             else  val = - log(Psi[j][m][RotNumber[j]*n+k] + epsilon) + log(Psi[j][m][RotNumber[j]*n+l] + epsilon);
                             if (val<minval) minval = val;
                           }
			  //cout<<" m ="<<m<<" n ="<<n<<" Psi(j,m,k)"<<Psi[j][m][RotNumber[m]*k+n]<< " Psi(j,m,l)"<< Psi[j][m][RotNumber[m]*l+n]<< " val ="<<val<<" minval ="<<minval<<endl;

                       }    
		     diff_energy += minval;                              
                   }
                        	                      
               }
             
                 if (diff_energy>(epsilon+alpha) && AcceptRes[j][k] == 1) 
                 {
		   totrem = totrem + 1;
                   AcceptRes[j][k] = 0;
		   // cout<<"tot="<<totrem<<" j ="<<j<<" k ="<<k<<" l ="<<l<<" m ="<<m<<" Accept[j][k] ="<<AcceptRes[j][k]<<" diff_energy ="<<diff_energy<<" "<<fabs(diff_energy - alpha)<<endl;
                 } 
		//cout<<"tot="<<totrem<<" j ="<<j<<" k ="<<k<<" l ="<<l<<" m ="<<m<<" Accept[j][k] ="<<AcceptRes[j][k]<<" diff_energy ="<<diff_energy<<" "<<fabs(diff_energy - alpha)<<endl;
          
		//cout<<" j ="<<j<<" k ="<<k<<" l ="<<l<<" m ="<<m<<" Accept[j][k] ="<<AcceptRes[j][k]<<" diff_energy ="<<diff_energy<<endl;
	    }
             l++;              
	  }             
      }
    }        

    //cout<<"Iter "<<i<<"removed rotamers "<<totrem<<endl;
    i++;
 } 
     
 }


void RotProtein::ApplyDEE(double alpha)
  {
    DEE(alpha);
    //SecondDEE(alpha);
    FinishDEE();
  }

void RotProtein::SecondDEE(double alpha)
  {

   int j,k,l,m,n,o,p;
   double diff_energy, val, minval;
   double epsilon = 1e-200;

   for(j=0;j<numberresidues;j++)
    {
    
      for(k=0;k<RotNumber[j];k++) 
       {
         l = 0;
	 while(l<RotNumber[j] && AcceptRes[j][k]==1) 
	   {
	    if (l != k)
	    {
             diff_energy = - log(Ls[j][k] + epsilon) + log(Ls[j][l] + epsilon);

	     // cout<<" Init diff_energy ="<<diff_energy<<endl;
    	      for(m=0;m<numberresidues;m++)
	       {
               
		 if( (j != m) && (Psi[j][m] != (double*)0) ) 
		   {
                      for(n=0;n<RotNumber[m];n++)
		       {
                        if(AcceptRes[m][n] == 1)
   	                 {
                          for(o=0;o<numberresidues;o++)
	                   {
               	            if( (o != m) && (j != o) && (Psi[j][o] != (double*)0)) 
		              {
                                  minval = 1000000.0;
                                  for(p=0;p<RotNumber[o];p++)
		                   {
    			             if(AcceptRes[o][p] == 1)
   	                              {
                                        if(j<m)  val = - log(Psi[j][m][RotNumber[m]*k+n] + epsilon) + log(Psi[j][m][RotNumber[m]*l+n] + epsilon);
                                        else  val = - log(Psi[j][m][RotNumber[j]*n+k] + epsilon) + log(Psi[j][m][RotNumber[j]*n+l] + epsilon);
                                     
                                         if(j<o)  val += (- log(Psi[j][o][RotNumber[o]*k+p] + epsilon) + log(Psi[j][o][RotNumber[o]*l+p] + epsilon));
                                         else     val += (- log(Psi[j][o][RotNumber[j]*p+k] + epsilon) + log(Psi[j][o][RotNumber[j]*p+l] + epsilon));
                                   
                                         if (val<minval) minval = val;
                                                         
                                        }
				     //cout<<" m ="<<m<<" n ="<<n<<" Psi(j,m,k)", Psi[j][m][RotNumber[m]*k+n], " Psi(j,m,l)", Psi[j][m][RotNumber[m]*l+n], " val ="<<val<<" minval ="<<minval<<endl;
                                     }
                                 }
                               } 
                              } 
                             }                              

                         }    
		                         
	          diff_energy += minval;
	       }             	                      
	    
                if (-1*(diff_energy - alpha)<epsilon) AcceptRes[j][k] = 0;
	    }  
	    //cout<<" j ="<<j<<" k ="<<k<<" l ="<<l<<" m ="<<m<<" Accept[j][k] ="<<AcceptRes[j][k]<<" diff_energy ="<<diff_energy<<endl;
	    l++;
           }              
       }             
      }
}        



void RotProtein::FinishDEE()
  {
 int j,k,remaining_rot;

  NResiduesAfterDEE = 0;
  for(j=0;j<numberresidues;j++)
   {
    remaining_rot = 0;
    for(k=0;k<RotNumber[j];k++) if(AcceptRes[j][k]==1) AuxArray[remaining_rot++] = k;   //The valid rotamers conf are determined

    if (remaining_rot==0)  
     {
       for(k=0;k<RotNumber[j];k++)  AuxArray[remaining_rot++] = k;   //Something wrong with energy function and DEE(inconsistency happened here,best way of solving it)  
       //cout<<"One case here "<<j<<endl;
     }

    if (remaining_rot==1)  
      {
       Fixed[j] = AuxArray[0];   //Fixed will be used in the evaluation
       VarPos[j] = -1;
       //cout<<"fixed j ="<<j<<" fixed val ="<<Fixed[j]<<endl;
      }
    else 
     {
       Fixed[j] = -1;
       RemainResPositions[NResiduesAfterDEE] = j; 
       VarPos[j] = NResiduesAfterDEE;                                  // Position of the original variable in the reduced array
       NFinalRot[NResiduesAfterDEE] = remaining_rot;                   //Number of remaining rotamers for i
       FinalRot[NResiduesAfterDEE] = new int[remaining_rot];           // Store the remaining rotamers
       for(k=0;k<remaining_rot;k++)  FinalRot[NResiduesAfterDEE][k] = AuxArray[k];
       //cout<<"res "<<j<<" NRes "<<NResiduesAfterDEE<<" FinalNRes "<<NFinalRot[NResiduesAfterDEE]<<endl;
       NResiduesAfterDEE++;
     }       
   }


  //cout<<NResiduesAfterDEE<<endl;
  //for(j=0;j<NResiduesAfterDEE;j++) cout<<RemainResPositions[j]<<" "<<NFinalRot[j]<<endl;
   
  }


// **********************************************  FoldPotential Class ****************




void FoldPotential::EmptyPsi()  
{ 
  int i,j;
  Psi = new double**[numberresidues];

  for(i=0;i<numberresidues;i++)
    {
      Psi[i] = new double*[numberresidues]; 
      for(j=0;j<numberresidues;j++)  Psi[i][j] = (double*)0;
    }

}





void FoldPotential::PrintAdjacencyMatrix()  
{
  int i,j;
  
  for(i=0;i<numberresidues;i++) 
  {
    for(j=0;j<numberresidues;j++) 
      {     
	cout<<Matrix[i][j]<<" ";
      }
    cout<<endl<<"--------"<<i<<"------ "<<endl;
    }
}




FoldPotential::FoldPotential(FILE* stream, double temp)  // Creates matrixs for TE13 potential function.
{  
  int i;
  char filedetails[30];
 
  t  = temp;
  NumberAmiAcids = 20;
  fscanf(stream, "%s %d \n", &allfiledetails,&numberresidues);
  //cout<<filedetails<<" "<<numberresidues<<endl;

  sequence = new unsigned int[numberresidues];

  for(i=0;i<numberresidues;i++) 
   {
     //cout<<i<<endl;
     fscanf(stream, "%s \n", &filedetails);
     //cout<<filedetails;
   
if (strcmp(filedetails,"ALA") == 0) sequence[i] = 0;
else if (strcmp(filedetails,"ARG") == 0) sequence[i] = 1;
else if (strcmp(filedetails,"ASN") == 0) sequence[i] = 2;
else if (strcmp(filedetails,"ASP") == 0) sequence[i] = 3;
else if (strcmp(filedetails,"CYS") == 0) sequence[i] = 4;
else if (strcmp(filedetails,"GLN") == 0) sequence[i] = 5;
else if (strcmp(filedetails,"GLU") == 0) sequence[i] = 6;
else if (strcmp(filedetails,"GLY") == 0) sequence[i] = 7;
else if (strcmp(filedetails,"HIS") == 0) sequence[i] = 8;
else if (strcmp(filedetails,"ILE") == 0) sequence[i] = 9;
else if (strcmp(filedetails,"LEU") == 0) sequence[i] = 10;
else if (strcmp(filedetails,"LYS") == 0) sequence[i] = 11;
else if (strcmp(filedetails,"MET") == 0) sequence[i] = 12;
else if (strcmp(filedetails,"PHE") == 0) sequence[i] = 13;
else if (strcmp(filedetails,"PRO") == 0) sequence[i] = 14;
else if (strcmp(filedetails,"SER") == 0) sequence[i] = 15;
else if (strcmp(filedetails,"THR") == 0) sequence[i] = 16;
else if (strcmp(filedetails,"TRP") == 0) sequence[i] = 17;
else if (strcmp(filedetails,"TYR") == 0) sequence[i] = 18;
else if (strcmp(filedetails,"VAL") == 0) sequence[i] = 19;
 else cout<<"Unknown aminoacid -- "<<filedetails<<endl;

// cout<<" "<<sequence[i];
 }    
  //  cout<<endl;
}



void FoldPotential::FillPotential(FILE* stream)  // Creates matrixs for TE13 potential function.
{  
  int i,j,k,l;
  double dist;
  
  order  = new int[numberresidues]; 
  Matrix = new unsigned int*[numberresidues];

      for(j=0;j<numberresidues;j++) 
      {
 	  order[j] = j+1;
          Matrix[j] =  new unsigned int[numberresidues];
          for(k=0;k<numberresidues;k++)  Matrix[j][k] = 0;
          Matrix[j][j] = 1;
      }    

 
 Ls = new double*[numberresidues];

 for(j=0;j<numberresidues;j++)
   {
    Ls[j] = new double[NumberAmiAcids];
    for(k=0;k<NumberAmiAcids;k++)
      {  
	Ls[j][k] = 1; //It is assumed that the single energy contribution is 0 for all the components 
      }
  
   }          
                    
 
 EmptyPsi();  


 for(i=0;i<numberresidues-1;i++)
   for(j=i+1;j<numberresidues;j++)
     {
       fscanf(stream, "%lf ",&dist);
       if(dist>2.0 && dist<=9.0) 
        {
          Matrix[i][j] = 1;
          Matrix[j][i] = 1;
          Psi[i][j] = new double[NumberAmiAcids*NumberAmiAcids];
          for(k=0;k<NumberAmiAcids;k++)
            for(l=0;l<NumberAmiAcids;l++)
	      {
		Psi[i][j][k*NumberAmiAcids+l] = exp(-1*PotTable(dist,k,l)/t);
              } 
         }
     }
 //cout<<"Finished potential "<<endl;
}

  

double FoldPotential::PotTable(double dist, int k, int l)
{
  int pos;
  double PotValues[13][400] = {-9.962,4.121,-3.312,-4.035,-9.707,-1.494,13.884,-3.078,-9.222,-2.599,-3.295,4.570,0.803,-2.214,-2.038,-8.228,13.466,-5.572,7.537,8.568,4.121,7.565,-2.111,-3.678,4.082,-1.823,-0.016,-0.565,-2.455,1.982,6.507,8.160,-2.991,0.111,0.042,-0.765,3.180,3.975,-1.250,-3.039,-3.312,-2.111,-0.885,-2.396,2.641,3.285,1.128,5.021,-2.302,4.164,4.947,6.196,4.758,4.636,-1.337,-3.953,-0.066,9.654,-2.761,3.695,-4.035,-3.678,-2.396,6.014,-0.165,0.579,9.443,-5.684,-5.871,4.350,-0.762,-5.839,3.547,8.735,3.937,-0.669,1.627,10.011,2.608,8.674,-9.707,4.082,2.641,-0.165,-14.314,-2.483,-3.253,-2.198,-6.185,5.353,3.203,3.586,-1.816,-7.582,4.590,2.291,1.995,-9.994,3.507,-5.396,-1.494,-1.823,3.285,0.579,-2.483,4.790,8.223,6.935,3.695,-0.337,1.833,1.024,1.738,3.365,-3.034,1.954,3.045,-6.051,-4.409,-4.545,13.884,-0.016,1.128,9.443,-3.253,8.223,10.021,1.718,-1.284,7.657,4.121,0.945,3.418,6.333,10.014,-7.110,5.072,-3.989,4.426,0.919,-3.078,-0.565,5.021,-5.684,-2.198,6.935,1.718,-7.986,4.789,-6.519,4.574,4.426,4.864,-9.970,-4.444,-7.485,-7.859,-4.429,-9.939,-1.439,-9.222,-2.455,-2.302,-5.871,-6.185,3.695,-1.284,4.789,-5.260,1.134,2.665,-0.019,2.970,-5.139,-5.853,-9.986,-9.467,-9.941,-1.749,-3.041,-2.599,1.982,4.164,4.350,5.353,-0.337,7.657,-6.519,1.134,10.056,10.942,3.927,5.112,4.661,5.340,5.570,5.343,2.016,1.891,10.078,-3.295,6.507,4.947,-0.762,3.203,1.833,4.121,4.574,2.665,10.942,10.734,9.922,2.807,11.786,10.038,2.447,8.248,4.203,2.275,10.793,4.570,8.160,6.196,-5.839,3.586,1.024,0.945,4.426,-0.019,3.927,9.922,6.260,4.335,-3.648,10.026,3.282,0.145,-3.911,-0.691,-3.279,0.803,-2.991,4.758,3.547,-1.816,1.738,3.418,4.864,2.970,5.112,2.807,4.335,4.549,5.608,2.585,1.393,7.443,3.278,-4.250,10.038,-2.214,0.111,4.636,8.735,-7.582,3.365,6.333,-9.970,-5.139,4.661,11.786,-3.648,5.608,4.922,-1.551,-6.956,8.282,6.190,10.051,4.450,-2.038,0.042,-1.337,3.937,4.590,-3.034,10.014,-4.444,-5.853,5.340,10.038,10.026,2.585,-1.551,3.674,7.105,2.935,-3.886,-8.372,5.038,-8.228,-0.765,-3.953,-0.669,2.291,1.954,-7.110,-7.485,-9.986,5.570,2.447,3.282,1.393,-6.956,7.105,-4.224,-6.187,-7.749,8.163,4.479,13.466,3.180,-0.066,1.627,1.995,3.045,5.072,-7.859,-9.467,5.343,8.248,0.145,7.443,8.282,2.935,-6.187,2.395,8.336,4.629,3.526,-5.572,3.975,9.654,10.011,-9.994,-6.051,-3.989,-4.429,-9.941,2.016,4.203,-3.911,3.278,6.190,-3.886,-7.749,8.336,-0.658,-1.261,1.058,7.537,-1.250,-2.761,2.608,3.507,-4.409,4.426,-9.939,-1.749,1.891,2.275,-0.691,-4.250,10.051,-8.372,8.163,4.629,-1.261,-2.146,1.372,8.568,-3.039,3.695,8.674,-5.396,-4.545,0.919,-1.439,-3.041,10.078,10.793,-3.279,10.038,4.450,5.038,4.479,3.526,1.058,1.372,-0.939,-10.000,3.863,-3.284,-3.797,-10.000,-1.524,3.802,-2.513,-9.239,-2.649,-3.430,4.521,0.782,-2.254,-2.075,-7.269,2.925,-5.740,3.550,-0.308,3.863,7.400,-2.133,-3.708,4.078,-1.839,-1.928,-0.598,-2.466,1.405,6.325,8.132,-3.000,0.094,-0.037,-0.791,3.153,3.968,-1.266,-3.079,-3.284,-2.133,-0.918,-2.423,2.451,3.269,1.142,2.173,-2.313,4.141,4.912,6.167,4.749,4.619,-1.356,-3.573,0.221,4.948,-2.778,3.664,-3.797,-3.708,-2.423,5.773,-0.358,0.558,9.374,-5.883,-5.980,4.296,-0.805,-6.035,3.535,8.712,3.913,-0.702,1.594,10.000,2.585,8.634,-10.000,4.078,2.451,-0.358,-6.807,-2.516,-3.700,-2.320,-6.192,5.114,3.178,3.567,-1.834,-7.604,4.565,2.268,1.978,-10.000,3.491,-5.432,-1.524,-1.839,3.269,0.558,-2.516,4.782,8.199,6.901,3.685,-0.357,1.800,0.999,1.729,3.349,-3.407,1.930,3.035,-6.057,-4.796,-4.633,3.802,-1.928,1.142,9.374,-3.700,8.199,10.000,1.653,-1.298,7.622,3.683,0.894,3.404,6.305,9.938,-7.285,4.975,-3.997,4.348,0.887,-2.513,-0.598,2.173,-5.883,-2.320,6.901,1.653,-7.959,4.773,-6.559,4.517,4.381,4.847,-10.000,-4.476,-8.137,-7.891,-4.440,-10.000,-1.500,-9.239,-2.466,-2.313,-5.980,-6.192,3.685,-1.298,4.773,-5.267,1.093,2.645,-0.036,2.965,-5.149,-5.862,-10.000,-9.480,-9.945,-1.963,-3.057,-2.649,1.405,4.141,4.296,5.114,-0.357,7.622,-6.559,1.093,10.000,10.000,3.848,5.087,4.596,5.317,5.538,5.312,1.993,1.833,10.000,-3.430,6.325,4.912,-0.805,3.178,1.800,3.683,4.517,2.645,10.000,10.000,9.852,2.495,10.000,10.000,2.381,8.213,4.172,2.059,10.000,4.521,8.132,6.167,-6.035,3.567,0.999,0.894,4.381,-0.036,3.848,9.852,6.237,4.260,-3.888,10.000,3.245,0.108,-3.920,-0.737,-3.588,0.782,-3.000,4.749,3.535,-1.834,1.729,3.404,4.847,2.965,5.087,2.495,4.260,4.534,5.366,2.576,1.362,7.441,3.275,-4.260,10.000,-2.254,0.094,4.619,8.712,-7.604,3.349,6.305,-10.000,-5.149,4.596,10.000,-3.888,5.366,4.875,-1.569,-7.030,8.197,6.179,10.000,3.237,-2.075,-0.037,-1.356,3.913,4.565,-3.407,9.938,-4.476,-5.862,5.317,10.000,10.000,2.576,-1.569,3.664,7.079,2.904,-3.946,-8.567,4.857,-7.269,-0.791,-3.573,-0.702,2.268,1.930,-7.285,-8.137,-10.000,5.538,2.381,3.245,1.362,-7.030,7.079,-4.481,-6.257,-7.780,8.134,4.437,2.925,3.153,0.221,1.594,1.978,3.035,4.975,-7.891,-9.480,5.312,8.213,0.108,7.441,8.197,2.904,-6.257,0.198,8.328,4.605,3.482,-5.740,3.968,4.948,10.000,-10.000,-6.057,-3.997,-4.440,-9.945,1.993,4.172,-3.920,3.275,6.179,-3.946,-7.780,8.328,-0.673,-1.290,1.014,3.550,-1.266,-2.778,2.585,3.491,-4.796,4.348,-10.000,-1.963,1.833,2.059,-0.737,-4.260,10.000,-8.567,8.134,4.605,-1.290,-2.220,1.325,-0.308,-3.079,3.664,8.634,-5.432,-4.633,0.887,-1.500,-3.057,10.000,10.000,-3.588,10.000,3.237,4.857,4.437,3.482,1.014,1.325,-2.181,-10.000,3.567,-4.050,-2.867,-7.353,-2.486,-1.651,-0.739,-6.801,-3.742,-3.303,3.360,0.382,-2.291,-3.226,-0.939,-0.816,-4.521,-0.941,-3.235,3.567,6.970,-2.945,-6.154,4.044,-1.978,-3.862,-0.091,-2.512,-0.060,6.003,7.791,-3.212,-1.269,-0.198,-1.655,4.402,1.861,-1.362,-3.525,-4.050,-2.945,-1.364,-4.233,2.233,2.820,0.455,-0.457,-2.814,4.058,3.898,5.392,4.692,4.339,-2.011,-4.471,0.797,0.148,-3.129,3.553,-2.867,-6.154,-4.233,5.497,-0.739,0.442,8.768,-3.573,-6.360,3.283,-1.317,-8.664,3.112,8.169,3.887,-2.707,-0.795,10.000,2.546,8.018,-7.353,4.044,2.233,-0.739,-7.185,-2.573,-4.199,-3.133,-6.047,4.617,1.706,3.513,-2.137,-4.424,4.134,2.143,2.125,-10.000,3.726,-6.724,-2.486,-1.978,2.820,0.442,-2.573,4.754,7.333,6.791,2.959,-0.657,0.928,0.736,1.701,3.145,-2.996,0.851,2.709,-7.237,-5.901,-4.790,-1.651,-3.862,0.455,8.768,-4.199,7.333,10.000,1.489,-1.396,6.579,3.009,-0.626,3.233,4.871,9.597,-8.586,4.253,-4.024,4.213,0.489,-0.739,-0.091,-0.457,-3.573,-3.133,6.791,1.489,-6.461,4.499,-4.460,4.199,4.858,4.705,-5.792,-4.088,-5.491,-5.537,-4.262,-9.716,-1.868,-6.801,-2.512,-2.814,-6.360,-6.047,2.959,-1.396,4.499,-5.284,1.112,2.312,-0.433,2.948,-5.171,-5.897,-7.783,-10.000,-10.000,-2.571,-3.673,-3.742,-0.060,4.058,3.283,4.617,-0.657,6.579,-4.460,1.112,9.555,10.000,3.680,4.452,4.025,4.997,5.360,4.678,1.941,3.312,10.000,-3.303,6.003,3.898,-1.317,1.706,0.928,3.009,4.199,2.312,10.000,10.000,9.655,1.627,10.000,9.603,1.435,7.917,3.723,1.128,8.526,3.360,7.791,5.392,-8.664,3.513,0.736,-0.626,4.858,-0.433,3.680,9.655,6.158,4.000,-4.191,10.000,2.475,-0.478,-4.050,-1.393,-2.407,0.382,-3.212,4.692,3.112,-2.137,1.701,3.233,4.705,2.948,4.452,1.627,4.000,4.508,3.602,2.453,1.210,7.385,3.263,-3.565,9.513,-2.291,-1.269,4.339,8.169,-4.424,3.145,4.871,-5.792,-5.171,4.025,10.000,-4.191,3.602,5.171,-2.645,-4.807,7.168,5.767,10.000,1.699,-3.226,-0.198,-2.011,3.887,4.134,-2.996,9.597,-4.088,-5.897,4.997,9.603,10.000,2.453,-2.645,3.626,8.761,4.860,-4.834,-9.123,4.383,-0.939,-1.655,-4.471,-2.707,2.143,0.851,-8.586,-5.491,-7.783,5.360,1.435,2.475,1.210,-4.807,8.761,-4.976,-5.338,-6.004,7.822,3.399,-0.816,4.402,0.797,-0.795,2.125,2.709,4.253,-5.537,-10.000,4.678,7.917,-0.478,7.385,7.168,4.860,-5.338,-2.579,7.837,4.283,2.961,-4.521,1.861,0.148,10.000,-10.000,-7.237,-4.024,-4.262,-10.000,1.941,3.723,-4.050,3.263,5.767,-4.834,-6.004,7.837,-0.962,1.809,0.229,-0.941,-1.362,-3.129,2.546,3.726,-5.901,4.213,-9.716,-2.571,3.312,1.128,-1.393,-3.565,10.000,-9.123,7.822,4.283,1.809,2.428,0.482,-3.235,-3.525,3.553,8.018,-6.724,-4.790,0.489,-1.868,-3.673,10.000,8.526,-2.407,9.513,1.699,4.383,3.399,2.961,0.229,0.482,0.023,-9.326,4.673,-0.465,-0.351,-7.379,-3.797,-1.912,0.383,-3.403,-3.228,-7.319,3.240,-0.815,-2.589,-4.294,-0.469,-0.524,-3.611,-5.618,-8.136,4.673,5.282,-3.139,-10.000,2.895,-6.056,-6.900,-0.912,-0.319,-4.209,3.117,9.597,-5.439,-2.361,-2.530,-2.765,3.013,0.022,-4.333,-4.249,-0.465,-3.139,0.392,-6.576,-0.604,1.164,-2.381,-2.035,-0.124,3.234,1.444,4.967,4.388,5.169,-2.268,-6.718,-0.256,-4.767,-4.331,-0.024,-0.351,-10.000,-6.576,4.250,-2.262,0.092,5.065,-0.718,-5.558,-1.292,0.266,-9.280,1.240,7.134,4.144,-0.595,-3.022,9.743,0.867,5.302,-7.379,2.895,-0.604,-2.262,-5.583,-2.421,-6.083,-1.322,-6.269,-0.441,-1.563,0.116,-3.931,-3.747,1.416,-1.720,-0.639,-10.000,4.984,-3.188,-3.797,-6.056,1.164,0.092,-2.421,3.798,3.738,6.484,1.409,-0.476,-1.577,-0.801,1.391,0.905,-1.852,-1.599,-0.540,-7.651,-5.040,-5.716,-1.912,-6.900,-2.381,5.065,-6.083,3.738,9.848,-0.177,-3.726,3.388,-2.864,-3.609,2.702,5.039,6.163,-10.000,1.995,-4.603,2.121,0.507,0.383,-0.912,-2.035,-0.718,-1.322,6.484,-0.177,-3.459,4.607,-2.326,0.575,1.689,2.809,-0.782,-3.204,-1.005,-0.233,-1.229,-6.618,-0.670,-3.403,-0.319,-0.124,-5.558,-6.269,1.409,-3.726,4.607,-4.131,-0.975,-0.757,0.519,2.735,-7.198,-6.610,-5.723,-9.867,-10.000,-3.004,-3.808,-3.228,-4.209,3.234,-1.292,-0.441,-0.476,3.388,-2.326,-0.975,4.093,10.000,1.803,0.917,1.455,3.412,7.690,3.133,1.367,4.550,5.876,-7.319,3.117,1.444,0.266,-1.563,-1.577,-2.864,0.575,-0.757,10.000,8.483,6.899,-1.484,4.842,4.625,-1.436,5.142,0.397,-3.178,1.358,3.240,9.597,4.967,-9.280,0.116,-0.801,-3.609,1.689,0.519,1.803,6.899,8.154,1.685,-6.386,8.589,1.113,2.163,-5.805,-2.597,-0.119,-0.815,-5.439,4.388,1.240,-3.931,1.391,2.702,2.809,2.735,0.917,-1.484,1.685,3.982,2.045,1.704,0.093,5.447,1.447,-5.001,4.100,-2.589,-2.361,5.169,7.134,-3.747,0.905,5.039,-0.782,-7.198,1.455,4.842,-6.386,2.045,4.288,-3.982,-5.723,5.316,3.509,9.931,-2.729,-4.294,-2.530,-2.268,4.144,1.416,-1.852,6.163,-3.204,-6.610,3.412,4.625,8.589,1.704,-3.982,2.229,9.392,5.038,-5.021,-10.000,1.910,-0.469,-2.765,-6.718,-0.595,-1.720,-1.599,-10.000,-1.005,-5.723,7.690,-1.436,1.113,0.093,-5.723,9.392,-4.573,-6.065,-3.532,4.913,-0.552,-0.524,3.013,-0.256,-3.022,-0.639,-0.540,1.995,-0.233,-9.867,3.133,5.142,2.163,5.447,5.316,5.038,-6.065,-4.091,6.992,4.411,0.305,-3.611,0.022,-4.767,9.743,-10.000,-7.651,-4.603,-1.229,-10.000,1.367,0.397,-5.805,1.447,3.509,-5.021,-3.532,6.992,-1.127,2.193,-4.248,-5.618,-4.333,-4.331,0.867,4.984,-5.040,2.121,-6.618,-3.004,4.550,-3.178,-2.597,-5.001,9.931,-10.000,4.913,4.411,2.193,3.029,-3.315,-8.136,-4.249,-0.024,5.302,-3.188,-5.716,0.507,-0.670,-3.808,5.876,1.358,-0.119,4.100,-2.729,1.910,-0.552,0.305,-4.248,-3.315,-1.208,-3.380,5.523,-0.436,2.027,-2.836,-2.734,-0.210,0.884,-3.737,-4.655,-7.193,-1.053,-2.607,-2.817,-4.110,1.009,-2.131,-4.930,-5.003,-4.462,5.523,2.590,-0.227,-5.728,4.181,-5.783,-3.765,-1.820,1.322,-4.676,-1.705,10.000,-7.524,-2.234,-0.840,-0.254,0.281,0.548,-5.273,-5.440,-0.436,-0.227,-0.221,-6.290,-6.282,1.208,-4.039,0.449,-0.059,2.283,-1.730,7.218,1.455,4.998,-3.920,-4.164,-4.166,-6.159,-4.077,0.455,2.027,-5.728,-6.290,2.234,-0.813,-2.889,6.718,0.386,-6.035,-6.561,0.386,-9.309,-1.463,2.750,3.932,0.302,-1.986,7.779,-0.794,2.035,-2.836,4.181,-6.282,-0.813,-9.908,-3.317,-3.776,-2.987,-8.184,-3.956,-8.157,-0.529,-2.433,-5.332,-3.722,-3.585,-2.627,-6.654,2.921,-2.881,-2.734,-5.783,1.208,-2.889,-3.317,0.574,0.178,3.340,-0.373,-3.708,-4.823,-1.245,-1.057,-2.132,-1.551,0.141,-3.847,-9.788,-7.493,-5.480,-0.210,-3.765,-4.039,6.718,-3.776,0.178,10.000,2.992,-6.672,2.863,-7.112,-9.450,-0.561,-1.128,6.242,-6.466,0.760,-5.284,-0.765,-6.845,0.884,-1.820,0.449,0.386,-2.987,3.340,2.992,-1.753,3.025,-3.886,-3.742,1.081,-0.413,-4.093,-2.518,1.129,-0.931,2.721,-5.160,-2.913,-3.737,1.322,-0.059,-6.035,-8.184,-0.373,-6.672,3.025,-2.829,-3.203,-2.739,0.680,0.440,-10.000,-6.589,-2.576,-4.792,-10.000,0.039,-1.734,-4.655,-4.676,2.283,-6.561,-3.956,-3.708,2.863,-3.886,-3.203,-3.711,-0.784,-0.155,-2.997,-0.951,-1.443,6.811,-0.202,1.775,1.257,-3.150,-7.193,-1.705,-1.730,0.386,-8.157,-4.823,-7.112,-3.742,-2.739,-0.784,0.813,-2.565,-2.895,1.208,-2.831,-2.537,-0.059,-3.482,-6.205,-3.092,-1.053,10.000,7.218,-9.309,-0.529,-1.245,-9.450,1.081,0.680,-0.155,-2.565,10.000,0.977,-5.793,6.946,2.097,-0.815,-3.535,-2.330,-5.627,-2.607,-7.524,1.455,-1.463,-2.433,-1.057,-0.561,-0.413,0.440,-2.997,-2.895,0.977,1.191,-0.674,-0.297,2.706,4.349,-1.380,-7.617,0.983,-2.817,-2.234,4.998,2.750,-5.332,-2.132,-1.128,-4.093,-10.000,-0.951,1.208,-5.793,-0.674,1.396,-2.721,-3.979,-3.493,0.988,1.339,-5.818,-4.110,-0.840,-3.920,3.932,-3.722,-1.551,6.242,-2.518,-6.589,-1.443,-2.831,6.946,-0.297,-2.721,0.047,8.386,3.099,-3.462,-10.000,-3.761,1.009,-0.254,-4.164,0.302,-3.585,0.141,-6.466,1.129,-2.576,6.811,-2.537,2.097,2.706,-3.979,8.386,-4.238,-2.271,-3.828,3.431,-3.758,-2.131,0.281,-4.166,-1.986,-2.627,-3.847,0.760,-0.931,-4.792,-0.202,-0.059,-0.815,4.349,-3.493,3.099,-2.271,-4.743,6.123,0.174,-4.464,-4.930,0.548,-6.159,7.779,-6.654,-9.788,-5.284,2.721,-10.000,1.775,-3.482,-3.535,-1.380,0.988,-3.462,-3.828,6.123,-2.849,2.971,-6.174,-5.003,-5.273,-4.077,-0.794,2.921,-7.493,-0.765,-5.160,0.039,1.257,-6.205,-2.330,-7.617,1.339,-10.000,3.431,0.174,2.971,1.358,-4.486,-4.462,-5.440,0.455,2.035,-2.881,-5.480,-6.845,-2.913,-1.734,-3.150,-3.092,-5.627,0.983,-5.818,-3.761,-3.758,-4.464,-6.174,-4.486,-5.295,-1.036,4.955,1.012,8.328,-0.478,1.970,0.859,4.301,-1.419,-2.807,-4.260,-1.105,-1.503,-3.205,-4.838,0.431,-2.773,-4.349,-1.913,-3.593,4.955,0.377,-2.118,-7.259,3.564,-3.002,-3.122,-1.934,-1.422,-0.738,-0.401,9.822,-6.930,-3.203,-1.673,-3.331,-0.452,2.882,-3.332,-5.366,1.012,-2.118,-0.162,-2.999,-6.853,-0.393,-1.486,-0.575,-0.177,0.603,0.044,6.188,2.143,0.904,-4.318,-1.368,-4.669,-5.553,-5.511,0.533,8.328,-7.259,-2.999,2.663,-3.330,1.026,5.128,-3.102,-3.672,-0.872,-3.385,-6.317,-4.983,0.243,4.101,1.960,1.258,3.967,-8.039,1.658,-0.478,3.564,-6.853,-3.330,-9.906,-1.075,-3.020,-1.229,-4.590,-4.457,-9.085,-0.581,-0.810,-8.672,-2.234,-2.519,-2.160,-5.425,-0.823,-3.016,1.970,-3.002,-0.393,1.026,-1.075,-0.879,2.613,-0.767,-0.865,-4.887,-3.854,2.411,-3.526,-4.162,2.082,-0.523,0.295,-6.529,-6.850,-7.071,0.859,-3.122,-1.486,5.128,-3.020,2.613,8.917,3.396,-5.867,-2.165,-6.429,-6.150,0.413,-4.800,3.969,-4.568,0.171,-3.379,-1.795,-3.419,4.301,-1.934,-0.575,-3.102,-1.229,-0.767,3.396,0.303,1.539,-1.161,-4.272,3.113,1.117,-5.500,-0.995,-0.306,0.749,4.041,-3.826,-0.603,-1.419,-1.422,-0.177,-3.672,-4.590,-0.865,-5.867,1.539,-2.482,-6.518,-1.663,2.676,-4.293,-10.000,-4.700,-0.276,-2.787,-6.975,-1.918,-2.120,-2.807,-0.738,0.603,-0.872,-4.457,-4.887,-2.165,-1.161,-6.518,-9.219,-4.727,-3.957,-6.610,-4.867,-2.726,3.541,-2.688,-2.304,-1.742,-3.828,-4.260,-0.401,0.044,-3.385,-9.085,-3.854,-6.429,-4.272,-1.663,-4.727,-6.790,-3.213,-4.082,-5.269,-4.205,2.981,-1.985,-2.840,-6.433,-8.688,-1.105,9.822,6.188,-6.317,-0.581,2.411,-6.150,3.113,2.676,-3.957,-3.213,10.000,-1.232,-1.275,0.272,2.901,-2.153,-1.348,-2.312,-1.276,-1.503,-6.930,2.143,-4.983,-0.810,-3.526,0.413,1.117,-4.293,-6.610,-4.082,-1.232,0.785,-2.152,-0.960,4.190,4.105,-2.570,-9.303,-1.613,-3.205,-3.203,0.904,0.243,-8.672,-4.162,-4.800,-5.500,-10.000,-4.867,-5.269,-1.275,-2.152,-5.339,-3.095,-2.704,-2.836,-4.299,-3.864,-8.073,-4.838,-1.673,-4.318,4.101,-2.234,2.082,3.969,-0.995,-4.700,-2.726,-4.205,0.272,-0.960,-3.095,1.887,9.038,0.803,1.435,-9.475,-2.831,0.431,-3.331,-1.368,1.960,-2.519,-0.523,-4.568,-0.306,-0.276,3.541,2.981,2.901,4.190,-2.704,9.038,-3.225,-1.325,-2.738,-2.775,-2.045,-2.773,-0.452,-4.669,1.258,-2.160,0.295,0.171,0.749,-2.787,-2.688,-1.985,-2.153,4.105,-2.836,0.803,-1.325,-1.735,0.505,-3.213,-5.181,-4.349,2.882,-5.553,3.967,-5.425,-6.529,-3.379,4.041,-6.975,-2.304,-2.840,-1.348,-2.570,-4.299,1.435,-2.738,0.505,-5.051,-0.818,-6.649,-1.913,-3.332,-5.511,-8.039,-0.823,-6.850,-1.795,-3.826,-1.918,-1.742,-6.433,-2.312,-9.303,-3.864,-9.475,-2.775,-3.213,-0.818,0.033,-6.044,-3.593,-5.366,0.533,1.658,-3.016,-7.071,-3.419,-0.603,-2.120,-3.828,-8.688,-1.276,-1.613,-8.073,-2.831,-2.045,-5.181,-6.649,-6.044,-6.804,-1.174,3.696,1.579,5.085,0.369,3.102,-1.056,-0.159,0.227,-0.362,-0.073,1.439,-0.547,-2.157,-1.039,4.051,-0.282,-4.508,-3.274,-1.697,3.696,-2.554,-2.816,-4.182,1.319,-5.838,-7.250,0.924,-0.035,-0.406,0.814,10.000,-7.012,-2.903,-2.073,-0.139,-0.406,4.264,-6.050,-0.558,1.579,-2.816,0.612,-4.649,-4.986,0.218,-1.540,-0.315,2.112,-2.314,3.385,3.676,0.633,2.295,-1.769,2.930,-3.497,-3.525,-4.683,-0.469,5.085,-4.182,-4.649,-0.067,-4.913,1.096,3.783,0.339,-1.027,0.481,-3.728,-2.655,-4.999,0.562,2.951,5.213,0.623,2.427,-6.535,-1.214,0.369,1.319,-4.986,-4.913,-5.884,-0.316,-1.508,-1.981,-1.516,-0.718,-5.407,-0.719,0.830,-4.472,-3.232,-1.499,-4.437,-4.087,-2.743,-4.688,3.102,-5.838,0.218,1.096,-0.316,-2.712,1.164,1.711,-0.919,-0.177,-4.644,4.884,-3.061,-4.515,2.665,-0.637,0.976,-4.648,-8.721,-5.947,-1.056,-7.250,-1.540,3.783,-1.508,1.164,7.099,3.422,-4.207,-2.528,-1.683,-4.341,1.265,1.639,4.618,-3.616,2.152,-4.494,-1.907,-1.351,-0.159,0.924,-0.315,0.339,-1.981,1.711,3.422,-0.109,-0.611,-1.166,-1.891,1.895,0.311,-0.637,3.750,-1.320,-0.825,1.278,-2.070,-1.948,0.227,-0.035,2.112,-1.027,-1.516,-0.919,-4.207,-0.611,-0.808,-5.518,-0.779,5.027,-0.380,-7.932,-4.163,-2.048,-2.671,-5.463,0.849,-0.991,-0.362,-0.406,-2.314,0.481,-0.718,-0.177,-2.528,-1.166,-5.518,-6.823,-10.000,-1.386,-6.752,-7.300,-3.118,3.801,-5.554,-1.940,-1.595,-2.714,-0.073,0.814,3.385,-3.728,-5.407,-4.644,-1.683,-1.891,-0.779,-10.000,-8.281,-4.642,-4.751,-7.561,-3.647,5.214,-3.555,-4.437,-6.842,-8.222,1.439,10.000,3.676,-2.655,-0.719,4.884,-4.341,1.895,5.027,-1.386,-4.642,10.000,-3.336,0.224,-0.583,2.171,2.832,2.389,-1.198,-6.889,-0.547,-7.012,0.633,-4.999,0.830,-3.061,1.265,0.311,-0.380,-6.752,-4.751,-3.336,0.859,-4.707,1.467,6.717,1.863,-4.071,-8.557,-3.371,-2.157,-2.903,2.295,0.562,-4.472,-4.515,1.639,-0.637,-7.932,-7.300,-7.561,0.224,-4.707,-5.877,-2.096,-0.602,-4.237,-7.290,-3.795,-7.961,-1.039,-2.073,-1.769,2.951,-3.232,2.665,4.618,3.750,-4.163,-3.118,-3.647,-0.583,1.467,-2.096,5.059,10.000,-1.842,3.809,-6.378,-1.441,4.051,-0.139,2.930,5.213,-1.499,-0.637,-3.616,-1.320,-2.048,3.801,5.214,2.171,6.717,-0.602,10.000,-2.389,2.273,-4.441,-1.521,-3.193,-0.282,-0.406,-3.497,0.623,-4.437,0.976,2.152,-0.825,-2.671,-5.554,-3.555,2.832,1.863,-4.237,-1.842,2.273,-1.292,-1.352,-3.658,-2.405,-4.508,4.264,-3.525,2.427,-4.087,-4.648,-4.494,1.278,-5.463,-1.940,-4.437,2.389,-4.071,-7.290,3.809,-4.441,-1.352,-2.455,-3.213,-6.456,-3.274,-6.050,-4.683,-6.535,-2.743,-8.721,-1.907,-2.070,0.849,-1.595,-6.842,-1.198,-8.557,-3.795,-6.378,-1.521,-3.658,-3.213,1.014,-6.757,-1.697,-0.558,-0.469,-1.214,-4.688,-5.947,-1.351,-1.948,-0.991,-2.714,-8.222,-6.889,-3.371,-7.961,-1.441,-3.193,-2.405,-6.456,-6.757,-5.400,-3.007,1.017,0.840,1.844,-0.150,-0.600,0.283,-0.304,2.174,-0.339,1.185,2.595,1.882,1.099,-0.951,1.249,0.807,-5.245,-2.708,-0.492,1.017,-5.054,-3.551,-3.457,0.698,-4.199,-3.316,-0.315,0.954,-4.337,0.657,6.652,-8.931,0.841,-1.947,3.186,1.082,2.812,-4.481,0.615,0.840,-3.551,1.629,-4.121,-4.722,3.659,1.069,0.283,3.351,-1.553,6.685,4.591,-0.253,-1.208,-1.203,2.678,-2.205,-3.919,-4.192,-0.435,1.844,-3.457,-4.121,-1.848,-1.635,-0.546,0.747,0.595,0.655,1.116,-2.102,-3.830,-1.924,1.916,0.469,4.243,-0.760,1.115,-3.308,-0.274,-0.150,0.698,-4.722,-1.635,1.372,-0.827,-2.495,-3.047,0.373,0.062,-1.204,-2.409,1.744,-0.598,-3.442,-0.556,-4.860,-3.305,-2.450,-1.822,-0.600,-4.199,3.659,-0.546,-0.827,1.181,0.500,1.342,-1.022,1.526,-3.607,6.017,-1.925,-4.147,1.269,3.180,2.943,-3.322,-6.043,-3.259,0.283,-3.316,1.069,0.747,-2.495,0.500,1.041,0.843,-6.812,1.736,-2.349,-4.305,0.473,0.518,4.756,0.418,1.136,-4.584,-1.515,0.881,-0.304,-0.315,0.283,0.595,-3.047,1.342,0.843,0.810,-1.962,1.310,-0.626,3.531,0.382,-0.147,2.907,-0.081,-0.115,-1.628,0.228,1.694,2.174,0.954,3.351,0.655,0.373,-1.022,-6.812,-1.962,0.969,-3.142,2.111,7.236,4.194,-3.731,-3.874,-3.106,-0.510,-5.189,3.012,0.299,-0.339,-4.337,-1.553,1.116,0.062,1.526,1.736,1.310,-3.142,-4.210,-6.451,-3.952,-10.000,-8.687,-3.664,5.889,-4.777,-1.181,-1.336,-0.689,1.185,0.657,6.685,-2.102,-1.204,-3.607,-2.349,-0.626,2.111,-6.451,-2.305,-2.502,-1.713,-6.077,0.710,-0.275,-2.199,-4.631,-3.025,-3.457,2.595,6.652,4.591,-3.830,-2.409,6.017,-4.305,3.531,7.236,-3.952,-2.502,10.000,-4.282,-1.757,1.433,-0.975,1.594,-0.240,-1.981,-4.198,1.882,-8.931,-0.253,-1.924,1.744,-1.925,0.473,0.382,4.194,-10.000,-1.713,-4.282,2.879,-2.487,5.445,4.389,0.527,-5.131,-10.000,-0.699,1.099,0.841,-1.208,1.916,-0.598,-4.147,0.518,-0.147,-3.731,-8.687,-6.077,-1.757,-2.487,-7.969,0.034,-1.806,-1.874,-10.000,-3.061,-3.286,-0.951,-1.947,-1.203,0.469,-3.442,1.269,4.756,2.907,-3.874,-3.664,0.710,1.433,5.445,0.034,2.337,6.448,-0.649,5.213,-6.280,-1.910,1.249,3.186,2.678,4.243,-0.556,3.180,0.418,-0.081,-3.106,5.889,-0.275,-0.975,4.389,-1.806,6.448,-3.673,2.169,-4.509,-2.678,-2.362,0.807,1.082,-2.205,-0.760,-4.860,2.943,1.136,-0.115,-0.510,-4.777,-2.199,1.594,0.527,-1.874,-0.649,2.169,-0.730,-2.786,-0.880,-1.580,-5.245,2.812,-3.919,1.115,-3.305,-3.322,-4.584,-1.628,-5.189,-1.181,-4.631,-0.240,-5.131,-10.000,5.213,-4.509,-2.786,-0.528,-2.984,-6.203,-2.708,-4.481,-4.192,-3.308,-2.450,-6.043,-1.515,0.228,3.012,-1.336,-3.025,-1.981,-10.000,-3.061,-6.280,-2.678,-0.880,-2.984,0.878,-4.516,-0.492,0.615,-0.435,-0.274,-1.822,-3.259,0.881,1.694,0.299,-0.689,-3.457,-4.198,-0.699,-3.286,-1.910,-2.362,-1.580,-6.203,-4.516,-0.511,1.512,0.410,0.580,1.962,2.588,-0.087,0.756,-3.224,3.560,-0.860,0.560,-0.579,-0.595,2.542,-1.068,-2.535,-0.560,-3.048,-0.408,-3.479,0.410,-5.525,-2.590,-4.266,-2.797,-0.362,-3.552,1.157,1.232,-0.963,-0.170,1.788,-5.227,-0.746,-3.982,2.474,-0.016,1.418,-0.726,-0.122,0.580,-2.590,3.336,-2.782,-2.514,1.885,6.226,-1.665,2.312,-3.686,4.225,4.435,-1.553,0.320,-0.873,3.712,0.134,-2.765,-1.738,1.478,1.962,-4.266,-2.782,0.529,0.041,0.748,-0.312,-0.961,0.281,2.589,-0.216,-2.132,-1.217,2.076,2.966,1.623,-0.552,4.291,0.232,-0.308,2.588,-2.797,-2.514,0.041,3.626,1.336,-6.089,0.435,0.041,4.317,0.030,-4.349,-0.319,-0.595,-2.361,-0.862,-3.262,-3.409,-1.699,1.195,-0.087,-0.362,1.885,0.748,1.336,2.261,-1.889,0.393,1.113,4.271,-1.502,4.140,-3.347,-0.570,2.574,2.773,3.051,-3.677,-2.278,-2.564,0.756,-3.552,6.226,-0.312,-6.089,-1.889,-1.252,-1.270,-4.363,2.702,-0.463,-4.208,5.509,1.820,3.788,2.294,3.012,-5.928,-0.255,-1.088,-3.224,1.157,-1.665,-0.961,0.435,0.393,-1.270,-1.973,-1.368,1.694,1.150,-1.504,0.687,1.188,0.985,-1.634,-1.254,-1.313,-0.379,2.357,3.560,1.232,2.312,0.281,0.041,1.113,-4.363,-1.368,0.088,-1.461,4.243,3.116,5.712,-1.618,-3.873,-3.500,1.740,-4.952,2.520,-0.540,-0.860,-0.963,-3.686,2.589,4.317,4.271,2.702,1.694,-1.461,-4.800,-2.384,-0.241,-8.243,-2.749,-3.138,2.598,-3.921,0.801,1.060,-0.161,0.560,-0.170,4.225,-0.216,0.030,-1.502,-0.463,1.150,4.243,-2.384,0.775,-3.167,0.980,-3.541,4.538,-1.446,-1.097,-4.667,0.676,-2.397,-0.579,1.788,4.435,-2.132,-4.349,4.140,-4.208,-1.504,3.116,-0.241,-3.167,6.018,-2.066,1.924,0.553,1.408,2.851,0.050,0.210,-5.165,-0.595,-5.227,-1.553,-1.217,-0.319,-3.347,5.509,0.687,5.712,-8.243,0.980,-2.066,3.339,-1.309,4.958,4.025,-1.839,-2.931,-5.145,2.783,2.542,-0.746,0.320,2.076,-0.595,-0.570,1.820,1.188,-1.618,-2.749,-3.541,1.924,-1.309,-3.669,-0.883,0.557,-1.273,-10.000,1.537,1.231,-1.068,-3.982,-0.873,2.966,-2.361,2.574,3.788,0.985,-3.873,-3.138,4.538,0.553,4.958,-0.883,1.816,2.268,-1.170,6.807,-3.098,0.811,-2.535,2.474,3.712,1.623,-0.862,2.773,2.294,-1.634,-3.500,2.598,-1.446,1.408,4.025,0.557,2.268,-4.830,1.467,-2.943,0.986,-2.279,-0.560,-0.016,0.134,-0.552,-3.262,3.051,3.012,-1.254,1.740,-3.921,-1.097,2.851,-1.839,-1.273,-1.170,1.467,-2.669,-4.859,0.686,-1.704,-3.048,1.418,-2.765,4.291,-3.409,-3.677,-5.928,-1.313,-4.952,0.801,-4.667,0.050,-2.931,-10.000,6.807,-2.943,-4.859,1.202,0.389,-5.533,-0.408,-0.726,-1.738,0.232,-1.699,-2.278,-0.255,-0.379,2.520,1.060,0.676,0.210,-5.145,1.537,-3.098,0.986,0.686,0.389,2.497,2.602,-3.479,-0.122,1.478,-0.308,1.195,-2.564,-1.088,2.357,-0.540,-0.161,-2.397,-5.165,2.783,1.231,0.811,-2.279,-1.704,-5.533,2.602,-0.025,4.161,-0.186,-1.516,1.290,-0.301,-0.493,0.458,0.436,4.229,-0.452,-0.984,-0.046,-3.738,-0.719,0.914,-0.008,-0.906,-5.325,1.210,-1.953,-0.186,-2.861,-2.476,-1.462,-3.509,0.354,-1.446,1.299,1.827,2.257,1.369,-0.390,-5.035,1.727,0.687,1.886,-0.661,4.937,0.728,1.252,-1.516,-2.476,4.268,-1.760,-1.269,-0.991,3.734,0.067,1.250,-0.797,2.108,3.714,0.254,-2.055,-0.637,0.039,-0.045,-0.496,-0.632,2.720,1.290,-1.462,-1.760,0.441,-1.153,-0.344,0.918,0.632,-2.142,5.068,0.617,-1.116,0.157,0.117,2.392,0.715,-1.137,3.871,-0.442,-0.853,-0.301,-3.509,-1.269,-1.153,-2.055,-2.259,0.098,1.691,-2.796,4.689,0.997,-1.131,-0.843,-0.492,-0.458,-2.694,-2.933,-1.784,-0.461,-4.288,-0.493,0.354,-0.991,-0.344,-2.259,0.493,-3.646,1.097,0.520,4.541,0.145,1.507,0.813,2.212,-0.908,-0.339,1.799,-1.986,-1.579,-3.644,0.458,-1.446,3.734,0.918,0.098,-3.646,1.995,-2.596,-0.875,3.410,1.760,-2.902,7.401,1.684,2.670,0.777,1.947,-4.612,-0.348,0.673,0.436,1.299,0.067,0.632,1.691,1.097,-2.596,-2.166,-1.971,0.358,3.088,-1.378,0.496,-2.630,-1.254,0.602,-0.896,0.397,-2.424,1.343,4.229,1.827,1.250,-2.142,-2.796,0.520,-0.875,-1.971,-4.961,-1.733,3.860,5.474,6.302,-1.867,-4.903,-1.688,2.232,-1.241,-0.649,2.902,-0.452,2.257,-0.797,5.068,4.689,4.541,3.410,0.358,-1.733,-2.818,-2.162,-0.142,-2.984,-1.816,-0.370,0.936,-3.714,0.022,4.018,0.773,-0.984,1.369,2.108,0.617,0.997,0.145,1.760,3.088,3.860,-2.162,-1.902,-1.836,1.450,0.713,2.478,-0.124,-0.238,-2.883,2.260,0.672,-0.046,-0.390,3.714,-1.116,-1.131,1.507,-2.902,-1.378,5.474,-0.142,-1.836,3.361,-1.848,5.883,-1.508,0.620,-0.303,0.900,0.539,-3.033,-3.738,-5.035,0.254,0.157,-0.843,0.813,7.401,0.496,6.302,-2.984,1.450,-1.848,2.041,-5.647,3.901,4.509,-1.011,-0.897,-2.342,4.224,-0.719,1.727,-2.055,0.117,-0.492,2.212,1.684,-2.630,-1.867,-1.816,0.713,5.883,-5.647,-1.809,-0.588,0.702,-0.474,-10.000,1.679,1.504,0.914,0.687,-0.637,2.392,-0.458,-0.908,2.670,-1.254,-4.903,-0.370,2.478,-1.508,3.901,-0.588,1.634,2.083,0.443,4.955,-1.560,-0.820,-0.008,1.886,0.039,0.715,-2.694,-0.339,0.777,0.602,-1.688,0.936,-0.124,0.620,4.509,0.702,2.083,-1.304,1.266,-1.710,2.894,0.378,-0.906,-0.661,-0.045,-1.137,-2.933,1.799,1.947,-0.896,2.232,-3.714,-0.238,-0.303,-1.011,-0.474,0.443,1.266,-0.242,-5.172,-1.479,-2.789,-5.325,4.937,-0.496,3.871,-1.784,-1.986,-4.612,0.397,-1.241,0.022,-2.883,0.900,-0.897,-10.000,4.955,-1.710,-5.172,1.527,-0.069,-2.614,1.210,0.728,-0.632,-0.442,-0.461,-1.579,-0.348,-2.424,-0.649,4.018,2.260,0.539,-2.342,1.679,-1.560,2.894,-1.479,-0.069,0.529,2.341,-1.953,1.252,2.720,-0.853,-4.288,-3.644,0.673,1.343,2.902,0.773,0.672,-3.033,4.224,1.504,-0.820,0.378,-2.789,-2.614,2.341,-0.983,1.729,1.801,-2.015,-0.162,-2.798,-1.274,0.233,-0.062,3.337,0.837,0.707,1.156,-0.759,-1.827,0.383,1.869,-0.396,-2.684,-0.049,-0.756,1.801,0.069,-0.773,-1.734,-2.177,-1.757,-0.671,1.334,0.994,3.247,1.841,-1.010,-1.763,-0.036,2.527,0.574,0.940,6.332,2.341,-0.015,-2.015,-0.773,4.208,-1.174,-3.825,-0.856,3.911,2.498,1.456,-2.641,-0.886,3.564,-1.222,-2.483,1.043,-0.708,0.792,0.994,2.305,0.211,-0.162,-1.734,-1.174,-1.482,-0.919,0.800,1.280,1.866,-1.086,2.921,0.822,-0.675,0.290,0.884,3.201,3.376,-0.655,0.817,-1.112,0.049,-2.798,-2.177,-3.825,-0.919,-5.577,-0.163,1.963,0.598,-2.991,1.556,2.581,-1.961,-2.657,0.971,5.967,-1.238,-3.352,0.571,0.730,-0.480,-1.274,-1.757,-0.856,0.800,-0.163,-1.871,-0.002,0.344,0.659,3.372,0.019,0.172,2.635,2.869,-1.327,-2.192,2.601,-0.007,-0.237,-1.415,0.233,-0.671,3.911,1.280,1.963,-0.002,2.879,1.520,0.851,1.327,0.776,-1.039,6.374,2.674,0.517,0.832,1.829,-3.648,-0.909,0.554,-0.062,1.334,2.498,1.866,0.598,0.344,1.520,0.511,1.096,1.504,-0.566,-3.305,1.308,1.598,-0.114,-0.103,0.925,0.260,-1.283,0.242,3.337,0.994,1.456,-1.086,-2.991,0.659,0.851,1.096,-7.825,-1.425,4.306,1.724,5.694,-0.815,-4.105,-1.213,1.160,3.267,1.425,2.058,0.837,3.247,-2.641,2.921,1.556,3.372,1.327,1.504,-1.425,-8.120,0.679,-2.257,-3.314,1.389,2.217,-0.890,-3.562,-1.919,1.415,-1.385,0.707,1.841,-0.886,0.822,2.581,0.019,0.776,-0.566,4.306,0.679,-1.013,-1.676,-2.042,-1.219,0.500,0.689,-0.210,-0.347,-1.371,-0.424,1.156,-1.010,3.564,-0.675,-1.961,0.172,-1.039,-3.305,1.724,-2.257,-1.676,2.367,-1.358,5.502,0.646,0.449,1.300,0.728,0.086,-1.573,-0.759,-1.763,-1.222,0.290,-2.657,2.635,6.374,1.308,5.694,-3.314,-2.042,-1.358,2.395,-0.307,1.800,3.098,-0.570,0.483,-0.566,3.335,-1.827,-0.036,-2.483,0.884,0.971,2.869,2.674,1.598,-0.815,1.389,-1.219,5.502,-0.307,0.040,2.020,-1.312,-1.329,-6.497,0.127,-2.888,0.383,2.527,1.043,3.201,5.967,-1.327,0.517,-0.114,-4.105,2.217,0.500,0.646,1.800,2.020,-3.811,-0.259,1.282,3.106,-0.491,-1.039,1.869,0.574,-0.708,3.376,-1.238,-2.192,0.832,-0.103,-1.213,-0.890,0.689,0.449,3.098,-1.312,-0.259,1.134,0.258,1.141,-0.100,3.702,-0.396,0.940,0.792,-0.655,-3.352,2.601,1.829,0.925,1.160,-3.562,-0.210,1.300,-0.570,-1.329,1.282,0.258,0.716,-3.850,-2.531,0.348,-2.684,6.332,0.994,0.817,0.571,-0.007,-3.648,0.260,3.267,-1.919,-0.347,0.728,0.483,-6.497,3.106,1.141,-3.850,0.867,3.459,-1.935,-0.049,2.341,2.305,-1.112,0.730,-0.237,-0.909,-1.283,1.425,1.415,-1.371,0.086,-0.566,0.127,-0.491,-0.100,-2.531,3.459,1.550,0.603,-0.756,-0.015,0.211,0.049,-0.480,-1.415,0.554,0.242,2.058,-1.385,-0.424,-1.573,3.335,-2.888,-1.039,3.702,0.348,-1.935,0.603,-4.693,2.352,0.963,-0.119,0.825,2.644,-1.308,-0.270,-0.736,3.021,1.418,-0.075,-0.021,-0.869,-2.089,0.485,0.821,0.369,-1.152,-1.422,-0.899,0.963,1.829,-1.366,-0.537,-0.102,-1.309,2.300,-0.374,4.486,0.299,1.146,0.455,-1.104,-1.600,2.305,0.921,-2.254,5.519,1.482,1.093,-0.119,-1.366,0.917,-0.630,-0.874,1.010,2.822,3.040,3.259,-0.867,-2.493,3.126,-0.671,-1.527,2.024,-0.807,2.763,-2.539,-2.126,-0.184,0.825,-0.537,-0.630,3.185,-1.209,-0.903,3.733,-0.626,-0.782,1.040,0.148,-0.781,-0.527,-1.210,-0.085,0.669,1.476,-1.413,-2.462,-1.984,2.644,-0.102,-0.874,-1.209,-2.788,0.816,0.446,3.789,-1.744,0.103,0.438,-2.004,-2.817,-0.674,4.978,0.409,-3.187,2.409,4.863,2.293,-1.308,-1.309,1.010,-0.903,0.816,0.339,2.444,1.071,0.653,1.843,-1.137,0.092,4.198,-0.158,-2.075,0.648,0.402,2.146,2.205,-0.539,-0.270,2.300,2.822,3.733,0.446,2.444,0.897,2.548,1.889,-1.367,-0.086,-3.170,0.812,0.038,0.220,1.651,1.395,-0.673,0.251,0.777,-0.736,-0.374,3.040,-0.626,3.789,1.071,2.548,0.011,0.610,-1.036,-3.270,-1.681,-0.388,1.033,-0.737,0.269,0.978,0.573,0.261,-1.535,3.021,4.486,3.259,-0.782,-1.744,0.653,1.889,0.610,-10.000,0.324,0.717,2.504,5.734,-1.072,-3.525,-0.768,0.984,4.784,1.455,0.200,1.418,0.299,-0.867,1.040,0.103,1.843,-1.367,-1.036,0.324,-2.796,1.686,-0.040,-0.841,0.151,2.255,-2.222,-0.609,-5.175,1.468,-2.293,-0.075,1.146,-2.493,0.148,0.438,-1.137,-0.086,-3.270,0.717,1.686,-2.282,-1.875,-2.420,-1.349,2.452,0.453,1.263,1.029,-1.448,-0.102,-0.021,0.455,3.126,-0.781,-2.004,0.092,-3.170,-1.681,2.504,-0.040,-1.875,2.657,0.585,4.274,-1.160,1.417,-0.942,-0.364,-1.980,-0.565,-0.869,-1.104,-0.671,-0.527,-2.817,4.198,0.812,-0.388,5.734,-0.841,-2.420,0.585,-0.255,0.554,2.240,2.405,1.625,2.276,0.221,0.641,-2.089,-1.600,-1.527,-1.210,-0.674,-0.158,0.038,1.033,-1.072,0.151,-1.349,4.274,0.554,-1.533,1.199,-1.574,0.252,-1.115,0.000,-1.958,0.485,2.305,2.024,-0.085,4.978,-2.075,0.220,-0.737,-3.525,2.255,2.452,-1.160,2.240,1.199,-2.964,0.384,2.126,3.751,-0.276,-0.120,0.821,0.921,-0.807,0.669,0.409,0.648,1.651,0.269,-0.768,-2.222,0.453,1.417,2.405,-1.574,0.384,1.434,1.128,0.494,-0.991,1.227,0.369,-2.254,2.763,1.476,-3.187,0.402,1.395,0.978,0.984,-0.609,1.263,-0.942,1.625,0.252,2.126,1.128,1.037,-2.984,-0.354,0.117,-1.152,5.519,-2.539,-1.413,2.409,2.146,-0.673,0.573,4.784,-5.175,1.029,-0.364,2.276,-1.115,3.751,0.494,-2.984,1.053,3.517,-2.024,-1.422,1.482,-2.126,-2.462,4.863,2.205,0.251,0.261,1.455,1.468,-1.448,-1.980,0.221,0.000,-0.276,-0.991,-0.354,3.517,0.260,0.896,-0.899,1.093,-0.184,-1.984,2.293,-0.539,0.777,-1.535,0.200,-2.293,-0.102,-0.565,0.641,-1.958,-0.120,1.227,0.117,-2.024,0.896,-4.348,0.485,0.657,-1.120,-0.750,4.366,0.405,-1.524,0.264,3.424,1.053,2.118,-0.001,1.501,-0.623,-3.685,0.951,0.841,-0.789,-1.022,-0.020,0.657,0.186,0.395,-1.980,1.021,-0.033,-1.361,-2.357,7.573,-0.549,-2.068,1.422,0.257,0.513,0.618,2.077,-3.007,5.389,-1.519,-2.406,-1.120,0.395,1.517,-1.137,0.234,-1.000,3.325,1.551,4.841,0.121,-0.909,5.860,1.619,-0.955,2.514,0.361,3.010,-3.199,-2.066,0.522,-0.750,-1.980,-1.137,0.441,-1.974,-0.993,4.279,-2.223,1.195,-0.332,-0.203,-1.306,2.137,-1.155,1.885,1.505,-0.084,0.058,-0.233,-1.462,4.366,1.021,0.234,-1.974,-1.806,2.134,2.365,0.420,-2.947,0.977,2.042,-0.591,-2.505,-0.545,4.512,-0.369,-2.369,2.368,4.834,1.849,0.405,-0.033,-1.000,-0.993,2.134,0.665,1.751,-0.735,-1.921,0.725,2.007,-1.060,2.589,-2.780,-2.068,-0.732,-1.020,5.279,2.915,-2.187,-1.524,-1.361,3.325,4.279,2.365,1.751,-0.137,0.097,0.159,-2.348,1.095,-0.940,1.100,-1.253,-0.132,1.131,-1.594,0.311,0.540,0.365,0.264,-2.357,1.551,-2.223,0.420,-0.735,0.097,-0.273,1.677,0.476,-1.409,0.772,-0.943,-2.752,-0.665,-1.774,1.485,1.041,-0.086,0.028,3.424,7.573,4.841,1.195,-2.947,-1.921,0.159,1.677,-10.000,0.373,1.598,0.875,5.592,-4.987,-5.518,1.045,-0.177,4.523,2.930,1.169,1.053,-0.549,0.121,-0.332,0.977,0.725,-2.348,0.476,0.373,-1.607,0.153,1.818,-0.880,0.728,-0.267,-1.168,-0.168,-8.746,0.902,-0.207,2.118,-2.068,-0.909,-0.203,2.042,2.007,1.095,-1.409,1.598,0.153,1.638,-1.189,0.381,0.291,0.620,1.537,-0.452,1.584,-1.897,0.702,-0.001,1.422,5.860,-1.306,-0.591,-1.060,-0.940,0.772,0.875,1.818,-1.189,1.184,-1.966,1.727,1.087,2.856,0.095,-0.662,-1.688,-0.389,1.501,0.257,1.619,2.137,-2.505,2.589,1.100,-0.943,5.592,-0.880,0.381,-1.966,-2.090,3.986,0.833,-0.343,2.604,4.993,-0.928,1.246,-0.623,0.513,-0.955,-1.155,-0.545,-2.780,-1.253,-2.752,-4.987,0.728,0.291,1.727,3.986,0.888,1.538,-1.037,-0.737,-0.062,-0.941,-2.131,-3.685,0.618,2.514,1.885,4.512,-2.068,-0.132,-0.665,-5.518,-0.267,0.620,1.087,0.833,1.538,-1.754,-2.040,1.968,0.186,1.405,-1.512,0.951,2.077,0.361,1.505,-0.369,-0.732,1.131,-1.774,1.045,-1.168,1.537,2.856,-0.343,-1.037,-2.040,0.011,0.905,-0.219,-2.650,0.115,0.841,-3.007,3.010,-0.084,-2.369,-1.020,-1.594,1.485,-0.177,-0.168,-0.452,0.095,2.604,-0.737,1.968,0.905,0.560,-3.619,0.449,1.005,-0.789,5.389,-3.199,0.058,2.368,5.279,0.311,1.041,4.523,-8.746,1.584,-0.662,4.993,-0.062,0.186,-0.219,-3.619,1.617,1.688,0.631,-1.022,-1.519,-2.066,-0.233,4.834,2.915,0.540,-0.086,2.930,0.902,-1.897,-1.688,-0.928,-0.941,1.405,-2.650,0.449,1.688,-0.097,1.139,-0.020,-2.406,0.522,-1.462,1.849,-2.187,0.365,0.028,1.169,-0.207,0.702,-0.389,1.246,-2.131,-1.512,0.115,1.005,0.631,1.139,-0.356}; 

  if(dist>=2.0 && dist< 3.0) pos = 0;
  else if (dist >= 3.0)
    {
      pos = floor((dist-3.0)*2) + 1;
    } 
  return(PotValues[pos][k*NumberAmiAcids+l]);
}



FoldPotential::~FoldPotential()  
{  
  int j,k;
 
 if(order != (int*) 0) delete[] order;


  for(j=0;j<numberresidues;j++) 
    {    
      for(k=j+1;k<numberresidues;k++) if(Psi[j][k] != (double*)0)  delete[] Psi[j][k];   
      delete[] Psi[j];   
      if (Matrix != (unsigned int**)0)   delete[] Matrix[j]; 
      delete[] Ls[j];          
    }
   
  delete[] Psi;     
  if (Matrix != (unsigned int**)0)  delete[] Matrix;
  delete[] Ls; 
  delete[] sequence;
} 



double FoldPotential::CalculateEnergy(unsigned* assign)
  {
   double epsilon = 1e-200;
   double E;
   int j,k;
 
   E = 0;
  
   for(j=0;j<numberresidues;j++)
    {
     E = E - t*log(Ls[j][assign[j]] + epsilon);
     //cout<<j<<" val "<<Ls[j][assign[j]]<<"  "<<" 1-   "<<E<<endl;
     for(k=j+1;k<numberresidues;k++)
      {
     if(Psi[j][k] != (double*)0)  
        {
	  //   cout<<"--------------------"<<k<<"  "<<log(Psi[j][k][NumberAmiAcids*assign[j]+assign[k]])<<endl;
         E = E -t*log(Psi[j][k][NumberAmiAcids*assign[j]+assign[k]]+epsilon);          
        }  
      }
     //     cout<<" 2-  "<<E<<endl;
    }    
   return E;
  }


unsigned* FoldPotential::FindBestSol(InferenceClass* PotentialInference)
{
 unsigned* BestSol;
 int i;

 BestSol = new unsigned[PotentialInference->num_nodes];
 for (i=0;i<PotentialInference->num_nodes;i++) BestSol[i] = 0;
 PotentialInference->FindBest(BestSol);
 for (i=0;i<PotentialInference->num_nodes;i++) cout<<BestSol[i]<<" ";
 cout<<endl;
 return BestSol;
}



void FoldPotential::BestLoopySolutionMaxConfigurations(unsigned int *Card,double temperature,int MaxIter, unsigned** bestconf, double* energies, int numberconf, int* finalnumberconf)
{
  double** Lambda = 0; //This value is used only by the POTT model. 
  int i,j;
 
  
  InferenceClass*  PotentialInference  = new  InferenceClass(AT_LOOPY,GENERAL, numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);
  //FoldPotentialInference = new InferenceClass(AT_LOOPY,GENERAL, numberresidues ,Matrix,Card,Ls,Psi,Lambda,temperature);
 
  double E = PotentialInference->GetEnergy(sequence);

  int aux  =  PotentialInference->MaxConfigurationsLoopy(bestconf, energies, PotentialInference, numberconf,MaxIter);
 
  // PARA VER SI INICIALIZÓ BIEN, EVALUAR LA ENERGÍA CON POTENCIAL INFERENCE
  *finalnumberconf = aux; 
   delete PotentialInference;
 }



void FoldPotential::BestLoopySolution(unsigned int *Card,double temperature,int MaxIter)
{
  double** Lambda = 0; //This value is used only by the POTT model. 
  unsigned *BestSol;

 
  InferenceClass PotentialInference(AT_LOOPY,GENERAL, numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);
  PotentialInference.SetLoopy(MaxIter,MAX,SEQUENTIAL);
  PotentialInference.CreateAlgorithm();
  PotentialInference.MakeInferenceAlgorithm();

  BestSol = FindBestSol(&PotentialInference);
  delete[] BestSol;
  //delete PotentialInference;
 }


void FoldPotential::BestGBPSolution(unsigned int *Card,double temperature,int MaxIter)
{
  int i,j,k,auxpos,numRegs;
  int* regionsizes;
  int** AllInitRegions;
  unsigned *BestSol;
 
  /*
  double** Lambda = 0; //This value is used only by the POTT model.
  InferenceClass PotentialInference(AT_GBP,GENERAL,numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);

  cout<<"Inference initialized "<<endl;

  //PotentialInference = new InferenceClass(AT_GBP,GENERAL,numberresidues,Matrix,Card,Ls,Psi,Lambda,temperature);
  */
  MaxSubgraph = new maximalsubgraph(numberresidues,Matrix,15,5000,order);
  MaxSubgraph->FindAllCliques();
    
  numRegs = MaxSubgraph->NumberCliques;
  //regionsizes = new int[numRegs];
  //AllInitRegions = new int*[numRegs];

  

  int meansize = 0;
  int maxsize = 0;
  
  for(i=0;i<numRegs;i++)   //The set of regions is constructed
   {
     if(MaxSubgraph->CliquesSizes[i]>maxsize) maxsize =  MaxSubgraph->CliquesSizes[i];
      meansize += MaxSubgraph->CliquesSizes[i];

     //regionsizes[i] =  MaxSubgraph->CliquesSizes[i];
     /*
     AllInitRegions[i] = new int[regionsizes[i]];
     for(j=0;j<regionsizes[i];j++) AllInitRegions[i][j] = MaxSubgraph->ListCliques[i]->vars[j];
      
     for(j=0;j<regionsizes[i]-1;j++)  //The cliques are ordered, this is needed by inference
       for(k=j+1;k<regionsizes[i];k++)
	 { 
           if(AllInitRegions[i][j]>AllInitRegions[i][k])
             {
	       auxpos = AllInitRegions[i][j];
               AllInitRegions[i][j] = AllInitRegions[i][k];
               AllInitRegions[i][k] = auxpos;
             }
         }
     */
   } 
  cout<<allfiledetails<<" "<<numberresidues<<" "<<numRegs<<" "<<maxsize<<" "<<double(1.0*meansize/numRegs)<<endl;
  //PotentialInference.SetGBP(MaxIter,MAX,0.5,1,numRegs,regionsizes,AllInitRegions);
  //PotentialInference.CreateAlgorithm();
  //PotentialInference.MakeInferenceAlgorithm();
 
  //BestSol  = FindBestSol(&PotentialInference); 

  //delete[] BestSol;
   delete MaxSubgraph;
  //delete[] regionsizes;
  //for(i=0;i<numRegs;i++)  delete[] AllInitRegions;
  //delete PotentialInference;
 }





/*  *********************************   SNP CLASS ********************************** */



SNPs::SNPs(FILE* streamPair, FILE* streamTriple)  
{    
 
  SNPsNames = new char*[MaxnumberSNPs];
  for (int i=0;i<MaxnumberSNPs;i++) SNPsNames[i] = new char[40];
  ReadFilePairSNPs(streamPair);
  ReadFileTripleSNPs(streamTriple);   
  OrderTags();
}

void SNPs::OrderTags()
{
  int i,j,k;
   IndexOrderedTags = new int[numberSNPs];
   for(i=0;i<numberSNPs;i++)  IndexOrderedTags[i] = i;
   for(i=0;i<numberSNPs-1;i++)
    for(j=i+1;j<numberSNPs;j++)
      {
	if(strcmp(SNPsNames[IndexOrderedTags[j]],SNPsNames[IndexOrderedTags[i]])<0)
	{
          k = IndexOrderedTags[i];
          IndexOrderedTags[i] = IndexOrderedTags[j];
          IndexOrderedTags[j] = k;
	}
      }
} 

void SNPs::InitBasicPairStructures(int nSNPs)
  {  
  
    ntagging = new int[nSNPs];
    memset(ntagging, 0, sizeof(int)*nSNPs); 
 
  }


void SNPs::InitBasicTripleStructures(int nSNPs)
  {  
    npairtagging = new int[nSNPs];
    npairtagged = new int[nSNPs];
    memset(npairtagging, 0, sizeof(int)*nSNPs);
    memset(npairtagged, 0, sizeof(int)*nSNPs);
  }

void SNPs::DeleteBasicPairStructures()
  {  
    delete[] ntagging;    
  }


void SNPs::DeleteBasicTripleStructures()
  {  
     delete[] npairtagging;
     delete[] npairtagged;
  }


void SNPs::ReadFileTripleSNPs(FILE *stream)  
{
  char snp1[40];
  char snp2[40];
  char snp3[40];
  int found1,found2,found3;
  int pos_snp1, pos_snp2, pos_snp3;
  double  valcorr;
  int i,j,k,l; 

 

 InitBasicTripleStructures(numberSNPs);

 
 ntriples = 0;
  while(fscanf(stream, "%s %s %s %lg", snp1,snp2,snp3,&valcorr) != EOF)
  {
   
   i = 0;
    found1 = 0; found2 = 0; found3 = 0;

     
    while( (found1+found2+found3) !=3 && i<numberSNPs)
      { 
        if(!found1)
	{
         found1 = strcmp(SNPsNames[i],snp1)==0;
	 pos_snp1 = found1*i;
        }

        if(!found2)
	{
         found2 = strcmp(SNPsNames[i],snp2)==0;
	 pos_snp2 = found2*i;
        }

        if(!found3)
	{
         found3 = strcmp(SNPsNames[i],snp3)==0;
	 pos_snp3 = found3*i;
        }

        i++;

      } 



     l=0;
     found1 = 0;
        while (!found1 && l<ntagging[pos_snp1])
         {
	   found1 = tagged[pos_snp1][l]==pos_snp3;
	   l++; 
         }
  if(found1==0)
      { 
        l=0;
        while (!found1 && l<ntagging[pos_snp2])
         {
	   found1 = tagged[pos_snp2][l]==pos_snp3;
	   l++; 
         }
      }
	       
  if(found1==0)
   {
    npairtagging[pos_snp1]++; 
    npairtagging[pos_snp2]++;
    npairtagged[pos_snp3]++;

    //   cout<<i<<" "<<found1<<" "<<found2<<"  "<<found3<<" "<<pos_snp1<<" "<<pos_snp2<<" "<<pos_snp3<<" "<<
    // npairtagging[pos_snp1]<<" "<<npairtagging[pos_snp2]<<" "<<npairtagged[pos_snp3]<<endl;
    ntriples++;
   }
  }




  
  tagging_as_pair = new int*[numberSNPs];
  tagged_by_pair  = new int*[numberSNPs];
  tagging_as_pair_corr = new double*[numberSNPs];
  //tagged_by_pair_corr  = new double*[numberSNPs];

  
  for (i=0;i<numberSNPs;i++)
    {
       if (npairtagging[i]>0) 
	{
         tagging_as_pair[i] =  new int[npairtagging[i]];   
         tagged_by_pair[i] =   new int[npairtagging[i]];    
         tagging_as_pair_corr[i] =  new double[npairtagging[i]];
        }
    }
 
  fseek(stream,0,SEEK_SET); //Beginning of the file;

  DeleteBasicTripleStructures();
  InitBasicTripleStructures(numberSNPs);


  while(fscanf(stream, "%s %s %s %lg", snp1,snp2,snp3,&valcorr) != EOF)
  {
    //cout<<i<<" "<<snp1<<" "<<snp2<<" "<<snp3<<" "<<valcorr<<endl;
    i = 0;
    found1 = 0; found2 = 0; found3 = 0;

     
    while((found1+found2+found3) !=3 && i<numberSNPs)
      { 
        if(!found1)
	{
         found1 = strcmp(SNPsNames[i],snp1)==0;
	 pos_snp1 = found1*i;
        }
        if(!found2)
	{
         found2 = strcmp(SNPsNames[i],snp2)==0;
	 pos_snp2 = found2*i;
        }
        if(!found3)
	{
         found3 = strcmp(SNPsNames[i],snp3)==0;
	 pos_snp3 = found3*i;
        }
	i++;   
   }
    
     l=0;
     found1 = 0;
        while (!found1 && l<ntagging[pos_snp1])
         {
	   found1 = tagged[pos_snp1][l]==pos_snp3;
	   l++; 
         }
  if(found1==0)
      { 
        l=0;
        while (!found1 && l<ntagging[pos_snp2])
         {
	   found1 = tagged[pos_snp2][l]==pos_snp3;
	   l++; 
         }
      }

if(found1==0)
  {
    tagging_as_pair[pos_snp1][npairtagging[pos_snp1]] = pos_snp2;
    tagging_as_pair[pos_snp2][npairtagging[pos_snp2]] = pos_snp1;
    tagged_by_pair[pos_snp2][npairtagging[pos_snp2]] = pos_snp3;
    tagged_by_pair[pos_snp1][npairtagging[pos_snp1]] = pos_snp3;

    tagging_as_pair_corr[pos_snp1][npairtagging[pos_snp1]] = valcorr;
    tagging_as_pair_corr[pos_snp2][npairtagging[pos_snp2]] = valcorr;
   

    npairtagging[pos_snp1]++; 
    npairtagging[pos_snp2]++;
    npairtagged[pos_snp3]++;

    //cout<<i<<" "<<found1<<" "<<found2<<"  "<<found3<<" "<<pos_snp1<<" "<<pos_snp2<<" "<<pos_snp3<<" "<<
    //npairtagging[pos_snp1]<<" "<<npairtagging[pos_snp2]<<" "<<npairtagged[pos_snp3]<<endl;
   }
  }


  numberSelfTagged = 0; 
   
  for (i=0;i<numberSNPs;i++)
    {
      if (ntagging[i]==1 && npairtagged[i]==0) 
	{
         numberSelfTagged++;  
         //cout<<" ("<<i<<","<<ntagging[i]<<") "<<endl;
        }
     }


  numberSNPsNeedingTag = numberSNPs - numberSelfTagged;
  SNPsNeedingTag = new int[numberSNPsNeedingTag];

  if (numberSelfTagged>0)  SelfTagged = new int[numberSelfTagged];
  else  SelfTagged  = (int*)0;

  //cout<<numberSelfTagged<<" "<<numberSNPsNeedingTag<<endl;

   j=0; k=0;

   for (i=0;i<numberSNPs;i++)
    {
      if (ntagging[i]==1 && npairtagged[i]==0) SelfTagged[j++] = i;
      else SNPsNeedingTag[k++] = i;
    }
    
   

}




void SNPs::ReadFilePairSNPs(FILE *stream)  
{
  char snp1[40];
  char snp2[40];
  
  
  int found1,found2;
  int pos_snp1, pos_snp2;
  double  valcorr;
  int i,j,k; 

 
 InitBasicPairStructures(MaxnumberSNPs);
 
 numberSNPs = 0;
 npairs = 0;

  while(fscanf(stream, "%s %s %lg", snp1,snp2,&valcorr) !=EOF)
  {
    i = 0;
    //cout<<i<<" "<<snp1<<" "<<snp2<<" "<<valcorr<<endl;
    found1 = 0; found2 = 0; 

    while((found1+found2) !=2 && i<numberSNPs)
      { 
        if(!found1)
	{
         found1 = strcmp(SNPsNames[i],snp1)==0;
	 pos_snp1 = found1*i;
        }
        if(!found2)
	{
         found2 = strcmp(SNPsNames[i],snp2)==0;
	 pos_snp2 = found2*i;
        }      
        i++;
      }
       

    if(!found1) 
      {             
        SNPsNames[numberSNPs][0] = 0;
        strcat(SNPsNames[numberSNPs],snp1);
        pos_snp1 = numberSNPs;
        numberSNPs++;        
      }

     if(!found2 && strcmp(snp1,snp2)!=0) 
      {
       
        //SNPsNames[numberSNPs] = new char[20];
        SNPsNames[numberSNPs][0] = 0;
        strcat(SNPsNames[numberSNPs],snp2);
       	pos_snp2 = numberSNPs;
        numberSNPs++; 
      }
     else if(!found2 && strcmp(snp1,snp2)==0)
       {
	 pos_snp2 = pos_snp1;
       }

    ntagging[pos_snp1]++; 
    if(pos_snp1 != pos_snp2)  ntagging[pos_snp2]++; 
    //cout<<numberSNPs<<" "<<found1<<" "<<found2<<"  "<<pos_snp1<<" "<<pos_snp2<<" "<<strcmp(snp1,snp2)<<endl;
    npairs++; 
 }


  tagged  = new int*[numberSNPs];
  tagged_corr  = new double*[numberSNPs];
  




 for (i=0;i<numberSNPs;i++)
    {
      tagged[i] = new int[ntagging[i]];
      tagged_corr[i] = new double[ntagging[i]]; 
    }
  

  DeleteBasicPairStructures();
  InitBasicPairStructures(numberSNPs);
 

   
  fseek(stream,0,0); //Beginning of the file;
 

  while(fscanf(stream, "%s %s %lg", snp1,snp2,&valcorr) != EOF)
  {
   
    i = 0;
    found1 = 0; found2 = 0;



    while((found1+found2) !=2 && i<numberSNPs)
      { 
        if(!found1)
	{
         found1 = strcmp(SNPsNames[i],snp1)==0;
	 pos_snp1 = found1*i;
        }
        if(!found2)
	{
         found2 = strcmp(SNPsNames[i],snp2)==0;
	 pos_snp2 = found2*i;
        }
        i++;
       }
    
    

    tagged[pos_snp1][ntagging[pos_snp1]] = pos_snp2;
    tagged[pos_snp2][ntagging[pos_snp2]] = pos_snp1;
    tagged_corr[pos_snp1][ntagging[pos_snp1]] = valcorr;
    tagged_corr[pos_snp2][ntagging[pos_snp2]] = valcorr;




    ntagging[pos_snp1]++; 
    if(pos_snp1 != pos_snp2)  ntagging[pos_snp2]++; 

    //cout<<i<<" "<<pos_snp1<<" "<<pos_snp2<<" "<<ntagging[pos_snp1]<<" "<<ntagging[pos_snp2]<<endl;    
  }
}


void SNPs::SaveTags(FILE *stream,unsigned* solution) 
  {

    int i,k;
  for (i=0;i<numberSNPs;i++) 
   {
   
    for (k=0;k<numberSelfTagged;k++)
      if(SelfTagged[k]==IndexOrderedTags[i]) fprintf(stream, "%s\n", SNPsNames[SelfTagged[k]]);
          
    for (k=0;k<numberSNPsNeedingTag;k++) 
   {
     if(solution[k]==1)
       {
          if(SNPsNeedingTag[k]==IndexOrderedTags[i]) 
            fprintf(stream, "%s\n", SNPsNames[SNPsNeedingTag[k]]);
 
       }
   }
  }
 
 }


int SNPs::MultiCalculateNumberTaggedSNPs(unsigned* solution)
  {
    int i,j,covert,pick_one;
    unsigned* auxSNPs;
    int result;
 
        
  auxSNPs = new unsigned[numberSNPs];
  memset(auxSNPs, 0, sizeof(unsigned)*numberSNPs); 

 

 covert = 0;  //Number of SNPs currently tagged

 
 for (i=0;i<numberSNPs;i++)
   {
      if(solution[i]==1) 
       {
        auxSNPs[i] = 2;
        covert++;    //The tagging SNPS are covered by themselves
       }
   }

 
 i = 0; 
 while(covert<numberSNPs && i<numberSNPs)  //Those directly tagged by a single SNP
   {
     if(solution[i]==1)
       {
           j=0;
	   while (covert<numberSNPs && j<ntagging[i])
            {
	      // cout<<i<<" "<<j<<" "<<SNPsNames[j]<<" "<<tagged[i][j]<<" "<<covert<<endl; 
	     if(auxSNPs[tagged[i][j]]==0)
	       {
                 auxSNPs[tagged[i][j]] = 1;  
		 covert++;
               } 
	     j++;
            }
       }
      i++;

   } 
   

  i = 0;
  while(covert<numberSNPs && i<numberSNPs)  //Those tagged by a pair of SNPs
   {
      if(solution[i]==1)
       {
           j=0;        
	   while (covert<numberSNPs && j<npairtagging[i])
            {
	     if(auxSNPs[tagging_as_pair[i][j]]==2 && auxSNPs[tagged_by_pair[i][j]]==0 )
	       {
                 auxSNPs[tagged_by_pair[i][j]]=1;
                 covert++;
               } 
	     j++;
            }     

         }          
	 i++;                 
   }  

  

 result = 0;
 
  // The result is the number of tagged SNPs 
  for (i=0;i<numberSNPs;i++) result += (auxSNPs[i]>0);


  delete[] auxSNPs; 
  return result;
}



int SNPs::CalculateNumberTaggedSNPs(unsigned* solution)
  {
    int i,j,k,covert,pick_one;
    unsigned* auxSNPs;
    int result;
 
        


 auxSNPs = new unsigned[numberSNPs];
 memset(auxSNPs, 0, sizeof(unsigned)*numberSNPs); 

 

 covert = numberSelfTagged;         //Those that are not related to any other SNP are fixed
    


 for (k=0;k<numberSNPsNeedingTag;k++)
   {
     i = SNPsNeedingTag[k];
     if(solution[k]==1) 
       {
        auxSNPs[i] = 2;
        covert++;    //The tagging SNPS are covered by themselves
       }
   }

 
 k = 0; 
 while(covert<numberSNPs && k<numberSNPsNeedingTag)  //Those directly tagged by a single SNP
   {
     i = SNPsNeedingTag[k];
     if(solution[k]==1)
       {
           j=0;
	   while (covert<numberSNPs && j<ntagging[i])
            {
	      // cout<<i<<" "<<j<<" "<<SNPsNames[j]<<" "<<tagged[i][j]<<" "<<covert<<endl; 
	     if(auxSNPs[tagged[i][j]]==0)
	       {
                 auxSNPs[tagged[i][j]] = 1;  
		 covert++;
               } 
	     j++;
            }
       }
      k++;

   } 
   

  k = 0;
  while(covert<numberSNPs && k<numberSNPsNeedingTag)  //Those tagged by a pair of SNPs
   {
     i = SNPsNeedingTag[k];
     if(solution[k]==1)
       {
           j=0;        
	   while (covert<numberSNPs && j<npairtagging[i])
            {
	     if(auxSNPs[tagging_as_pair[i][j]]==2 && auxSNPs[tagged_by_pair[i][j]]==0 )
	       {
                 auxSNPs[tagged_by_pair[i][j]]=1;
                 covert++;
               } 
	     j++;
            }     

         }          
	 k++;                 
   }  



   while(covert<numberSNPs)    //Those not covered yet are identified and randomly set to be tagging SNPs
    {
     pick_one = randomint(numberSNPs-covert)+1;
    
     j = 0;
     k = 0;
     while(k<numberSNPsNeedingTag && j<pick_one)
      {
       i = SNPsNeedingTag[k];
       if(auxSNPs[i] == 0) j++;
       k++;
      }
     // At this point i is the new tagging SNP
     auxSNPs[i] = 2;  
     solution[k-1] = 1; //The solution is modified
     covert++;
 
     k = 0; 

 while(covert<numberSNPs && k<numberSNPsNeedingTag)  //Those directly tagged by a single SNP
   {
           j=0;
	   while (covert<numberSNPs && j<ntagging[i])
            {
	     if(auxSNPs[tagged[i][j]]==0)
	       {
                 auxSNPs[tagged[i][j]] = 1;  
		 covert++;
               } 
	     j++;
            }       
      k++;

   } 
  
   
  k = 0;
  while(covert<numberSNPs && k<numberSNPsNeedingTag)  //Those tagged by a pair of SNPs
   {
          j=0;        
	   while (covert<numberSNPs && j<npairtagging[i])
            {
	     if(auxSNPs[tagging_as_pair[i][j]]==2 && auxSNPs[tagged_by_pair[i][j]]==0 )
	       {
                 auxSNPs[tagged_by_pair[i][j]]=1;
                 covert++;
               } 
	     j++;
            }         
	 k++;                 
   }  

  

    }  
  
  delete[] auxSNPs;  
  
  result = numberSelfTagged;

  for (k=0;k<numberSNPsNeedingTag;k++) result += solution[k];

 
   return result;
}


void SNPs::CreateMatrix(unsigned** matrix)
  {
    // The matrix has numberSNPsNeedingTag^2 elements
    int i,j,k;
    unsigned* auxSNPs;
   
        

    auxSNPs = new unsigned[numberSNPs]; // Maps the position in the needingtag array
                                        //  to the original SNPs positions
    memset(auxSNPs, numberSNPs+1, sizeof(unsigned)*numberSNPs); 
    for (k=0;k<numberSNPsNeedingTag;k++) auxSNPs[SNPsNeedingTag[k]] = k; 
 


 for (k=0;k<numberSNPsNeedingTag;k++)
   {

     i = SNPsNeedingTag[k];
     for (j=0; j<ntagging[i];j++)
        {
          matrix[k][auxSNPs[tagged[i][j]]] = 1;  
        } 

      for (j=0; j<npairtagging[i];j++) 
	{

	  matrix[k][auxSNPs[tagged_by_pair[i][j]]]=1;
          matrix[auxSNPs[tagged_by_pair[i][j]]][k]=1;
	 
          if(auxSNPs[tagging_as_pair[i][j]] < numberSNPs+1) // To be sure it is a needingtag SNP
	    {   
              matrix[auxSNPs[tagging_as_pair[i][j]]][auxSNPs[tagged_by_pair[i][j]]] = 1;
              matrix[auxSNPs[tagged_by_pair[i][j]]][auxSNPs[tagging_as_pair[i][j]]] = 1;
            } 
        }

   } 	
 /*
   for (k=0;k<80;k++)
    {
      for (j=0;j<80;j++) cout<<matrix[k][j]<<" ";
      cout<<endl;
    }
 */      
 delete[] auxSNPs;
} 



SNPs::~SNPs()  
{   
  int i;

  DeleteBasicPairStructures();
  
  /*

for (i=0;i<10;i++)
    {
  for (j=0;j<npairtagging[i];j++)
    {
      cout<<i<<" "<<j<<" "<<npairtagging[i]<<" "<<tagging_as_pair[i][j]<<" "<<tagged_by_pair[i][j]<<endl;
    }
    }
  */


  for (i=0;i<numberSNPs;i++)
    {
      //cout<<i<<" "<<npairtagging[i]<<endl;
  
      if (npairtagging[i]>0) 
	{
         delete[] tagging_as_pair[i];      
         delete[] tagged_by_pair[i];    
         delete[] tagging_as_pair_corr[i];
        }  
      
      
      delete[] tagged[i];
      delete[] tagged_corr[i]; 
      
       
    }
 


  DeleteBasicTripleStructures();


  for (int i=0;i<MaxnumberSNPs;i++) delete[]  SNPsNames[i];

  if(numberSelfTagged>0) delete[] SelfTagged;

  delete[] SNPsNeedingTag;
  delete[] tagged;
  delete[] tagging_as_pair;
  delete[] tagged_by_pair;
  delete[] tagged_corr;
  delete[] tagging_as_pair_corr;
  delete[] IndexOrderedTags;
  delete[]  SNPsNames;
 
}
    

/* *********************  SNPSets Class ************************** */
/* *****************************************************************/


SNPSets::SNPSets(int nSNPSets)  
{    
 
  numberSNPSets = nSNPSets;
  TheSNPSets = new SNPs*[numberSNPSets];
  TheSNPIndex = new int*[numberSNPSets];
  BackSNPIndex = new int*[numberSNPSets];
  SNPsNames = new char*[MaxnumberSNPs];
  for (int i=0;i<MaxnumberSNPs;i++) SNPsNames[i] = new char[60];
 
}

SNPSets::~SNPSets()  
{   
  int i;

  for (i=0;i<numberSNPSets;i++) 
    {
     delete TheSNPSets[i];
     delete[] BackSNPIndex[i];
     delete[] TheSNPIndex[i];
   }
  for (int i=0;i<MaxnumberSNPs;i++) delete[]  SNPsNames[i];
  delete[]  TheSNPSets;
  delete[] SNPsNames; 
  delete[] TheSNPIndex;
  delete[] BackSNPIndex;


}

void SNPSets::Initialize()  
{    

  char snp1[50];
  int found;
  int i,j,k;

  numberSNPs = 0;
  for (i=0;i<numberSNPSets;i++)
    {
      TheSNPIndex[i] = new int[MaxnumberSNPs];
      BackSNPIndex[i] = new int[TheSNPSets[i]->numberSNPs];

          
      for (j=0;j<TheSNPSets[i]->numberSNPs;j++)
	{
         strcpy(snp1,TheSNPSets[i]->SNPsNames[j]);
       
         if(i==0)
	   {
             //SNPsNames[numberSNPs] = 0;
               
	     strcpy(SNPsNames[numberSNPs],snp1);
             //cout<<"Steps "<<i<<"  "<<j<<" "<<SNPsNames[numberSNPs]<<" "<<numberSNPs<<endl;
             TheSNPIndex[i][numberSNPs] = j;
             BackSNPIndex[i][j] = numberSNPs;
             numberSNPs++;
           
           }
         else
	   {
	     k =0;
             found = 0;
             while(found!=1 && k<numberSNPs)
	       {
		 found = (strcmp(SNPsNames[k],snp1)==0);
                 k++; 
               }
             if(found) 
               {
                TheSNPIndex[i][k-1] = j;
                BackSNPIndex[i][j] = k-1;
               }
             else
               {            
                strcpy(SNPsNames[numberSNPs],snp1);
                //cout<<"Steps "<<i<<"  "<<j<<" "<<SNPsNames[numberSNPs]<<" "<<numberSNPs<<endl;
                TheSNPIndex[i][numberSNPs] = j;
                BackSNPIndex[i][j] = numberSNPs;
                numberSNPs++;
               }

           }
	}
    }

}


void SNPSets::SaveTags(FILE *stream,unsigned* solution) 
  {

    int i,k;
  for (i=0;i<numberSNPs;i++) 
   {  
     if(solution[i]==1)
            fprintf(stream, "%s\n", SNPsNames[i]);
 
          
   }
  }



void SNPSets::CreateMatrix(unsigned** matrix)
  {
    // The matrix has numberSNPs^2 elements
    int i,j,k,l;
    unsigned* auxSNPs;
   

  for (l=0;l<numberSNPSets;l++)  {


    auxSNPs = new unsigned[TheSNPSets[l]->numberSNPs]; // Maps the position in the needingtag array
                                        //  to the original SNPs positions
    memset(auxSNPs, TheSNPSets[l]->numberSNPs+1, sizeof(unsigned)*TheSNPSets[l]->numberSNPs); 
    
  for (k=0;k<TheSNPSets[l]->numberSNPsNeedingTag;k++) auxSNPs[TheSNPSets[l]->SNPsNeedingTag[k]] = k; 
 

 for (k=0;k<TheSNPSets[l]->numberSNPsNeedingTag;k++)
   {

     i = TheSNPSets[l]->SNPsNeedingTag[k];
     for (j=0; j<TheSNPSets[l]->ntagging[i];j++)
        {
          matrix[BackSNPIndex[l][k]][BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagged[i][j]]]] = 1;  
        } 

      for (j=0; j<TheSNPSets[l]->npairtagging[i];j++) 
	{

	  matrix[BackSNPIndex[l][k]][BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagged_by_pair[i][j]]]]=1;
          matrix[BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagged_by_pair[i][j]]]][BackSNPIndex[l][k]]=1;
	 
          if(auxSNPs[TheSNPSets[l]->tagging_as_pair[i][j]] < TheSNPSets[l]->numberSNPs+1) // To be sure it is a needingtag SNP
	    {   
              matrix[BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagging_as_pair[i][j]]]][BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagged_by_pair[i][j]]]] = 1;
              matrix[BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagged_by_pair[i][j]]]][BackSNPIndex[l][auxSNPs[TheSNPSets[l]->tagging_as_pair[i][j]]]] = 1;
            } 
        }

   } 	
 
 delete[] auxSNPs;
    }
} 




 void SNPSets::MultiCalculateNumberTaggedSNPs(unsigned* solution, double* Results)
  {
    int j,l;
    unsigned* auxsolution;
    int numberones;
    double auxresult;

    numberones = 0;

  auxsolution = new unsigned[numberSNPs];      
 
  for(j=0;j<numberSNPs;j++) numberones+=solution[j];
  Results[0] = numberSNPs-numberones;

  for (l=0;l<numberSNPSets;l++)
   {
     
     for(j=0;j<TheSNPSets[l]->numberSNPs;j++)
       {
         if(solution[BackSNPIndex[l][j]]==1) auxsolution[j]=1;
         else auxsolution[j]=0;             
       }
	  
	   auxresult = (1.0*TheSNPSets[l]->MultiCalculateNumberTaggedSNPs(auxsolution));

	 Results[l+1] = auxresult/(1.0*TheSNPSets[l]->numberSNPs);

	 cout<<l<<" "<<TheSNPSets[l]->numberSNPs<<"  "<<auxresult<<" "<<Results[l+1]<<endl;         
   } 
  
  delete  auxsolution;

}







