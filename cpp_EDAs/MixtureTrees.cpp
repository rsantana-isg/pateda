#include "auxfunc.h"  
#include "Popul.h"
//#include "Treeprob.h"
#include "MixtureTrees.h"
#include "AbstractTree.h"
#include "FDA.h"
#include <math.h>
#include <time.h>
#include <iostream>
#include <fstream>  

using namespace std;

extern FILE *outfile;

	MixtureTrees::MixtureTrees(int Nvar,int NTrees,int NPoint, int initLamb, int MaxLsteps,double MaxMProb, double Smooth, double SI, int prior)
	{
        int i;

		NumberVars = Nvar;
		NumberTrees = NTrees;
		NumberPoints = NPoint;
		InitialLambda = initLamb;
		LearningSteps = MaxLsteps-1;
                MaxMixtProb = MaxMProb;
                Smoothing = Smooth;
                SelInt = SI;
                Prior = prior;
		EveryTree = new AbstractProbModel* [NumberTrees];
		lambdavalues = new double[NumberTrees];
                TreeProb = new double[NumberTrees];
		LikehoodValues = new double[LearningSteps]; 
		Probabilities = new double* [NumberTrees+1];
		for(i=0; i < NumberTrees; i++)
                  {
                      TreeProb[i] = 0;
                      lambdavalues[i] = 0;
                   }
                  

                if(Smoothing>0) Smooth_alpha = new double[NumberTrees];
	}


MixtureTrees::~MixtureTrees()
	{
        int i;
		delete[] EveryTree;
		delete[] lambdavalues;
                delete[] TreeProb;
		delete[] LikehoodValues; 
                for(i=0; i < NumberTrees+1; i++) 
		    if (Probabilities[i] != (double*)0) delete[] Probabilities[i];
   	        delete[] Probabilities;
                if(Smoothing>0) delete[] Smooth_alpha;
	}


void MixtureTrees::SetNpoints(int NPoints, double* pvect)
{
  int i,j;
     CNumberPoints = NPoints;
     for(i=0; i < NumberTrees+1; i++)
       {
         Probabilities[i] = new double[CNumberPoints+1];
         for (j=0; j < CNumberPoints+1; j++) Probabilities[i][j] = 0;
       }
  PopProb = pvect;
}

void MixtureTrees::RemoveProbabilities()
{
  int i;
    for(i=0; i < NumberTrees+1; i++)
       {
	   delete[] Probabilities[i];
           Probabilities[i]  = (double*)0;
       }
}


void MixtureTrees::Calculate_Smooth_alpha()
{
  int i;

  if (Smoothing>0) 
    {
     for(i=0; i < NumberTrees; i++)
       {
          if (Probabilities[i][CNumberPoints]>0) Smooth_alpha[i] = (Smoothing/Probabilities[i][CNumberPoints]);
          else Smooth_alpha[i] = 0;
       }
    }
}


double MixtureTrees::Calculate_Best_Prior(int N, int dim, double penalty)
{
  double prior;

   prior  = (penalty*N)/(NumberVars*pow(2,dim-1));
   return prior; 

}

void MixtureTrees::RandomLambdas()	
{ 
   	int i;
        double aux;
        aux = 0;

	for(i=0; i < NumberTrees; i++) 
         {
          lambdavalues[i] = myrand();
          aux += lambdavalues[i];
         }
	for(i=0; i < NumberTrees; i++) lambdavalues[i] /= aux;
}


void MixtureTrees::UniformLambdas()	
{ // Initially all the  Trees have the same lambda coefficient
   	int i;
	for(i=0; i < NumberTrees; i++) lambdavalues[i] = (1) / double (NumberTrees);
	
}

/*
void MixtureTrees::FitnessPropLambdas()	
{ // Initially all the  Trees have lambda coefficient linearly proportional to fitness
   	int i;
	double TotFitness;
        
        TotFitness=0;
	for(i=0; i < NumberPoints; i++) 
	{
	    lambdavalues[TreeOrigin[i]] += AllPoints->Evaluations[i];
        TotFitness += AllPoints->Evaluations[i];
    }
	
	for(i=0; i < NumberTrees; i++) 
		if (TotFitness>0) lambdavalues[i] /= TotFitness;
		else lambdavalues[i] = 0;
}
*/

void MixtureTrees::FindGammaValues()
{
	int i,j;
	for(i=0; i < NumberTrees+1; i++) 
		Probabilities[i][CNumberPoints] = 0;
     
	//cout<<"Numb "<<CNumberPoints<<endl;
	for(i=0; i < NumberTrees; i++) 
	{
  	 for(j=0; j < CNumberPoints; j++) 
	 {
	 // Equation Numb (3.17) in Melina's Thesis
	 if(Probabilities[NumberTrees][j]>0) Probabilities[i][j] = (Probabilities[i][j]/Probabilities[NumberTrees][j])*(NumberPoints*PopProb[j]);
	 else Probabilities[i][j] = 0;
      Probabilities[i][CNumberPoints] +=  Probabilities[i][j];
     }  
    }
	
/*		
  for(i=0; i <NumberTrees+1; i++) 
       {
	 cout<< TreeProb[i]<<" "<< MeanMixtProb<<endl;
      for(j=0; j <CNumberPoints+1; j++) 
	 {
           cout<<Probabilities[i][j]<<" ";
         }
	cout<<endl;
       }
*/	
}


void MixtureTrees::UpdateTotProb()
{
	int i,j;
	for(i=0; i < CNumberTrees+1; i++) Probabilities[i][CNumberPoints] = 0;
     
	for(i=0; i < NumberTrees; i++) 
	{
         Probabilities[i][CNumberPoints] = 0;
  	 for(j=0; j < CNumberPoints; j++) Probabilities[i][CNumberPoints] +=  Probabilities[i][j];                Probabilities[CNumberTrees][CNumberPoints] += Probabilities[i][CNumberPoints];
        }	
}


void MixtureTrees::NormalizeProb()
{
 
	int i,j;
    for(i=0; i < CNumberPoints; i++) 
	Probabilities[NumberTrees][i] = 0;
	
    for(i=0; i < NumberTrees; i++) 
	{
  	 for(j=0; j < CNumberPoints; j++) 
	 {
        	 // Equation Numb (3.18) in Melina's Thesis
           if( Probabilities[i][CNumberPoints]>0)
	     Probabilities[i][j] /= Probabilities[i][CNumberPoints];
	  else Probabilities[i][j] = 0;
          Probabilities[NumberTrees][j] += Probabilities[i][j];
         }     
        }
           
  for(i=0; i < CNumberPoints; i++) 
     {
		Probabilities[NumberTrees][i] /= NumberTrees ;
		
     }
     
/*  
cout<<"Normalized"<<endl;

  for(i=0; i <NumberTrees+1; i++) 
       {
	 cout<< TreeProb[i]<<" "<< MeanMixtProb<<endl;
      for(j=0; j <CNumberPoints+1; j++) 
	 {
           cout<<Probabilities[i][j]<<" ";
         }
	cout<<endl;
       }
*/  
  
}

void MixtureTrees::PrintProb()
{
	int i,j;
    
   
    for(i=0; i <= NumberTrees; i++) 
      {
        for(j=0; j <= CNumberPoints; j++) 
	  {
	    
            cout<<Probabilities[i][j]<<" ";
          }
	cout<<endl;
      }
  }

double MixtureTrees::Prob (unsigned* vector)
 {
  // The probability of the vector given the Mixture of trees
 // is calculated

  double auxprob;
  int i;

  auxprob =0;
  //for(i=0; i < NumberVars; i++) cout<<vector[i]<<" ";
  //cout<<endl;
  for(i=0; i < NumberTrees; i++) 
	{
	  auxprob += EveryTree[i]->Prob(vector,0)*lambdavalues[i]; 
	}
  return auxprob;
}
  
void MixtureTrees::UpdateTreesProb()
{	
	int i,j;
    	double* VectorProb;
        

 if(Smoothing==0)
   for(i=0; i < NumberTrees; i++) EveryTree[i]->UpdateModel(Probabilities[i],CNumberPoints,AllPoints); 
 else
   {  // Here the smothing procedure is carried out
     VectorProb = new double[CNumberPoints];
     Calculate_Smooth_alpha();
     for(i=0; i < NumberTrees; i++)
       {
          //cout<<"alpha "<<Smooth_alpha[i]<<endl;
         for(j=0; j < CNumberPoints; j++)
	   {
         VectorProb[j] = (1-Smooth_alpha[i])*Probabilities[i][j] + (Smooth_alpha[i])*Probabilities[NumberTrees][j];
           }
          EveryTree[i]->UpdateModel(VectorProb,CNumberPoints,AllPoints);
       }
     delete[] VectorProb;
   }
      
}

void MixtureTrees::UpdateForestProb()
{	
	int i,j;
    	double* VectorProb;
        

 if(Smoothing==0)
   for(i=0; i < NumberTrees; i++) EveryTree[i]->UpdateModelForest(Probabilities[i],CNumberPoints,AllPoints); 
 else
   {  // Here the smothing procedure is carried out
     VectorProb = new double[CNumberPoints];
     Calculate_Smooth_alpha();
     for(i=0; i < NumberTrees; i++)
       {
          //cout<<"alpha "<<Smooth_alpha[i]<<endl;
         for(j=0; j < CNumberPoints; j++)
	   {
         VectorProb[j] = (1-Smooth_alpha[i])*Probabilities[i][j] + (Smooth_alpha[i])*Probabilities[NumberTrees][j];
           }
          EveryTree[i]->UpdateModelForest(VectorProb,CNumberPoints,AllPoints);
       }
     delete[] VectorProb;
   }

}	


void MixtureTrees::FindCurrentCoefficients()
{ 
	int i;

  	for(i=0; i < NumberTrees; i++) 
		lambdavalues[i] = Probabilities[i][CNumberPoints] / NumberPoints;

}

void MixtureTrees::FindCurrentCoefficients(double* baselambdavalues, int typ)
{ 
	int i;
     
        //cout<<" Before "<<endl;
        //PrintProb();
        FillEqualProbabilities();
        //cout<<" Before "<<endl;
        //PrintProb();
        //cout<<" PopProb "<<endl;
        //for(i=0; i <CNumberPoints; i++) cout<<PopProb[i]<<" ";
        //cout<<endl; 
 
	if (typ==1) 
          {
           FindGammaValues();
           NormalizeProb();
           for(i=0; i < NumberTrees; i++) 
	      baselambdavalues[i] = Probabilities[i][CNumberPoints] / NumberPoints;
          }
        else
 	{
         UpdateTotProb();     
  	 for(i=0; i < NumberTrees; i++) 
	   baselambdavalues[i] = Probabilities[i][CNumberPoints] / Probabilities[CNumberTrees][CNumberPoints];        
        }
        //cout<<" Now again "<<endl;
        //PrintProb();
	
       
}

void MixtureTrees::LearningMixture(int Type)
{
switch(Type) 
   { 
     case 1: LearningMixtureMeila(); break; // Classical learning mixture method     case 3:  break;
     case 7: LearningMixtureMeila(); break; // Classical learning mixture method     case 3:  break;
   }
}

void MixtureTrees::MixturesInit(int Type,int InitTreeStructure, double* pvect, double Complexity,int CliqMaxLength, int MaxNumCliq, int LearningType, int Cycles) 
{
    if (InitTreeStructure==4) MixturesInitProd(3,pvect,Complexity);
    else
    {
     switch(Type) 
     { 
       case 1: MixturesInitMeila(InitTreeStructure,pvect,Complexity); break; 
       //case 1: MixturesInitProd(3,pvect,Complexity); break; // Mixture on the same Mut Inf Matrix 
      case 3: MixturesInitProd(3,pvect,Complexity); break; // Mixture on the same Mut Inf Matrix 
      case 7: MixturesInitKikuchi(pvect,Complexity,CliqMaxLength,MaxNumCliq,LearningType,Cycles); break; 
       
    }
    }
}


void MixtureTrees::MixturesInitMeila(int InitTreeStructure,double* pvect, double Complexity) 
{ 
  int i; 
  BinaryTreeModel *OneTree; 
 
  // if(InitTreeStructure==0 || InitTreeStructure==1) RandomLambdas();  else 
  UniformLambdas(); 
   for(i=0; i<NumberTrees;i++) 
    { 
      OneTree =  new BinaryTreeModel(AllPoints->vars,Complexity,NumberPoints); 
      if(NumberTrees==1) OneTree->InitTree(3,CNumberPoints,pvect,AllPoints,NumberPoints);
      else  OneTree->InitTree(InitTreeStructure,CNumberPoints,pvect,AllPoints,NumberPoints);
      //OneTree->PrintModel();
      EveryTree[i]=OneTree; 
    } 
} 


void MixtureTrees::MixturesInitKikuchi(double* pvect, double Complexity, int CliqMaxLength, int MaxNumCliq, int LearningType, int Cycles) 
{ 
  int i; 
  DynFDA *OneTree; 
 
   UniformLambdas(); 
   for(i=0; i<NumberTrees;i++) 
    { 
      //MyMarkovNet = new DynFDA(vars,CliqMaxLength,MaxNumCliq,Complexity,Prior,LearningType,Cycles);  
      OneTree = new DynFDA(AllPoints->vars,CliqMaxLength,MaxNumCliq,Complexity-(0.05*i),Prior,LearningType,Cycles);  
      OneTree->InitTree(CNumberPoints,pvect,AllPoints,NumberPoints);
      EveryTree[i]=OneTree; 
    } 
} 


 void MixtureTrees::MixturesInitProd(int InitTreeStructure,double* pvect, double Complexity) 
 { 
   int i; 
   BinaryTreeModel *OneTree; 
   OneTree =  new BinaryTreeModel(AllPoints->vars, Complexity,NumberPoints); 
   OneTree->InitTree(InitTreeStructure,CNumberPoints,pvect,AllPoints,NumberPoints);
   EveryTree[0]=OneTree;
   //OneTree->PrintModel();
   //OneTree->PrintAllProbs();
    for(i=1; i<NumberTrees;i++) 
     { 
       OneTree =  new BinaryTreeModel(AllPoints->vars, Complexity,NumberPoints); 
       OneTree->ImportProbFromTree((BinaryTreeModel*)EveryTree[i-1]);
       OneTree->ImportMutInfFromTree((BinaryTreeModel*)EveryTree[i-1]);
       OneTree->PutInMutInfFromTree((BinaryTreeModel*)EveryTree[i-1]); 
       OneTree->rootnode =   OneTree->RandomRootNode(); //Better random because Univ not updated
       OneTree->MakeTree(OneTree->rootnode); 
       EveryTree[i]=OneTree; 
       //OneTree->PrintModel();
       //OneTree->PrintMut(); 
       EveryTree[i]->SetGenPoolLimit(CNumberPoints);
       ((BinaryTreeModel*)EveryTree[i])->SetNPoints(NumberPoints);
     } 
    //PutPriors();
    FindCoefficientsFromMI();
 } 

 void MixtureTrees::MixturesInitGreedy(int InitTreeStructure ,double* pvect, double Complexity) 
 { 
   int i;    
   BinaryTreeModel *OneTree; 

   OneTree = new BinaryTreeModel(AllPoints->vars,Complexity,NumberPoints); 	  
   OneTree->InitTree(2,CNumberPoints,pvect,AllPoints,NumberPoints);
   EveryTree[0] = OneTree;

   for(i=1; i<NumberTrees;i++) 
   { 
     OneTree = new BinaryTreeModel(AllPoints->vars,Complexity,NumberPoints); 	  
     OneTree->SetGenPoolLimit(CNumberPoints);
     EveryTree[i] = OneTree;
   } 
 } 




 void MixtureTrees::LearningMixtureMeila()
 {
	 // Here the Mixture has been created and initialized we initial trees 
   int i,goahead;
 
  goahead = 1;
  Count = 0;
  MeanMixtProb = 0;
  //AllPoints->Print();

    while ( Count<LearningSteps && goahead && MaxMixtProb>MeanMixtProb) 
	{
         
	  if(Count>0) FindCurrentCoefficients();
	  
	  FillProbabilities();
 	  FindGammaValues();
          NormalizeProb(); 
	  //UpdateForestProb(); 
          UpdateTreesProb();	
/*
      for(i=0; i < NumberTrees; i++) 
	{
          TreeProb[i] = 0;
     for(int j=0; j <CNumberPoints; j++) 
	 {
          double lambdaXprob  = (EveryTree[i]->Prob(AllPoints->Ind(j),0));           
	  TreeProb[i] += lambdaXprob;
	  cout<<i<<"  "<<j<<"  "<<lambdaXprob<<"  "<<TreeProb[i]<<endl;        
	 }    
    }
*/
    
	  CalculateILikehood(Count); 
          goahead = (Count==0) || (LikehoodValues[Count]-BestLikehood> 0.5); //fabs(BestLikehood/double(10000)));
          if(Count==0) BestLikehood = LikehoodValues[0];
          else if (LikehoodValues[Count]-BestLikehood> 0) BestLikehood = LikehoodValues[Count];
	  
	 
	  /*  	 
        for(i=0; i<NumberTrees;i++)
            {              
			 cout<<endl;
               cout<<"lambda"<<i<<" = "<<lambdavalues[i]<<endl;  
               cout<<"TreeProb"<<i<<" = "<<TreeProb[i]<<endl; 
		 
             }
	   
	      cout<<"Iter="<<Count<<" Likehood="<<LikehoodValues[Count]<<" Prob="<<MeanMixtProb<<" BestLikehood="<<BestLikehood<<" ProbPop "<<MeanMixtProb<<endl;      
	  */
           	 Count++;     
  
       
}

      PutPriors();
 BestProb = Prob(AllPoints->P[0]); //Only when the optimum is the first	
	
 //PrintProb();
//cout<<"Iter="<<Count<<" Likehood="<<LikehoodValues[Count]<<" Prob="<<MeanMixtProb<<" "<<goahead<<" "<<goahead<<" Final"<<endl;        
 	
}

void MixtureTrees::PutPriors()
{
    int i;
  double univ_prior, biv_prior;
   
               switch(Prior) 
                     { 
                       case 0: break; // No prior 
			 case 1: 
			     univ_prior = 1; //Calculate_Best_Prior(NumberPoints,1,1);//SelInt
                             biv_prior = 1;// Calculate_Best_Prior(NumberPoints,2,1);
                               for(i=0; i < NumberTrees; i++) 
                                     EveryTree[i]->SetPrior(univ_prior,biv_prior,NumberPoints);
			        
                               break; // Recommended prior 

                       case 2: for(i=0; i < NumberTrees; i++)
			        {
                                 univ_prior = Calculate_Best_Prior(NumberPoints,1,SelInt*(1+TreeProb[i]*(1-lambdavalues[i])));
                                 biv_prior = Calculate_Best_Prior(NumberPoints,2,SelInt*(1+TreeProb[i]*(1-lambdavalues[i])));
                            
                                 EveryTree[i]->SetPrior(univ_prior,biv_prior,NumberPoints);
                                }
                               break; // Adaptive Prior
                       
                     } 

	
}


void MixtureTrees::FindCoefficientsFromMI()
{
int i;
 double auxtot=0;
 //cout<<"MI"<<" ";
  	for(i=0; i < NumberTrees; i++) 
	{
	    if(EveryTree[i]->TreeL>0)
              {
                auxtot += EveryTree[i]->TreeL;
                lambdavalues[i] = EveryTree[i]->TreeL;
              }
	    else lambdavalues[i] = 0;
            //cout<<EveryTree[i]->TreeL<<"  ";
        }
        //cout<<endl;
        for(i=0; i < NumberTrees; i++) 
	   if(EveryTree[i]->TreeL>0) lambdavalues[i] /= auxtot;

}

void MixtureTrees::LearningMixture(int init, int steps)
{
	// Here the Mixture has been created and initialized we initial trees 
  int Count;
  //int i;

  Count = 0;
	while ( Count< steps )
	{
	  if(init==0 && Count==0)
	  {
		FillProbabilities();
		FindGammaValues();
        NormalizeProb();
	   }
	  else 
	  {	 
       UpdateTreesProb();
	   FindCurrentCoefficients();
       FillProbabilities();
	   FindGammaValues();
       NormalizeProb();
	  }
      CalculateLikehood(init+Count);
/*
          PrintProb();
	  printf("%d\n ",Count);
	  for(i=0; i<NumberTrees;i++)  printf("%f\n ",lambdavalues[i]);
          fprintf(outfile,"%d\n ",Count);
	  for(i=0; i<NumberTrees;i++)  fprintf(outfile,"%f\n ",lambdavalues[i]);
          for(i=0; i<NumberTrees;i++)  EveryTree[i]->PrintModel();
          printf(" \n %f \n ",LikehoodValues[Count]);   
          fprintf(outfile,"%f \n ",LikehoodValues[Count]);
*/       
      	  Count++;
}	 

}

void MixtureTrees::CalculateLikehood(int step)
{ 	 // Equation Numb (3.19) in Melina's Thesis

	int i,j;
    double auxval,auxval1;
	
	LikehoodValues[step] = 0;
    
if (NumberTrees==1)
    for(j=0; j < CNumberPoints; j++) LikehoodValues[step] += log(EveryTree[0]->Prob(AllPoints->Ind(j),0)); 
    else
      {
       for(i=0; i < NumberTrees; i++) 
	{
         if(lambdavalues[i]>0)
	   LikehoodValues[step] += Probabilities[i][CNumberPoints]*log(lambdavalues[i]);
           auxval = 0;
  	 for(j=0; j < CNumberPoints; j++) 
	  { 
	   auxval1 = EveryTree[i]->Prob(AllPoints->Ind(j),0); 
           if(auxval1>0)
	     auxval += Probabilities[i][j] * log(auxval1);
          }  
	 LikehoodValues[step] += auxval*Probabilities[i][CNumberPoints];	
        }
      }
}

double MixtureTrees::CalculateILikehood(int step)
{ 	
 int i,j;
    double auxval,auxval1,sumprob;
	
	LikehoodValues[step] = 0;
        sumprob = 0;  
        for(i=0; i < NumberTrees; i++)  TreeProb[i] = 0;      
        //MeanMixtProb  =0 ;
         
        for(j=0; j < CNumberPoints; j++)
	{
         auxval1 = 0;
         for(i=0; i < NumberTrees; i++)  
	  {          
	   auxval  = (EveryTree[i]->Prob(AllPoints->Ind(j),0));
           auxval1 += auxval*lambdavalues[i];
           TreeProb[i] += auxval; //Modificado
          }
	   
	 // if (auxval1>0) LikehoodValues[step]  += (PopProb[j]*(log(PopProb[j])-log(auxval1)));     //else  LikehoodValues[step]  -= (CNumberPoints*PopProb[j]*log(PopProb[j]));   

        if (auxval1>0) LikehoodValues[step]  += (NumberPoints*PopProb[j]*(log(auxval1)));   
         sumprob += auxval1;
         //cout<<" j "<<j<<"RealProb "<<PopProb[j]<<" prob "<<auxval<<" sumprob "<<sumprob<<" likeh "<<LikehoodValues[step]<<endl;
         }
         MeanMixtProb = sumprob;
	return LikehoodValues[step];
}


 
double MixtureTrees::CalculateOptimalLikehood()
{ 	
 int j;
 double Likehood;
 Likehood = 0;
         for(j=0; j < CNumberPoints; j++)  
            Likehood  -= PopProb[j]*log(PopProb[j]);            
	 return Likehood;
}

int MixtureTrees::StopCriteria(int count)
{
    
   switch(StopCrit)
    { 
	  case 0: return (count>=LearningSteps); break;
	  case 1: return (LikehoodValues[count]>LearningThreshold);  break;
      case 2: return ((LikehoodValues[count]-LikehoodValues[count-1])>LearningThreshold);  break; 
       case 3: return (count>=LearningSteps || MaxMixtProb<MeanMixtProb); break;
	}
return 0;
}

void MixtureTrees::SetStopCriteria(int StopCase,int NSteps,double Lthreshold)
 { 
    StopCrit = StopCase;
	LearningSteps = NSteps;
	LearningThreshold = Lthreshold;
 }
 
void MixtureTrees::SetPop(Popul* pop)
 { 
	 AllPoints = pop;
 }

void MixtureTrees::SetTrees(int pos, AbstractProbModel* tree)
 { 
	 EveryTree[pos] = tree;
 }

void MixtureTrees::SamplingFromMixture(Popul *FinalPoints)
{
   double cutoff,tot;
   int i,j;
 
   //AllPoints->SetElit(FinalPoints->elit,FinalPoints);
   
   for(i=FinalPoints->elit; i<FinalPoints->psize; i++)
   {
	 // Here it is necessary to apply the SUS method
     cutoff = myrand();	
	 j = 0;
	 tot = lambdavalues[0];
	 while ( (cutoff>tot) && (j+1<NumberTrees) )
	 { 
		 j++;
		 tot += lambdavalues[j];
         } 
        EveryTree[j]->GenIndividual(FinalPoints,i);
   } 
}

void MixtureTrees::Findvar_assig(int NT,int totv,int* var_assig)
{
// Calculates how many variables will be generated by each tree in 
// the mutual sampling step

    int i,tvars,auxn;
    tvars = 0;
    for(i=0;i<NT;i++) 
    {
	if(lambdavalues[i]>0) var_assig[i] = int(lambdavalues[i]*totv);
        else var_assig[i] = 0;
        tvars += var_assig[i];
    }
    
    // cout <<"first";
    //for(j=0;j<NumberTrees;j++) cout<<var_assig[j]<<" ";
    //cout<<endl;
    /*if (var_assig[0]<0)
   {
      for(j=0;j<NumberTrees;j++) cout<<lambdavalues[j]<<" ";
      cout<<endl;
      }*/
    while (tvars>totv) 
    {
        auxn = randomint(NumberTrees);
	    if(var_assig[auxn]>0)
	    { 
		var_assig[auxn]--;
                tvars--;
	    } 
    }
   
  while (tvars<totv) 
    { 
       auxn = randomint(NumberTrees);
         if(var_assig[auxn]<totv)
	    { 
             var_assig[auxn]++;
             tvars++;
            }
    }
}


void MixtureTrees::SamplingIndividual(Popul *NewPop, int pos, int cprior)
{
  double auxprob,aux2,cutoff,auxuniv; 
 int aux,j,i; 

  for (j=0;j<NumberVars; j++ ) 
    {  
     cutoff = myrand(); 
     auxuniv=0; 
     for (i=0;i<NumberTrees; i++)
     {
          if ( ((BinaryTreeModel*)EveryTree[i])->Tree[j]==-1)  auxuniv +=   (((BinaryTreeModel*)EveryTree[i])->AllProb[j]*lambdavalues[i]);	   
        else
       {        
	   if (NewPop->P[pos][((BinaryTreeModel*)EveryTree[i])->Tree[j]]==1) 
		 { 
	   	  if (j< ((BinaryTreeModel*)EveryTree[i])->Tree[j])  
		  { 
		   aux = j*(2*NumberVars-j+1)/2 + ((BinaryTreeModel*)EveryTree[i])->Tree[j]-2*j-1; 
		   aux2=( ((BinaryTreeModel*)EveryTree[i])->AllSecProb[3][aux]); 
		  } 
		  else  
		  { 
 		   aux =  ((BinaryTreeModel*)EveryTree[i])->Tree[j]*(2*NumberVars- ((BinaryTreeModel*)EveryTree[i])->Tree[j]+1)/2 +j -2* ((BinaryTreeModel*)EveryTree[i])->Tree[j]-1; 
		   aux2=( ((BinaryTreeModel*)EveryTree[i])->AllSecProb[3][aux]); 
		  } 
		  auxprob=(aux2*NumberPoints+cprior)/( ((BinaryTreeModel*)EveryTree[i])->AllProb[ ((BinaryTreeModel*)EveryTree[i])->Tree[j]]*NumberPoints+2*cprior); 
		 } 
		else 
		{ 
                if(j< ((BinaryTreeModel*)EveryTree[i])->Tree[j])  
		 { 
		  aux = j*(2*NumberVars-j+1)/2 + ((BinaryTreeModel*)EveryTree[i])->Tree[j]-2*j-1; 
		  aux2=( ((BinaryTreeModel*)EveryTree[i])->AllSecProb[2][aux]); 
		 } 
		 else  
		 { 
		  aux =  ((BinaryTreeModel*)EveryTree[i])->Tree[j]*(2*NumberVars- ((BinaryTreeModel*)EveryTree[i])->Tree[j]+1)/2 +j -2* ((BinaryTreeModel*)EveryTree[i])->Tree[j]-1; 
		  aux2=( ((BinaryTreeModel*)EveryTree[i])->AllSecProb[1][aux]); 
		 } 
	         auxprob=(aux2*NumberPoints+cprior)/((1- ((BinaryTreeModel*)EveryTree[i])->AllProb[ ((BinaryTreeModel*)EveryTree[i])->Tree[j]])*NumberPoints+2*cprior); 
		} 
                auxuniv += auxprob*lambdavalues[i];
       }
        if (cutoff >= auxuniv) NewPop->P[pos][j]=0; 
  	else NewPop->P[pos][j]=1;    
//	if (i>0) cout<<"auxuniv "<<auxuniv<< " i "<<i<<" j "<<j<<" T[j] "<<((BinaryTreeModel*)EveryTree[i])->Tree[j]<<" X(T[j])= "<<NewPop->P[pos][((BinaryTreeModel*)EveryTree[i])->Tree[j]]<<" BivEn "<<aux2<<"  AllProb "<<((BinaryTreeModel*)EveryTree[i])->AllProb[((BinaryTreeModel*)EveryTree[i])->Tree[j]]<<" cprob "<<auxprob<<" cutoff "<<cutoff<<" X(j)= "<<NewPop->P[pos][j]<<endl;
     }  
    }
}


void MixtureTrees::SamplingFromMixtureCycles(Popul *FinalPoints)
{
    int i,j;

    SamplingFromMixture(FinalPoints);
 for(j=0; j<1; j++)
 for(i=FinalPoints->elit; i<FinalPoints->psize; i++)
     SamplingIndividual(FinalPoints,i,1);
}


void MixtureTrees::SamplingFromMixtureMixt(Popul *FinalPoints)
{
    //double cutoff,tot;
   int i,j;
   int *var_assig; // The number of variables that will be sampled by each  tree
   int *ordering,*mvector; // ordering of trees to be sampled

   var_assig = new int[NumberTrees];
   mvector = new int[NumberVars];
   ordering = new int[NumberTrees];
   Findvar_assig(NumberTrees,NumberVars,var_assig);
   InitPerm(NumberTrees,ordering); 
   //for(j=0;j<NumberTrees;j++) cout<<var_assig[j]<<" ";
   //cout<<endl;
   for(i=FinalPoints->elit; i<FinalPoints->psize; i++)
   {
      RandomPerm(NumberTrees,NumberTrees,ordering); 
      for(j=0;j<NumberVars;j++) 
        {
         mvector[j] = -1;
        }
      for(j=0;j<NumberTrees;j++)
      if (var_assig[ordering[j]]>0) 
         EveryTree[ordering[j]]->GenPartialIndividual(FinalPoints,i,var_assig[ordering[j]],mvector,1);
   } 

   delete[] var_assig;
   delete[] ordering;
   delete[] mvector;
}

void MixtureTrees::SamplingFromMixtureHMixing(Popul *FinalPoints)
{
   int i,j,togen;
   int *var_assig; // The number of variables that will be sampled by each  tree
   int *ordering,*mvector; // ordering of trees to be sampled

   var_assig = new int[NumberTrees];
   mvector = new int[NumberVars];
   ordering = new int[NumberVars];
   Findvar_assig(NumberTrees,NumberVars,var_assig);
   InitPerm(NumberVars,ordering); 
   //for(j=0;j<NumberTrees;j++) cout<<var_assig[j]<<" ";
   //cout<<endl;
   for(i=FinalPoints->elit; i<FinalPoints->psize; i++)
   {
      RandomPerm(NumberVars,NumberVars,ordering); 
      for(j=0;j<NumberVars;j++)  mvector[j] = -1;
      togen = NumberVars;
      for(j=0;j<NumberTrees;j++)
      if (var_assig[j]>0) 
      {
         EveryTree[j]->GenPartialIndividual(FinalPoints,i,togen,mvector,1);
         togen -=  var_assig[j];  
         for(j=NumberVars-togen;j<NumberVars;j++)  mvector[ordering[j]] = -1;
      }
   } 

   delete[] var_assig;
   delete[] ordering;
   delete[] mvector;
}



void MixtureTrees::CalcMixtProb(int NTrees, double* lambdas,double* ExtraProbs)
{
	int i,j;
        double lambdaXprob;
    
	for(j=0; j <CNumberPoints+1; j++) ExtraProbs[j] = 0;
       
    for(i=0; i < NTrees; i++) 
     {
       for(j=0; j <CNumberPoints; j++) 
	 {      
	  lambdaXprob  = (EveryTree[i]->Prob(AllPoints->Ind(j),0));           	      
          ExtraProbs[j] += lambdaXprob * lambdas[i];
         }         
     }   
}

double MixtureTrees::SearchStructure(double minerror, int* full,int etime,int coeftype,double complexcoef)
{
    int Best_Edge_Type,edgetype,i,j,k,l,kk,MNumberTrees,gain,feasible,aux ;
    double A,B,ui,uj,err,lik_err;
    int *bivpos;
    double *rm;
    double *Ai;
    double AB[4];
    double BB[4];
    //double auxtm[4];
    double *tm,*besttm;
    int Best_Tree,Best_Edge_i,Best_Edge_j;

    B=0; ui =0; uj=0; Best_Edge_Type =0; //rinit
    Best_Tree = 0; Best_Edge_i = 0;Best_Edge_j=0; //rinit

   bivpos = new int[CNumberPoints];
    rm = new double[CNumberPoints];
    Ai = new double[CNumberPoints];
    tm = new double[4]; besttm = new double[4];
    gain = 0;

     ((BinaryTreeModel*)EveryTree[0])->PrintModel();
 

    if( etime==0) MNumberTrees=1;
    else
     MNumberTrees = (NumberTrees+1<CNumberTrees)?NumberTrees+1:CNumberTrees;
      //MNumberTrees = 1; //Primero para un solo arbol

 for(k=MNumberTrees-1; k >=0; k--)
  {   
     if (k==MNumberTrees-1  && etime>0 && NumberTrees<CNumberTrees ) 
     {   
        for(kk=0;kk<=MNumberTrees-1; kk++) baselambdavalues[kk] = lambdavalues[kk];\
	NumberTrees++;
	FindCurrentCoefficients(baselambdavalues,coeftype);
        NumberTrees--;
        Printlambdas(baselambdavalues,NumberTrees+1);   
        //UpdateCoefAfterAddition_trad(k,NumberTrees+1);   
         //UpdateCoefAfterAddition(int changedtree)      	
	FillProbabilitiesSimple();         
        CalcMixtProb(MNumberTrees,baselambdavalues,ExtraTotProb);       //
     }
    
   if   ( etime==0 || k<MNumberTrees-1 || k==0 ||baselambdavalues[k]>0.00001)       
    { 
	for(i=0; i < NumberVars-1; i++) //NumberVars-1
     { 
      for(j=i+1; j < NumberVars; j++)
      {
	  AB[0]=0.0; AB[1]=0.0; AB[2]=0.0; AB[3]=0.0; BB[0]=0.0; BB[1]=0.0; BB[2]=0.0; BB[3]=0.0;
       edgetype = ((BinaryTreeModel*)EveryTree[k])->Edge_Cases(i, j); 
       if (edgetype>0)
        { 
         for(l=0; l <CNumberPoints; l++)
          {        
           bivpos[l] =  2*AllPoints->P[l][i]+AllPoints->P[l][j];
           ui = ((BinaryTreeModel*)EveryTree[k])->AllProb[i];  uj = ((BinaryTreeModel*)EveryTree[k])->AllProb[j];
	   if(k==MNumberTrees-1 && etime>0 && NumberTrees<CNumberTrees)
               A = PopProb[l]-(ExtraTotProb[l]-Probabilities[k][l]*baselambdavalues[k]);              else
               A = PopProb[l]-(Probabilities[CNumberTrees][l]-Probabilities[k][l]*lambdavalues[k]);
	        
           switch(bivpos[l]) 
           { 
      	    case 0: B = Probabilities[k][l]/( (1-ui)*(1-uj)); break;          
            case 1: B = Probabilities[k][l]/( (1-ui)*(uj)); break;     
            case 2: B = Probabilities[k][l]/(  (ui)*(1-uj)); break;            
            case 3: B = Probabilities[k][l]/(  (ui)*(uj)); break;   
           }                   
                     
            Ai[l] = A; rm[l] = B; AB[bivpos[l]] += 2*A*B;  BB[bivpos[l]] += 2*B*B;
    
	   /*
	   
	    if ((etime==0 && i==1 && j==2)) 
            {
              AllPoints->Print(l);
              cout<<" l "<<l<<" A "<<A<< " B "<<B<<" ui "<<ui<<" uj "<<uj<<" PopProb[l] "<<PopProb[l]<<" Probtot "<<Probabilities[CNumberTrees][l]<<" Probk"<<Probabilities[k][l]<<" bl " <<bivpos[l]<<" Ai[l] "<<Ai[l]<<" AB "<<AB[bivpos[l]]<<" BB "<<BB[bivpos[l]]<<endl; 
	      }
	   */
          }

          
									

 if (k>0) feasible = FindParams(edgetype,AB,BB,ui,uj,tm); //if (edgetype==4 || edgetype==5)
 else 
 {
  feasible = 1;
  if (j<i) 
   {
    aux =j*(2*NumberVars-j+1)/2 +i-2*j-1; 

    tm[0]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[0][aux];
    tm[2]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[1][aux]; 
    tm[1]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[2][aux];
    tm[3]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[3][aux];   
   }
  else
   {
    aux = i*(2*NumberVars-i+1)/2 +j -2*i-1;
    tm[0]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[0][aux];
    tm[1]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[1][aux]; 
    tm[2]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[2][aux];
    tm[3]=((BinaryTreeModel*)EveryTree[k])->AllSecProb[3][aux];
   }
 }
      if(feasible)
       {                       
	  if(k==MNumberTrees-1 && etime>0 && NumberTrees<CNumberTrees) 
           {
              err =CalculateQuadError(baselambdavalues,Ai,k,bivpos,tm,rm); 
              //lik_err = CalculateILikehood(baselambdavalues,k,bivpos,tm,rm,NumberTrees+1);     
              lik_err = CalculateMDL(baselambdavalues,k,bivpos,tm,rm,NumberTrees+1,1,complexcoef);           }
         else 
	  {
           err =CalculateQuadError(lambdavalues,Ai,k,bivpos,tm,rm);
           //lik_err = CalculateILikehood(lambdavalues,k,bivpos,tm,rm,NumberTrees);
           lik_err = CalculateMDL(lambdavalues,k,bivpos,tm,rm,NumberTrees,0,complexcoef);
          };     
  
	  //cout<<"t "<<etime<<" tree "<<k<<" type "<<edgetype<<" i "<<i<<" j "<<j<<" tm[0] "<<tm[0]<<" tm[1] "<<tm[1]<<" tm[2] "<<tm[2]<<" tm[3] "<<tm[3]<<" ui "<<ui<<" uj "<<uj<<" error "<<err<<" lkh "<<lik_err<<endl; 

/*	    
         if(err<minerror)
 	   {
	       *full = 0;  minerror = err;
               Best_Tree = k; Best_Edge_i = i; Best_Edge_j = j; Best_Edge_Type = edgetype;
               besttm[0]=tm[0]; besttm[1]=tm[1]; besttm[2]=tm[2]; besttm[3]=tm[3];    
           }
	 
        if ((etime==1 && i==2 && j==3)) 
        { 
         auxtm[0]=0.470426; auxtm[1]=0.0234211; auxtm[2]= 0.0603349; auxtm[3]= 0.445818;  
         err = CalculateQuadError(lambdavalues,Ai,k,bivpos,auxtm,rm);
         lik_err = CalculateILikehood(lambdavalues,k,bivpos,auxtm,rm); 
         cout<< "In this case err "<<err<<" and lik "<<lik_err<<endl;
        }
*/	  
 	   if( (1*lik_err)<=minerror)
 	   {
	       //cout<<" min "<<lik_err<<endl;
	       *full = 0;  minerror = 1*lik_err;
               Best_Tree = k; Best_Edge_i = i; Best_Edge_j = j; Best_Edge_Type = edgetype;
               besttm[0]=tm[0]; besttm[1]=tm[1]; besttm[2]=tm[2]; besttm[3]=tm[3];    
	   }
	  
       }	 
       } 
      }
     }
   }

 }
 int x;
 if (! *full) //In Add_Edge the first node is set as a son of the second
  {
     if(Best_Edge_Type==1  || Best_Edge_Type==3  || Best_Edge_Type== 4 ) 
            x = EveryTree[Best_Tree]->Add_Edge(Best_Edge_i,Best_Edge_j);
     else
            x = EveryTree[Best_Tree]->Add_Edge(Best_Edge_j,Best_Edge_i);   
     EveryTree[Best_Tree]->AdjustBivProb(Best_Edge_i,Best_Edge_j,besttm);
 
     //cout<<"Best-tree"<< Best_Tree<<" type edge "<<Best_Edge_Type<<"  "<<Best_Edge_i<<" uj "<<Best_Edge_j<<" minerror "<<minerror<<endl;
     
   if  (Best_Tree==MNumberTrees-1 && NumberTrees<CNumberTrees && etime>0)
     {
        //FillProbabilitiesSimple(Best_Tree,0); 
        //UpdateCoefAfterAddition_trad(Best_Tree,NumberTrees);  
	NumberTrees++;
        MixtComplexity += NumberVars-1;
        FindCurrentCoefficients(baselambdavalues,coeftype);
        Setlambdas(baselambdavalues,NumberTrees);   
        cout<< "Truelamdas "; Printlambdas(lambdavalues,NumberTrees);     
     }
   else 
     {
	 //FillProbabilitiesSimple(Best_Tree,1);  
        //UpdateCoefAfterAddition_trad(Best_Tree,NumberTrees);   
	 if(NumberTrees>1)
          {
           FindCurrentCoefficients(baselambdavalues,coeftype);
           Setlambdas(baselambdavalues,NumberTrees);   
          }
         MixtComplexity ++;
	 cout<< "Truelamdas "; Printlambdas(lambdavalues,NumberTrees);          
     }
  }  
    
   delete[] bivpos;
   delete[] rm; 
   delete[] Ai; 
   delete[] tm;
   return minerror;
 }

void MixtureTrees::Printlambdas(double* lambd,int NTrees)
{
    int kk;
     for(kk=0;kk<NTrees; kk++) cout<<" lambda"<<kk<<"  "<<lambd[kk];     
     cout<<endl;     
}

void  MixtureTrees::Setlambdas(double* lambd,int NTrees)
{
    int kk;
     for(kk=0;kk<NTrees; kk++) lambdavalues[kk] = lambd[kk];          
}

double MixtureTrees::FindParamIndep(double *AB,double *BB, double* tm)
    {
	double delta, auxdelta;
        //cout<<"P1BB "<<BB[0]<<" "<<BB[1]<<" "<<BB[2]<<" "<<BB[3]<<endl;
        //cout<<"P1AB "<<AB[0]<<" "<<AB[1]<<" "<<AB[2]<<" "<<AB[3]<<endl;

        delta = (BB[0]*BB[1]*BB[2]*BB[3]- AB[0]*BB[1]*BB[2]*BB[3]) - (BB[0]*AB[1]*BB[2]*BB[3]+ BB[0]*BB[1]*AB[2]*BB[3]+ BB[0]*BB[1]*BB[2]*AB[3]); 
       
	//cout<<"Den is "<<delta<<" ";
        auxdelta = (BB[1]*BB[2]*BB[3]+BB[0]*BB[2]*BB[3]+BB[0]*BB[1]*BB[3]+BB[0]*BB[1]*BB[2]);
        if(auxdelta>0) delta = delta / auxdelta;
       
        //cout<<"Delta is "<<delta<<endl;
	if (BB[0] != 0) tm[0] = (AB[0]+delta)/BB[0]; 
        if (BB[1] != 0) tm[1] = (AB[1]+delta)/BB[1]; 
        if (BB[2] != 0) tm[2] = (AB[2]+delta)/BB[2]; 
        if (BB[3] != 0) tm[3] = (AB[3]+delta)/BB[3]; 
    
        return(tm[0]+tm[1]+tm[2]+tm[3]);
   }

int MixtureTrees::FindParams(int edgetype, double *AB,double *BB, double ui, double uj, double* tm)
{
     tm[0] = 0; tm[1] =0; tm[2] = 0; tm[3] = 0; 

  
     switch(edgetype) 
          { 
            case 1: FindParamIndep(AB,BB,tm);  break; // To be defined 
   	    case 2: FindParamDep(AB,BB,uj,1,tm);    break;  
            case 3: FindParamDep(AB,BB,ui,0,tm);    break;     
            case 4: FindAllParamDep(AB,BB,ui,uj,tm);    break;
            case 5: FindAllParamDep(AB,BB,uj,ui,tm);    break;  
            case 6: FindAllParamDep(AB,BB,ui,uj,tm);    break;
            case 7: FindAllParamDep(AB,BB,uj,ui,tm);    break;            
         }   
     //cout<<tm[0]<<" "<<tm[1]<<" "<<tm[2]<<" "<<tm[3]<<" "<<"edge "<<edgetype<<" ui "<<ui<<" uj "<<uj<<endl;
     if (!(tm[0]>=0 && tm[1]>=0 && tm[2]>=0 && tm[3]>=0) || (fabs(tm[0]+tm[1]+tm[2]+tm[3]-1)>0.000001))
     {
         
          Normalize_Param(tm);
	  //   cout<<"Norm "<<tm[0]<<" "<<tm[1]<<" "<<tm[2]<<" "<<tm[3]<<" "<<"edge "<<edgetype<<" ui "<<ui<<" uj "<<uj<<endl;
    
         switch(edgetype) 
          { 
	    case 1: return 1;
            case 2: return (fabs(tm[1]+tm[3] - uj)<0.000001);     break;  
            case 3: return (fabs(tm[2]+tm[3] - ui)<0.000001);     break;     
            case 4: return ((fabs(tm[2]+tm[3] - ui)<0.000001) && (fabs(tm[1]+tm[3] - uj)<0.000001)); break;
            case 5: return ((fabs(tm[2]+tm[3] - uj)<0.000001) && (fabs(tm[1]+tm[3] - ui)<0.000001)); break;
            case 6: return ((fabs(tm[2]+tm[3] - ui)<0.000001) && (fabs(tm[1]+tm[3] - uj)<0.000001)); break;
            case 7: return ((fabs(tm[2]+tm[3] - uj)<0.000001) && (fabs(tm[1]+tm[3] - ui)<0.000001)); break;          
          }
     }
     return 1;
     
}
double MixtureTrees::FindParamDep(double *AB,double *BB, double univ, int dep, double* tm)
    {
	double delta1,delta2;
	
	delta1 =0; delta2 =0; //rinit

       //cout<<"PDBB "<<BB[0]<<" "<<BB[1]<<" "<<BB[2]<<" "<<BB[3]<<endl;
        //cout<<"PDAB "<<AB[0]<<" "<<AB[1]<<" "<<AB[2]<<" "<<AB[3]<<endl;
     
	
    if (dep==1) // Case where q is the node with fixed univariate marg.
         {
          if ((BB[0]+BB[2]) != 0) delta1 = ((1-univ)*BB[0]*BB[2] -BB[2]*AB[0]- BB[0]*AB[2])/(BB[0]+BB[2]);
          if ((BB[1]+BB[3]) != 0) delta2 = ((univ)*BB[1]*BB[3] -BB[3]*AB[1]- BB[1]*AB[3])/(BB[1]+BB[3]);

	 if (BB[0]!= 0) tm[0] = (AB[0]+delta1)/BB[0]; 
         if (BB[1]!= 0) tm[1] = (AB[1]+delta2)/BB[1]; 
         if (BB[2]!= 0) tm[2] = (AB[2]+delta1)/BB[2]; 
         if (BB[3]!= 0) tm[3] = (AB[3]+delta2)/BB[3]; 
	 }
        else
         {        // Case where p is the node with fixed univariate marg.
           if ((BB[0]+BB[1]) !=0) delta1 = ((1-univ)*BB[0]*BB[1] -BB[1]*AB[0]- BB[0]*AB[1])/(BB[0]+BB[1]);
           if ((BB[2]+BB[3]) !=0) delta2 = ((univ)*BB[2]*BB[3] -BB[2]*AB[2]- BB[2]*AB[3])/(BB[2]+BB[3]);

	  if (BB[0]!=0) tm[0] = (AB[0]+delta1)/BB[0]; 
          if (BB[1]!=0) tm[1] = (AB[1]+delta1)/BB[1]; 
          if (BB[2]!=0) tm[2] = (AB[2]+delta2)/BB[2]; 
          if (BB[3]!=0) tm[3] = (AB[3]+delta2)/BB[3]; 
	 }
        return(tm[0]+tm[1]+tm[2]+tm[3]);
   }


double MixtureTrees::FindAllParamDep(double *AB,double *BB, double univ1, double univ2, double* tm)
    {
	double delta2,delta3,delta4;
       
        //cout<<"P2BB "<<BB[0]<<" "<<BB[1]<<" "<<BB[2]<<" "<<BB[3]<<endl;
        //cout<<"P2AB "<<AB[0]<<" "<<AB[1]<<" "<<AB[2]<<" "<<AB[3]<<endl;

	 // Case where both nodes have fixed univariate marg.

	delta2 = -(-univ2*BB[3]*BB[0]+BB[0]*BB[2]+BB[3]*BB[0]+AB[3]*BB[0]-BB[0]*BB[2]*univ2-AB[2]*BB[0]-BB[0]*BB[3]*univ1-univ2*BB[3]*BB[1]-BB[1]*AB[2]+AB[1]*BB[2]+AB[3]*BB[1]+BB[1]*univ1*BB[2]-BB[3]*AB[0]+AB[1]*BB[3]-BB[1]*BB[2]*univ2-AB[0]*BB[2])/(BB[0]+BB[1]+BB[2]+BB[3]);

	
       delta3 = (BB[0]*BB[2]-AB[2]*BB[0] +AB[3]*BB[0]-univ2*BB[3]*BB[0] +BB[3]*BB[0]-BB[0]*BB[2]*univ2-BB[1]*BB[0]*univ1+BB[1]*BB[0]-AB[1]*BB[0]-BB[0]*BB[3]*univ1-AB[0]*BB[1]-AB[0]*BB[2]-BB[3]*AB[0])/(BB[0]+BB[1]+BB[2]+BB[3]);



       delta4 = -(AB[2]*BB[0]-BB[0]*BB[2]+BB[0]*BB[2]*univ2+AB[3]*BB[2]+AB[0]*BB[2]-BB[1]*univ1*BB[2]+AB[2]*BB[3]-AB[1]*BB[2]-univ1*BB[3]*BB[2]+BB[1]*BB[2]*univ2+BB[1]*AB[2])/(BB[0]+BB[1]+BB[2]+BB[3]);

         
/*
 if ((delta2>0 && delta2<0.00000001) ||  (delta2>-0.00000001 && delta2<0)) delta2 = 0.0;
 if ((delta3>0 && delta3<0.00000001) ||  (delta3>-0.00000001 && delta3<0)) delta3 = 0.0;
 if ((delta4>0 && delta4<0.00000001) ||  (delta4>-0.00000001 && delta4<0)) delta4 = 0.0;
*/
 	  if (BB[0]!=0) tm[0] = (AB[0]+delta3)/BB[0]; 
          if (BB[1]!=0) tm[1] = (AB[1]+delta2+delta3)/BB[1]; 
          if (BB[2]!=0) tm[2] = (AB[2]+delta4)/BB[2]; 
          if (BB[3]!=0) tm[3] = (AB[3]+delta2+delta4)/BB[3];

	  //cout<<tm[0]<<" "<<tm[1]<<" "<<tm[2]<<" "<<tm[3]<<" "<<endl;
	 
        return(tm[0]+tm[1]+tm[2]+tm[3]);
    }

void MixtureTrees::Normalize_Param(double* tm)
    {
	double aux,tot;
        int i;

        aux = tm[0]; 
        tot = tm[0];
  

    for (i=1;i<4;i++) 
	{ 
	    if (tm[i]<aux) aux = tm[i]; 
    	    tot += tm[i];             
        }

        if (aux<0)
	{ tot = 0;  
         for (i=0;i<4;i++) 
          {
            tm[i] -= aux;
            tot += tm[i];
          }
	}
        if (tot>0) for (i=0;i<4;i++) tm[i] /= tot;
    }

void MixtureTrees::LearningMixtureExact(int coeftype,double complexcoef)
{
  double olderror,newerror;
  int i, full;
  Count = 0;
  MixtComplexity = NumberVars;

  ExtraTotProb = new double[CNumberPoints+1];
  baselambdavalues = new double[NumberTrees+1];
  
  lambdavalues[0]=1.0;

  // El primer tree debe venir inicializado con las probabilidades univariadas y biv, pero sin estructura.
  
  full = 0; 
  i = 0;
  newerror = 1000.0;
  olderror = 1000.0;

      
   
  while (i<1  && newerror<=olderror)  // While it is possible to add edges to the mixture do it
   {    
    olderror = newerror;
    if (i==0) InitNumberTrees(); 
    FillProbabilitiesSimple(); 
    CalculateILikehood(i);
    /*  for(j=0; j<NumberTrees;j++)
            {              
		cout<<"lambda"<<j<<" = "<<lambdavalues[j]<<" ";               
             }
    */
    //cout<<"  Iter="<<i<<" Likehood="<<LikehoodValues[i]<<endl; 
    full = 1;
    //newerror = SearchStructure(olderror,&full,i,coeftype,complexcoef); 	
    NewTreeStructure(); 
    ((AbstractTreeModel*)EveryTree[1])->CleanTree();
    i++;
  }  
   NumberTrees++;  
    FillProbabilitiesSimple(); 
    CalculateILikehood(i);
   NumberTrees--;    
 
    /*  for(j=0; j<NumberTrees;j++)
            {              
		cout<<"lambda"<<j<<" = "<<lambdavalues[j]<<" ";               
             }
    */
   // cout<<"  Iter="<<i<<" Likehood="<<LikehoodValues[i]<<endl; 
   FinishNumberTrees();	
   delete[] ExtraTotProb;
   delete[] baselambdavalues;
}

int MixtureTrees::NextStructure()
{
 double bestgain,gain;
 int besttree, besta, bestb;
 int i,j,k,l,aux,MNumberTrees;
 int *auxParents;

 besta = 0; bestb = 0; //rinit

 auxParents = new int[NumberVars];

  besttree = -1;
  bestgain = 0;

  for(i=0; i < NumberVars-1; i++)
   for(j=i+1; j < NumberVars; j++)
   {
       MNumberTrees = (NumberTrees+1<CNumberTrees)?NumberTrees+1:CNumberTrees;
  
    for(k=MNumberTrees-1; k >=0; k--)
      {
       for(l=0; l < NumberVars; l++)  auxParents[l] = ((AbstractTreeModel*)EveryTree[k])->Tree[l];
       if(EveryTree[k]->Add_Edge(i,j)>-1) 
          {
           if(k==NumberTrees) gain = CalculateGainEnlargedM(k);
           else gain = CalculateGain(k);
           //cout<<i<<" "<<j<<" "<<k<<" "<<gain<<endl;
           if (gain>bestgain)
	     {
	       cout<<i<<" "<<j<<" "<<k<<" "<<gain<<endl;
              besttree = k;
              besta = i;
              bestb = j;
              bestgain = gain;
             }
	  }
        for(l=0; l < NumberVars; l++)  ((AbstractTreeModel*)EveryTree[k])->Tree[l] = auxParents[l] ;
      }
   }   

  if (bestgain>0)
    {
     aux = EveryTree[besttree]->Add_Edge(besta,bestb);
     cout<<"T "<<besttree<<" ("<< besta<<","<<bestb<<")  "<<endl;    
    }
  if (besttree==NumberTrees || besttree==-1 )       
       {
        besttree=NumberTrees;
        NumberTrees++;
        EnlargeMixture();
       }
  delete[] auxParents;
  return besttree;
  
}

double  MixtureTrees::CalculateGain(int k)
{
	int j;
        double lambdaXprob,auxprob,temp,totprob;   
        
     auxprob = 0.0;
     totprob  = 0.0;
              
     for(j=0; j <CNumberPoints; j++) 
	 {
          lambdaXprob  = (EveryTree[k]->Prob(AllPoints->Ind(j),0))*PopProb[j]; 
	  auxprob +=  lambdaXprob;          
         }  
     temp = auxprob;    
     for(j=0; j <NumberTrees; j++) if(j!=k) temp += TreeProb[j];
     for(j=0; j <NumberTrees; j++) if(j!=k) totprob += (TreeProb[j]/temp)*TreeProb[j];
     totprob += (auxprob/temp)*auxprob;   	

     //cout<<"otherprob"<<auxprob<<" lambda "<<(auxprob/temp)<< endl;
     return -1*(log(MeanMixtProb) - log(totprob));


     //     return -1*(log(MeanMixtProb) - log(MeanMixtProb + (auxprob-TreeProb[k])*lambdavalues[k]));
     
  
 }


double  MixtureTrees::CalculateGainEnlargedM(int k)  //This is the case when the number of mixtures is increased
{
	int j;
        double temp,auxprob, totprob;   
        
     auxprob = 0;
     totprob = 0;
            
     for(j=0; j <CNumberPoints; j++) auxprob += (EveryTree[k]->Prob(AllPoints->Ind(j),0)); 	 
     temp = auxprob;    
     for(j=0; j <NumberTrees; j++) temp += TreeProb[j];
     for(j=0; j <NumberTrees; j++) totprob += (TreeProb[j]/temp)*TreeProb[j];
     totprob += (auxprob/temp)*auxprob;
     //cout<<"TreeProb "<<TreeProb[0]<<" lTree "<<(TreeProb[0]/temp)<< " auxprob"<<auxprob<<" lambda "<<(auxprob/temp)<< endl;
     return -1*(log(MeanMixtProb) - log(totprob));
 }


void MixtureTrees::UpdateTotProb(double* TotProb)
{
    int j;
  for(j=0; j <CNumberPoints+1; j++) Probabilities[NumberTrees][j] = TotProb[j];
}

void MixtureTrees::FillProbabilitiesSimple()
{
	int i,j;
        double lambdaXprob;
        MeanMixtProb = 0;
       
	for(j=0; j <CNumberPoints+1; j++) Probabilities[CNumberTrees][j] = 0;
        for(j=0; j <CNumberTrees+1; j++) Probabilities[j][CNumberPoints] = 0; 

    for(i=0; i < CNumberTrees; i++) 
     { 
         TreeProb[i] = 0;
	 for(j=0; j <CNumberPoints; j++) 
	 {      
	  lambdaXprob  = (EveryTree[i]->Prob(AllPoints->Ind(j),0));           	 
	  Probabilities[i][j] = lambdaXprob ;              
          if(i<NumberTrees) 
             {
                Probabilities[CNumberTrees][j] += (Probabilities[i][j])*lambdavalues[i];
                TreeProb[i] += (Probabilities[i][j])*lambdavalues[i];
             }
          Probabilities[i][CNumberPoints] += PopProb[j]*(Probabilities[i][j]);
         }  
        if(i<NumberTrees) Probabilities[CNumberTrees][j] += (Probabilities[i][j])*lambdavalues[i];
          Probabilities[i][CNumberPoints] += PopProb[j]*(Probabilities[i][j]);
        if(i<NumberTrees)  MeanMixtProb += TreeProb[i]*lambdavalues[i];      
     }  
       
    /*   cout<<endl<<"Proba1 "<<endl;
         PrintProb(); 
	 //for(j=0; j <CNumberPoints; j++) cout<<PopProb[j]<<" ";
	 cout<<endl; */
 }

void MixtureTrees::FillProbabilitiesSimple(int k,int update)
{
	int j;
        double lambdaXprob, temp;
       
        MeanMixtProb = 0;
        TreeProb[k] = 0; 
	Probabilities[k][CNumberPoints]=0;
        for(j=0; j <CNumberPoints; j++) 
	 {
     	     if(update)
	     {
              temp = Probabilities[k][j];
	      Probabilities[CNumberTrees][j] -=  temp*lambdavalues[k];
             }
            lambdaXprob  = (EveryTree[k]->Prob(AllPoints->Ind(j),0)); 
             Probabilities[k][j] = (lambdaXprob);  //* lambdavalues[k];  
             Probabilities[CNumberTrees][j] += Probabilities[k][j]*lambdavalues[k]; 
            Probabilities[k][CNumberPoints]+= PopProb[j]*Probabilities[k][j];       
         }  
/*	 cout<<endl<<"Proba "<<endl;
         PrintProb();
	 for(j=0; j <CNumberPoints; j++) cout<<PopProb[j]<<" ";
         cout<<endl; */
  }

void MixtureTrees::FillProbabilities()
{
	int i,j;
        double lambdaXprob;
        MeanMixtProb = 0;
       
	for(j=0; j <CNumberPoints+1; j++) Probabilities[NumberTrees][j] = 0;
        for(j=0; j <NumberTrees+1; j++) Probabilities[j][CNumberPoints] = 0; 


	//cout<<"The number of trees"<<NumberTrees<<endl;
    for(i=0; i < NumberTrees; i++) 
	{
          TreeProb[i] = 0;
     for(j=0; j <CNumberPoints; j++) 
	 {
	  
      // Equation Numb (3.15) in Melina's Thesis
	    lambdaXprob  = (EveryTree[i]->Prob(AllPoints->Ind(j),0));           
	  TreeProb[i] += lambdaXprob;
	  //cout<<i<<"  "<<j<<"  "<<lambdaXprob<<"  "<<TreeProb[i]<<endl;
	  Probabilities[i][j] = PopProb[j]*(lambdaXprob)  * lambdavalues[i];        
          Probabilities[NumberTrees][j] += (Probabilities[i][j]);
         
	 }  
       MeanMixtProb += TreeProb[i]*lambdavalues[i]; 
      
     }
    //cout<<endl<<"Proba "<<endl;
    //PrintProb();
 }

void MixtureTrees::FillEqualProbabilities()
{
	int i,j;
        double lambdaXprob;
     
	for(j=0; j <CNumberPoints+1; j++) Probabilities[NumberTrees][j] = 0;
        for(j=0; j <NumberTrees+1; j++) Probabilities[j][CNumberPoints] = 0; 

    for(i=0; i < NumberTrees; i++)	        
     for(j=0; j <CNumberPoints; j++) 
	 {
	  lambdaXprob  = (EveryTree[i]->Prob(AllPoints->Ind(j),0));           	 
	  Probabilities[i][j] = PopProb[j]*(lambdaXprob);        
          Probabilities[NumberTrees][j] += (Probabilities[i][j]);      
          //if(i<NumberTrees) Probabilities[CNumberTrees][j] += (Probabilities[i][j]);
          Probabilities[i][CNumberPoints] += PopProb[j]*(Probabilities[i][j]);   
	 }  
      
 }


void MixtureTrees::FillProbabilities(int k)
{
	int j;
        double lambdaXprob, temp;
       
        //MeanMixtProb -= TreeProb[k]*lambdavalues[k];

        MeanMixtProb = 0;
        TreeProb[k] = 0; 
      
        for(j=0; j <CNumberPoints; j++) 
	 {
     	    lambdaXprob  = (EveryTree[k]->Prob(AllPoints->Ind(j),0)); 
            //cout<<k<<"  "<<j<<"  "<<lambdaXprob<<endl;
	    TreeProb[k] += lambdaXprob;
            temp = Probabilities[k][j];
	    Probabilities[k][j] = (lambdaXprob);  //* lambdavalues[k];   
            Probabilities[NumberTrees][j] -=  temp;
            Probabilities[NumberTrees][j] += Probabilities[k][j];            
         }  
	for(j=0; j <NumberTrees; j++) MeanMixtProb += TreeProb[j]*lambdavalues[j];   
 
       //MeanMixtProb += TreeProb[k]*lambdavalues[k];

	/* 
     for(i=0; i <NumberTrees+1; i++) 
       {
	 //cout<< TreeProb[i]<<" "<< MeanMixtProb<<endl;
      for(j=0; j <CNumberPoints+1; j++) 
	 {
           cout<<Probabilities[i][j]<<" ";
         }
	cout<<endl;
       }
				   */      
 }

//These 2 functions are added to make compatible with previous implementations

void MixtureTrees::InitNumberTrees()	
{ 
   	
        CNumberTrees = NumberTrees;
        NumberTrees =1; // A mixture of trees has  at least two components	
}

void MixtureTrees::FinishNumberTrees()	
{ 

        NumberTrees = CNumberTrees;
       
}


void MixtureTrees::CalProb()
{
 int i;
 EveryTree[0]->CalProb(); 
  for (i=1;i<NumberTrees;i++)
    {
     ((BinaryTreeModel*)EveryTree[i])->ImportProb(((BinaryTreeModel*)EveryTree[0])->AllSecProb,((BinaryTreeModel*)EveryTree[0])->AllProb);
    }
}

void MixtureTrees::FindICurrentCoefficients()
{ 
	int i;
        double Allprob = 0;
        for(i=0; i < NumberTrees; i++) Allprob += TreeProb[i];
  	for(i=0; i < NumberTrees; i++) 
	  {
	    cout<< lambdavalues[i]<<"  "<< TreeProb[i]<<"  "<<MeanMixtProb<<endl;
            //lambdavalues[i] = lambdavalues[i]*TreeProb[i] / MeanMixtProb ;
              lambdavalues[i] =  TreeProb[i] / Allprob ;
          }

}

void MixtureTrees::EnlargeMixture() //This function creates the new value for the lambdas
{
   int j;
        double auxprob,auxcoefnum,auxcoefden,p,lambdaXprob;      
	//lambdavalues[NumberTrees-1] = 1; 
	//return;
    
    auxcoefnum = 0;
    auxcoefden = 0;
    p = 0;

    for(j=0; j <CNumberPoints; j++)
     {
       auxprob = (EveryTree[NumberTrees]->Prob(AllPoints->Ind(j),0)); 
       lambdaXprob = Prob(AllPoints->Ind(j));
       auxcoefnum  +=  (PopProb[j]-auxprob)*(lambdaXprob-auxprob);
       auxcoefden  +=  (lambdaXprob-auxprob)*(lambdaXprob-auxprob);     
     }

    NumberTrees++; //The number of trees is incremented here because first the prob of the mixt is used.

     if(auxcoefden != 0) p =  auxcoefnum/auxcoefden;
     if( p>0 && p<1)
       {
        for(j=0; j <NumberTrees-1; j++) lambdavalues[j] = p*lambdavalues[j];
        lambdavalues[NumberTrees-1] = 1-p;
       }
     else lambdavalues[NumberTrees-1] = 1;
     //cout<<"p"<<j<<"=  "<<p<<endl;

     /*   	
for(j=0; j <NumberTrees; j++) 
    {
     if(TreeProb[j]==0)
      {
        cout<<"Number of Points "<<CNumberPoints<<endl;
        EveryTree[j]->PrintModel();     
        cout<<endl; 
        EveryTree[j]->PrintProbMod();
        cout<<endl;
        AllPoints->Print();
        cout<<"Lamda"<<j<<"=  "<< lambdavalues[j]<<"TreeProb=  "<< TreeProb[j]<<"  "<<MeanMixtProb<<" auxprob "<<auxprob<<endl;
      }  
     
    }
     */	

    
}

void MixtureTrees::LearningMixtureGreedy()
{
	
  int i,besttree;
 
  Count = 0;
  //this->CalProb(); 
  InitNumberTrees();
  UniformLambdas();
  FillProbabilities();
 
  while ( ! StopCriteria(Count))
   {

    CurrentILikehood =  CalculateILikehood(Count);
    cout<<"Count "<<Count<<"  "<<" Likeh "<<CurrentILikehood<<" MeanMixtProb "<<MeanMixtProb<<endl;
    besttree = NextStructure1();
    FillProbabilities(besttree);//
    //FindGammaValues();
    //NormalizeProb();
    FindICurrentCoefficients();
    Count++;
         
      //printf("%d %d\n ",Count,Prior);
      //printf("%d\n ",LearningSteps);
      //    fprintf(outfile, "%d\n ",Count);	
      //   PrintProb(); 
      for(i=0; i<NumberTrees;i++)  printf("%f\n ",lambdavalues[i]);
      //for(i=0; i<NumberTrees;i++)  fprintf(outfile,"%f\n ",lambdavalues[i]);
         for(i=0; i<NumberTrees;i++)  EveryTree[i]->PrintModel();
      // for(i=0; i<NumberTrees;i++) printf(" %f ", TreeProb[i]);
       //      printf(" MeanProbMixt =  %f \n ",MeanMixtProb);  
       //   printf(" \n %f \n ",LikehoodValues[Count]);  
          // fprintf(outfile,"%f \n ",LikehoodValues[Count]);
      	
    }      
  //EveryTree[0]->PrintProbMod();
    FinishNumberTrees();	
}
// ---------------------------------------------------------

void MixtureTrees::LearningMixtureGreedy1()
{
  double univ_prior, biv_prior;
  int i,improve,goahead;
  Count = 0;
  

  
  InitNumberTrees();
  lambdavalues[0]=1.0;
  FillProbabilities();
  CalculateILikehood(Count);    
/*
     
  cout<<"Iter="<<Count<<" Likehood="<<LikehoodValues[Count]<<" Prob="<<MeanMixtProb<<" "<<endl; 
    
   for(i=0; i<NumberTrees;i++) 
     {
       EveryTree[i]->PrintModel();     
       cout<<endl; 
       //EveryTree[i]->PrintProbMod();
       //cout<<endl;
       cout<<"lambda"<<i<<"="<<lambdavalues[i]<<endl;
     }
    */
  
   Count = 1;  
   //    cout<<"Iter="<<Count<<" Ntrees "<<NumberTrees<<" Likehood="<<LikehoodValues[0]<<" Prob="<<MeanMixtProb<<" "<<endl;
   BestLikehood = LikehoodValues[0];
  
   improve = 2;
   //&& improve > 1
   while (NumberTrees<CNumberTrees && Count<LearningSteps && MaxMixtProb>MeanMixtProb && improve > 1)
   {
           SubstractProb(NumberTrees);
           //NumberTrees++; 
           EnlargeMixture();
           improve = 0;
           goahead = 1;

           while(goahead && Count<LearningSteps) // && MaxMixtProb>MeanMixtProb)
	    {  
	       
             FillProbabilities();
             //PrintProb();  
             FindGammaValues();
             NormalizeProb();
             UpdateTreesProb();
             FindCurrentCoefficients();
             CalculateILikehood(Count);                              
	         	       
             //PrintProb();
             for(i=0; i<NumberTrees;i++) 
             {
		 //EveryTree[i]->PrintModel();     
		 //cout<<endl;
              
	      cout<<"lambda"<<i<<" = "<<lambdavalues[i]<<endl;  
	      cout<<"TreeProb"<<i<<" = "<<TreeProb[i]<<endl; 
	     
             }
	     
	     
	    
              if ((NumberTrees==CNumberTrees) && (improve==0))
		       lsup= (LikehoodValues[Count]-BestLikehood); 

              goahead = (LikehoodValues[Count]-BestLikehood>0.005); //fabs(BestLikehood/double(10000)));
              //goahead = goahead && ((improve==0) ||(NumberTrees == CNumberTrees));  
              if (LikehoodValues[Count]-BestLikehood> 0) BestLikehood = LikehoodValues[Count];
              
            
	      // cout<<"Iter="<<Count<<" Ntrees "<<NumberTrees<<" Improve "<<improve<<" Likehood="<<LikehoodValues[Count]<<" Prob="<<MeanMixtProb<<"BestLikehood= "<<BestLikehood<<endl;                
             improve++;
             Count++;  
      
            }
    }      

  //EveryTree[0]->PrintProbMod();
    BestProb = Prob(AllPoints->P[0]); //Only when the optimum is the first

             switch(Prior) 
                     { 
                       case 0: break; // No prior 
                       case 1: univ_prior = Calculate_Best_Prior(NumberPoints,1,SelInt);
                               biv_prior = Calculate_Best_Prior(NumberPoints,2,SelInt);
                               for(i=0; i < NumberTrees; i++)
                                     EveryTree[i]->SetPrior(univ_prior,biv_prior,NumberPoints);
                               break; // Recommended prior 

                       case 2: for(i=0; i < NumberTrees; i++)
			        {
                                 univ_prior = Calculate_Best_Prior(NumberPoints,1,SelInt*(1+2*TreeProb[i]*(1-lambdavalues[i])));
                                 biv_prior = Calculate_Best_Prior(NumberPoints,2,SelInt*(1+2*TreeProb[i]*(1-lambdavalues[i])));
                                 EveryTree[i]->SetPrior(univ_prior,biv_prior,NumberPoints);
                                }
                               break; // Adaptive Prior                       
                     } 



    FinishNumberTrees();	
}


int MixtureTrees::NextStructure1()
{
 double bestgain,gain;
 int besttree, besta, bestb;
 int i,j,k,l,aux;
 int *auxParents;

besta = 0; bestb = 0; //rinit

 auxParents = new int[NumberVars];

  besttree = -1;
  bestgain = 0;
  
  for(i=0; i < NumberVars-1; i++)
   for(j=i+1; j < NumberVars; j++)
   {
     if (NumberTrees<CNumberTrees)     
      {
       k = NumberTrees-1;
       for(l=0; l < NumberVars; l++)  auxParents[l] = ((BinaryTreeModel*)EveryTree[k])->Tree[l];
       if(EveryTree[k]->Add_Edge(i,j)>-1) 
          {
           gain = CalculateGainEnlargedM(k);
           cout<<"i "<<i<<" j "<<j<<" k "<<k<<" gain  "<<gain<<endl;
           if (gain>bestgain)
	     {
  	      besttree = k;
              besta = i;
              bestb = j;
              bestgain = gain;
             }
	  } else cout<<i<<" "<<j<<" Can not be added"<<endl;
       for(l=0; l < NumberVars; l++)  ((BinaryTreeModel*)EveryTree[k])->Tree[l] = auxParents[l] ;
      }
   }   
     
  if (bestgain>0)
    {
     aux = EveryTree[besttree]->Add_Edge(besta,bestb); 
     cout<<"T "<<besttree<<" ("<< besta<<","<<bestb<<")  "<<endl;    
    }

  /*
   if (besttree==-1)       
       {
        besttree=NumberTrees;
        SubstractProb(besttree);
        NumberTrees++; 
        EnlargeMixture();
       }
  */

  delete[] auxParents;
  return besttree; 
}


void MixtureTrees::SubstractProb(int nexttree)
{	
	int j;
    	double* VectorProb,AllProb1;
        double Totprob,lambdaXprob;

	
     VectorProb = new double[CNumberPoints];
   
     Totprob = 0;
     AllProb1 = 0;
     //PrintProb();
     // cout<<endl;   
     for(j=0; j < CNumberPoints; j++)
       {
            lambdaXprob = Prob(AllPoints->Ind(j));
            //cout<<lambdaXprob<<" ";
            if(PopProb[j] > lambdaXprob) VectorProb[j] = fabs(PopProb[j] - lambdaXprob)* fabs(PopProb[j] - lambdaXprob)*PopProb[j]; 
            else VectorProb[j]=0;
            AllProb1 += lambdaXprob;
            Totprob +=  VectorProb[j];        
       }
     //cout<<endl;      
     //for(j=0; j < CNumberPoints; j++) cout<<PopProb[j]<<" "; 
     //cout<<endl;      
     //cout<<AllProb1<<endl;

     if (Totprob>0)
     { 
       for(j=0; j < CNumberPoints; j++) VectorProb[j] /= Totprob;  
       ((BinaryTreeModel*)EveryTree[nexttree])->rootnode = ((BinaryTreeModel*)EveryTree[nexttree])->RandomRootNode();
       EveryTree[nexttree]->UpdateModel(VectorProb,CNumberPoints,AllPoints);          delete[] VectorProb;     
     }
     else
     {
      ((BinaryTreeModel*)EveryTree[nexttree])->rootnode = ((BinaryTreeModel*)EveryTree[nexttree])->RandomRootNode();
     ((BinaryTreeModel*)EveryTree[nexttree])->MakeRandomTree(((BinaryTreeModel*)EveryTree[nexttree])->rootnode);
     ((BinaryTreeModel*)EveryTree[nexttree])->RandParam();
      }
       
	/* 	
     //EveryTree[nexttree]->CalProbFvect(AllPoints,VectorProb);
     //EveryTree[nexttree]->CalProbFvect(AllPoints,PopProb,CNumberPoints);
     ((BinaryTreeModel*)EveryTree[nexttree])->rootnode = ((BinaryTreeModel*)EveryTree[nexttree])->RandomRootNode();
     ((BinaryTreeModel*)EveryTree[nexttree])->MakeRandomTree(((BinaryTreeModel*)EveryTree[nexttree])->rootnode);
     ((BinaryTreeModel*)EveryTree[nexttree])->RandParam();
	*/
  
}	


   
void MixtureTrees::UpdateCoefAfterAddition_false(int changedtree) //This function updates the new value for the lambdas
{
   int j;
        double auxprob,auxcoefnum,auxcoefden,p,lambdaXprob,derbef,deraft;      
	//lambdavalues[NumberTrees-1] = 1; 
	//return;
    
    auxcoefnum = 0;
    auxcoefden = 0;
    p = 0;
     for(j=0; j <CNumberPoints; j++)
     {        
	 auxprob =  (EveryTree[changedtree]->Prob(AllPoints->Ind(j),0)); //  Probabilities[changedtree][j];  //
	 lambdaXprob = Prob(AllPoints->Ind(j));// Probabilities[CNumberTrees][j]; // //This is the current prob of the mixture (better to store it.    
       auxcoefnum  +=  (lambdaXprob-PopProb[j])*(lambdaXprob-auxprob);
       auxcoefden  +=  (lambdaXprob-auxprob)*(lambdaXprob-auxprob);     
       //cout<<PopProb[j]<<"  "<<Probabilities[changedtree][j]<< "  "<<EveryTree[changedtree]->Prob(AllPoints->Ind(j),0)<<"  "<<Probabilities[CNumberTrees][j]<<" "<<Prob(AllPoints->Ind(j),0)<<" "<<auxcoefnum<<" "<<auxcoefden<<" "<<endl;
     }

    if(auxcoefden != 0) p =  auxcoefnum/auxcoefden;

    derbef = (auxcoefden*(p-1.0)) - auxcoefnum;
    deraft = (auxcoefden*(p+1.0)) - auxcoefnum;
 
    cout<<auxcoefnum<<" "<<auxcoefden<<" p "<<p<<" "<<derbef<<" "<<deraft<<" "<<auxcoefden*p - auxcoefnum<<"  "<<auxcoefden*0.5 - auxcoefnum<<endl;
      
     if( p>0 && p<1)
        for(j=0; j <=1; j++) 
        {
	    if (j != changedtree) baselambdavalues[j] = (1-p); //*baselambdavalues[j];
         else  baselambdavalues[changedtree] = p;
        }           
}



void MixtureTrees::UpdateCoefAfterAddition(int changedtree) //This function updates the new value for the lambdas
{
   int j;
        double auxprob,auxcoefnum,auxcoefden,p,lambdaXprob,derbef,deraft;

	lambdaXprob = 0; //rinit      

	//lambdavalues[NumberTrees-1] = 1; 
	//return;
    
    auxcoefnum = 0;
    auxcoefden = 0;
    p = 0;
     for(j=0; j <CNumberPoints; j++)
     {        
	auxprob =   Probabilities[changedtree][j];  //(EveryTree[changedtree]->Prob(AllPoints->Ind(j),0));        lambdaXprob = Probabilities[CNumberTrees][j]-lambdavalues[changedtree]*auxprob; //Prob(AllPoints->Ind(j),0);  ; //This is the current prob of the mixture (better to store it.    
       auxcoefnum  +=  (lambdaXprob-PopProb[j])*(lambdaXprob-auxprob);
       auxcoefden  +=  (lambdaXprob-auxprob)*(lambdaXprob-auxprob);     
       //cout<<PopProb[j]<<"  "<<Probabilities[changedtree][j]<< "  "<<EveryTree[changedtree]->Prob(AllPoints->Ind(j),0)<<"  "<<Probabilities[CNumberTrees][j]<<" "<<Prob(AllPoints->Ind(j),0)<<" "<<auxcoefnum<<" "<<auxcoefden<<" "<<endl;
     }

    if(auxcoefden != 0) p =  auxcoefnum/auxcoefden;

    derbef = (auxcoefden*(p-1.0)) - auxcoefnum;
    deraft = (auxcoefden*(p+1.0)) - auxcoefnum;
 
    cout<<auxcoefnum<<" "<<auxcoefden<<" p "<<p<<" "<<derbef<<" "<<deraft<<auxcoefden*p - auxcoefnum<<endl;

    
     if( p>0 && p<1)
        for(j=0; j <=1; j++) 
        {
	    if (j != changedtree) baselambdavalues[j] = (1-p)*baselambdavalues[j];
         else  baselambdavalues[changedtree] = p;
        }           
}




void MixtureTrees::UpdateCoefAfterAddition_trad(int changedtree,int MaxTree) //This function updates the new value for the lambdas
{
   int j;
        double auxprob;      
	//cout<<NumberTrees<<endl;
        auxprob = 0;
	for(j=0; j <MaxTree; j++) auxprob +=  Probabilities[j][CNumberPoints];

        for(j=0; j <MaxTree; j++)
         {
          baselambdavalues[j] = Probabilities[j][CNumberPoints]/auxprob;
          //cout<<" lamb"<<j<<" "<<baselambdavalues[j];
         }
	//cout<<endl;       
}

double MixtureTrees::CalculateILikehood(double* baselambda, int ktree, int* bivpos, double* tm, double* rm,int MaxTree)
{   int i,j;
    double auxval,auxval1, Likehood,sumprob;
    Likehood = 0;sumprob=0;
         for(j=0; j < CNumberPoints; j++)
	 {
           auxval1 = 0;
           for(i=0; i < MaxTree; i++)  
	    {          
	     if (i==ktree) auxval = tm[bivpos[j]]*rm[j];    
             else
             auxval  = EveryTree[i]->Prob(AllPoints->Ind(j),0);
	     auxval1 += auxval*baselambda[i];
            } 
            if (auxval1>0)  Likehood  += (PopProb[j]*(log(PopProb[j])-log(auxval1)));  
	    else Likehood  -= (CNumberPoints*PopProb[j]*(log(PopProb[j])));  
            //if (auxval1>0) Likehood  += PopProb[j]*log(auxval1); 
            sumprob+=auxval1;
	    //cout<<" i "<<j<<" prob "<<auxval<<" sumprob "<<sumprob<<" likeh "<<Likehood<<endl;
         }
	 //cout<<"Likehood "<<Likehood<<endl;
	 MeanMixtProb = sumprob;
	    return Likehood;
}


double MixtureTrees::CalculateQuadError(double* baselambda, double* Ai, int ktree, int* bivpos, double* tm, double* rm)
{
    int l; 	
    double err;
    err = 0;
    for(l=0; l <CNumberPoints; l++)
    {
        err += (Ai[l]-baselambda[ktree]*tm[bivpos[l]]*rm[l])* (Ai[l]-baselambda[ktree]*tm[bivpos[l]]*rm[l]);    
	//cout<<" l "<<l<<" PopProb[l] "<<PopProb[l]<<" bl " <<bivpos[l]<<" Ai[l] "<<Ai[l]<<" tm "<<tm[bivpos[l]]<<" rm "<<rm[l]<<" err "<<err<<endl; 
    }
    return err;
}

double MixtureTrees::CalculateMDL(double* baselambda, int ktree, int* bivpos, double* tm, double* rm,int MaxTree,int NewTree, double complexcoef)
{  
    double Likehood,MDL;
    Likehood = CalculateILikehood(baselambda,ktree,bivpos,tm,rm,MaxTree);  
    if (NewTree==1) MDL = Likehood - (complexcoef*log(CNumberPoints)*(MixtComplexity+NumberVars+1))/2;
    else MDL = Likehood - (complexcoef*log(CNumberPoints)*(MixtComplexity+1))/2; 
    //cout<<"L "<<Likehood<<" Comp "<< MDL-Likehood<<"  MDL "<<MDL<<endl;
    return MDL;
}

void MixtureTrees::NewTreeStructure()
{
    int i,j,k,l,kk,MNumberTrees,gain,feasible,aux,rootnode ;
    double A,B,ui,uj;
    int *bivpos;
    double *rm, *tm, *Ai;
    double AB[4], BB[4];
    double** SecProbPros;

    B=0; ui = 0; uj=0; // rinit
    SecProbPros = new double*[4];   
    for(i=0; i < 4; i++) SecProbPros[i] = new double[NumberVars*(NumberVars-1)/2];

    bivpos = new int[CNumberPoints];
    rm = new double[CNumberPoints];
    Ai = new double[CNumberPoints];
    tm = new double[4]; 
    gain = 0;

     ((BinaryTreeModel*)EveryTree[0])->PrintModel();
 
        lambdavalues[0] = 1;
        MNumberTrees = NumberTrees+1;
        for(kk=0;kk<=MNumberTrees-1; kk++) baselambdavalues[kk] = lambdavalues[kk];
	NumberTrees++;
	FindCurrentCoefficients(baselambdavalues,1); //coeftype
        NumberTrees--;
        Printlambdas(baselambdavalues,NumberTrees+1);  
       	FillProbabilitiesSimple();         
        CalcMixtProb(MNumberTrees,baselambdavalues,ExtraTotProb);     
        //baselambdavalues[0] = 0.5; baselambdavalues[1] = 0.5;
	//lambdavalues[0] = 0.5; lambdavalues[1] = 0.5;
    k =  MNumberTrees-1;
    for(i=0; i < NumberVars-1; i++) //NumberVars-1
     { 
      for(j=i+1; j < NumberVars; j++)
      {
	  AB[0]=0.0; AB[1]=0.0; AB[2]=0.0; AB[3]=0.0; 
          BB[0]=0.0; BB[1]=0.0; BB[2]=0.0; BB[3]=0.0;          
          for(l=0; l <CNumberPoints; l++)
          {        
           bivpos[l] =  2*AllPoints->P[l][i]+AllPoints->P[l][j];
           ui = ((BinaryTreeModel*)EveryTree[k])->AllProb[i];  
           uj = ((BinaryTreeModel*)EveryTree[k])->AllProb[j];
	   A = PopProb[l]-(ExtraTotProb[l]-Probabilities[k][l]*baselambdavalues[k]);  
           switch(bivpos[l]) 
           { 
      	    case 0: B = Probabilities[k][l]/( (1-ui)*(1-uj)); break;          
            case 1: B = Probabilities[k][l]/( (1-ui)*(uj)); break;     
            case 2: B = Probabilities[k][l]/(  (ui)*(1-uj)); break;            
            case 3: B = Probabilities[k][l]/(  (ui)*(uj)); break;   
           }                                        
            Ai[l] = A; rm[l] = B; AB[bivpos[l]] += 2*A*B;  BB[bivpos[l]] += 2*B*B;
          }
	
         feasible = FindParams(1,AB,BB,ui,uj,tm); 
         aux = i*(2*NumberVars-i+1)/2 +j-2*i-1; 
         SecProbPros[0][aux] = tm[0]; SecProbPros[1][aux] = tm[1]; 
         SecProbPros[2][aux] = tm[2]; SecProbPros[3][aux] = tm[3];
       } 
      }
    //((BinaryTreeModel*)EveryTree[k])->CleanTree();
   ((BinaryTreeModel*)EveryTree[k])->CalMutInf(SecProbPros);  
   rootnode=((BinaryTreeModel*)EveryTree[k])->rootnode;
    ((BinaryTreeModel*)EveryTree[k])->MakeTree(rootnode);
    ((BinaryTreeModel*)EveryTree[k])->PutBivInTree(SecProbPros);
    ((BinaryTreeModel*)EveryTree[k])->Propagation(1);
    lambdavalues[0] = 1;
    for(kk=0;kk<=MNumberTrees-1; kk++) baselambdavalues[kk] = lambdavalues[kk];
	NumberTrees++;
	FindCurrentCoefficients(baselambdavalues,1); //coeftype	 
        Setlambdas(baselambdavalues,NumberTrees);
        FillProbabilitiesSimple();
        EveryTree[1]->CalculateILikehood(AllPoints,PopProb);  
        NumberTrees--;
        Printlambdas(baselambdavalues,NumberTrees+1);        
    cout<<"  LikehoodNewTree "<<EveryTree[1]->Likehood<<endl; 
    for(i=0; i < 4; i++) delete[] SecProbPros[i];
    delete[] SecProbPros;

   delete[] bivpos;
   delete[] rm; 
   delete[] Ai; 
   delete[] tm; 
 }


void MixtureTrees::Destroy()
{
 int i;
 for(i=0; i<NumberTrees;i++)  EveryTree[i]->Destroy();
}

void MixtureTrees::RemoveTrees()
{
 int i;
 for(i=0; i<NumberTrees;i++) delete EveryTree[i];
}


/********************************* Mixture of integers trees ******************************/


MixtureIntTrees::MixtureIntTrees(int Nvar,int NTrees,int NPoint, int initLamb, int MaxLsteps,double MaxMProb, double Smooth, double SI, int prior, unsigned int* card):MixtureTrees(Nvar,NTrees,NPoint,initLamb,MaxLsteps,MaxMProb,Smooth,SI,prior)
	{
	  Card = card;
	}
     
MixtureIntTrees::~MixtureIntTrees()
{
}


void MixtureIntTrees::MixturesInitMeila(int InitTreeStructure,double* pvect, double Complexity) 
{ 
  int i; 
  IntTreeModel *OneTree; 
 
  //if(InitTreeStructure==0 || InitTreeStructure==1) RandomLambdas(); else	
    UniformLambdas(); 
   
   for(i=0; i<NumberTrees;i++) 
    { 
   
      OneTree =  new IntTreeModel(AllPoints->vars,Complexity,NumberPoints,Card); 
      OneTree->InitTree(InitTreeStructure,CNumberPoints,pvect,AllPoints,NumberPoints);
      EveryTree[i]=OneTree; 
    } 

} 

void MixtureIntTrees::MixturesInitProd(int InitTreeStructure,double* pvect, double Complexity) 
{ 
  int i; 
  IntTreeModel *OneTree; 
  OneTree =  new IntTreeModel(AllPoints->vars, Complexity,NumberPoints,Card); 
  OneTree->InitTree(InitTreeStructure,CNumberPoints,pvect,AllPoints,NumberPoints);
  EveryTree[0]=OneTree;
  
   for(i=1; i<NumberTrees;i++) 
    { 
      OneTree =  new IntTreeModel(AllPoints->vars, Complexity,NumberPoints,Card); 
      OneTree->ImportProbFromTree((IntTreeModel*)EveryTree[i-1]);
      OneTree->ImportMutInfFromTree((IntTreeModel*)EveryTree[i-1]);
      OneTree->PutInMutInfFromTree((IntTreeModel*)EveryTree[i-1]); 
      OneTree->rootnode = OneTree->RandomRootNode();
      OneTree->MakeTree(OneTree->rootnode); 
      EveryTree[i]=OneTree; 
      EveryTree[i]->SetGenPoolLimit(CNumberPoints);
      ((IntTreeModel*)EveryTree[i])->SetNPoints(NumberPoints);
    } 
   PutPriors();
   FindCoefficientsFromMI();
} 
 
void MixtureIntTrees::MixturesInitGreedy(int InitTreeStructure ,double* pvect, double Complexity) 
{ 
  int i;    
  IntTreeModel *OneTree; 
 
  OneTree = new IntTreeModel(AllPoints->vars,Complexity,NumberPoints,Card); 	  
  OneTree->InitTree(2,CNumberPoints,pvect,AllPoints,NumberPoints);
  EveryTree[0] = OneTree;

  for(i=1; i<NumberTrees;i++) 
  { 
    OneTree = new IntTreeModel(AllPoints->vars,Complexity,NumberPoints,Card); 	  
    OneTree->SetGenPoolLimit(CNumberPoints);
    EveryTree[i] = OneTree;
  } 
} 


void MixtureIntTrees::CalProb()
{
 int i;
 EveryTree[0]->CalProb(); 
  for (i=1;i<NumberTrees;i++)
    {
     ((IntTreeModel*)EveryTree[i])->ImportProbFromTree((IntTreeModel*)EveryTree[0]);
	}
}

void MixtureIntTrees::RemoveTrees()
{
    int i;

 for(i=0; i<NumberTrees;i++) delete EveryTree[i];

}

void MixtureIntTrees::SubstractProb(int nexttree)
{	
	int j;
    	double* VectorProb,AllProb1;
        double Totprob,lambdaXprob;

	
     VectorProb = new double[CNumberPoints];
   
     Totprob = 0;
     AllProb1 = 0;
     //PrintProb();
     // cout<<endl;   
     for(j=0; j < CNumberPoints; j++)
       {
            lambdaXprob = Prob(AllPoints->Ind(j));
            //cout<<lambdaXprob<<" ";
            if(PopProb[j] > lambdaXprob) VectorProb[j] = fabs(PopProb[j] - lambdaXprob)* fabs(PopProb[j] - lambdaXprob)*PopProb[j]; 
            else VectorProb[j]=0;
            AllProb1 += lambdaXprob;
            Totprob +=  VectorProb[j];        
       }
     //cout<<endl;      
     //for(j=0; j < CNumberPoints; j++) cout<<PopProb[j]<<" "; 
     //cout<<endl;      
     //cout<<AllProb1<<endl;

     if (Totprob>0)
     { 
       for(j=0; j < CNumberPoints; j++) VectorProb[j] /= Totprob;  
       ((IntTreeModel*)EveryTree[nexttree])->rootnode = ((IntTreeModel*)EveryTree[nexttree])->RandomRootNode();
       EveryTree[nexttree]->UpdateModel(VectorProb,CNumberPoints,AllPoints);          delete[] VectorProb;     
     }
     else
     {
      ((IntTreeModel*)EveryTree[nexttree])->rootnode = ((IntTreeModel*)EveryTree[nexttree])->RandomRootNode();
     ((IntTreeModel*)EveryTree[nexttree])->MakeRandomTree(((IntTreeModel*)EveryTree[nexttree])->rootnode);
     ((IntTreeModel*)EveryTree[nexttree])->RandParam();
      }
       
	/* 	
     //EveryTree[nexttree]->CalProbFvect(AllPoints,VectorProb);
     //EveryTree[nexttree]->CalProbFvect(AllPoints,PopProb,CNumberPoints);
     ((IntTreeModel*)EveryTree[nexttree])->rootnode = ((IntTreeModel*)EveryTree[nexttree])->RandomRootNode();
     ((IntTreeModel*)EveryTree[nexttree])->MakeRandomTree(((IntTreeModel*)EveryTree[nexttree])->rootnode);
     ((IntTreeModel*)EveryTree[nexttree])->RandParam();
	*/
  
}	



