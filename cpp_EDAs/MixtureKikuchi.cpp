#include "auxfunc.h"  
#include "Popul.h"
#include "MixtureKikuchi.h"
#include "FDA.h"
#include <math.h>
#include <time.h>
#include <iostream.h>
#include <fstream.h>  


extern FILE *outfile;

	MixtureRegions::MixtureRegions(int Nvar,int NRegions,int NPoint, int initLamb, int MaxLsteps,double MaxMProb, int prior)
	{
        int i;

		NumberVars = Nvar;
		NumberRegions = NRegions;
		NumberPoints = NPoint;
		InitialLambda = initLamb;
		LearningSteps = MaxLsteps-1;
                MaxMixtProb = MaxMProb;
                Prior = prior;
		EveryRegion = new AbstractProbModel* [NumberRegions];
		lambdavalues = new double[NumberRegions];
                RegionProb = new double[NumberRegions];
		LikehoodValues = new double[LearningSteps]; 
		Probabilities = new double* [NumberRegions+1];
		for(i=0; i < NumberRegions; i++)
                  {
                      RegionProb[i] = 0;
                      lambdavalues[i] = 0;
                   }
                  

                if(Smoothing>0) Smooth_alpha = new double[NumberRegions];
	}


MixtureRegions::~MixtureRegions()
	{
        int i;
		delete[] EveryRegion;
		delete[] lambdavalues;
                delete[] RegionProb;
		delete[] LikehoodValues; 
                for(i=0; i < NumberRegions+1; i++) 
		    if (Probabilities[i] != (double*)0) delete[] Probabilities[i];
   	        delete[] Probabilities;
                if(Smoothing>0) delete[] Smooth_alpha;
	}


void MixtureRegions::SetNpoints(int NPoints, double* pvect)
{
  int i,j;
     CNumberPoints = NPoints;
     for(i=0; i < NumberRegions+1; i++)
       {
         Probabilities[i] = new double[CNumberPoints+1];
         for (j=0; j < CNumberPoints+1; j++) Probabilities[i][j] = 0;
       }
  PopProb = pvect;
}

void MixtureRegions::RemoveProbabilities()
{
  int i;
    for(i=0; i < NumberRegions+1; i++)
       {
	   delete[] Probabilities[i];
           Probabilities[i]  = (double*)0;
       }
}


void MixtureRegions::RandomLambdas()	
{ 
   	int i;
        double aux;
        aux = 0;

	for(i=0; i < NumberRegions; i++) 
         {
          lambdavalues[i] = myrand();
          aux += lambdavalues[i];
         }
	for(i=0; i < NumberRegions; i++) lambdavalues[i] /= aux;
}


void MixtureRegions::UniformLambdas()	
{ // Initially all the  Regions have the same lambda coefficient
   	int i;
	for(i=0; i < NumberRegions; i++) lambdavalues[i] = (1) / double (NumberRegions);
	
}


void MixtureRegions::FindGammaValues()
{
	int i,j;
	for(i=0; i < NumberRegions+1; i++) 
		Probabilities[i][CNumberPoints] = 0;
     
	//cout<<"Numb "<<CNumberPoints<<endl;
	for(i=0; i < NumberRegions; i++) 
	{
  	 for(j=0; j < CNumberPoints; j++) 
	 {
	 // Equation Numb (3.17) in Melina's Thesis
	 if(Probabilities[NumberRegions][j]>0) Probabilities[i][j] = (Probabilities[i][j]/Probabilities[NumberRegions][j])*(NumberPoints*PopProb[j]);
	 else Probabilities[i][j] = 0;
      Probabilities[i][CNumberPoints] +=  Probabilities[i][j];
     }  
    }
	
/*		
  for(i=0; i <NumberRegions+1; i++) 
       {
	 //cout<< RegionProb[i]<<" "<< MeanMixtProb<<endl;
      for(j=0; j <CNumberPoints+1; j++) 
	 {
           cout<<Probabilities[i][j]<<" ";
         }
	cout<<endl;
       }
*/	
}


void MixtureRegions::UpdateTotProb()
{
	int i,j;
	for(i=0; i < CNumberRegions+1; i++) Probabilities[i][CNumberPoints] = 0;
     
	for(i=0; i < NumberRegions; i++) 
	{
         Probabilities[i][CNumberPoints] = 0;
  	 for(j=0; j < CNumberPoints; j++) Probabilities[i][CNumberPoints] +=  Probabilities[i][j];                Probabilities[CNumberRegions][CNumberPoints] += Probabilities[i][CNumberPoints];
        }	
}


void MixtureRegions::NormalizeProb()
{
 
	int i,j;
    for(i=0; i < CNumberPoints; i++) 
	Probabilities[NumberRegions][i] = 0;
	
    for(i=0; i < NumberRegions; i++) 
	{
  	 for(j=0; j < CNumberPoints; j++) 
	 {
        	 // Equation Numb (3.18) in Melina's Thesis
           if( Probabilities[i][CNumberPoints]>0)
	     Probabilities[i][j] /= Probabilities[i][CNumberPoints];
	  else Probabilities[i][j] = 0;
          Probabilities[NumberRegions][j] += Probabilities[i][j];
         }     
        }
           
  for(i=0; i < CNumberPoints; i++) 
     {
		Probabilities[NumberRegions][i] /= NumberRegions ;
		
     }
     
  
//cout<<"Normalized"<<endl;
/*
  for(i=0; i <NumberRegions+1; i++) 
       {
	 //cout<< RegionProb[i]<<" "<< MeanMixtProb<<endl;
      for(j=0; j <CNumberPoints+1; j++) 
	 {
           cout<<Probabilities[i][j]<<" ";
         }
	cout<<endl;
       }
  
*/  
}

void MixtureRegions::PrintProb()
{
	int i,j;
    
   
    for(i=0; i <= NumberRegions; i++) 
      {
        for(j=0; j <= CNumberPoints; j++) 
	  {
	    
            cout<<Probabilities[i][j]<<" ";
          }
	cout<<endl;
      }
  }

double MixtureRegions::Prob (unsigned* vector)
 {
  // The probability of the vector given the Mixture of trees
 // is calculated

  double auxprob;
  int i;

  auxprob =0;
   for(i=0; i < NumberRegions; i++) 
	{
	  auxprob += EveryRegion[i]->Prob(vector,0)*lambdavalues[i]; 
	}
  return auxprob;
}
  
void MixtureRegions::UpdateRegionsProb()
{	
	int i,j;
    	double* VectorProb;
        for(i=0; i < NumberRegions; i++) EveryRegion[i]->UpdateModel(Probabilities[i],CNumberPoints,AllPoints); 
    
}

void MixtureRegions::FindCurrentCoefficients()
{ 
	int i;

  	for(i=0; i < NumberRegions; i++) 
		lambdavalues[i] = Probabilities[i][CNumberPoints] / NumberPoints;

}

void MixtureRegions::LearningMixture(int Type)
{
switch(Type) 
   { 
     case 1: LearningMixtureMeila(); break; // Classical learning mixture method     case 3:  break;
   }
}

void MixtureRegions::MixturesInit(int Type,int InitRegionStructure, double* pvect, double Complexity) 
{
   case 1: MixturesInitMeila(InitRegionStructure,pvect,Complexity); break; 
     
}


void MixtureRegions::MixturesInitMeila(int InitRegionStructure,double* pvect, double Complexity) 
{ 
  int i; 
  BinaryRegionModel *OneRegion; 
 
  // if(InitRegionStructure==0 || InitRegionStructure==1) RandomLambdas();  else 
  UniformLambdas(); 
   for(i=0; i<NumberRegions;i++) 
    { 
      OneRegion =  new BinaryRegionModel(AllPoints->vars,Complexity,NumberPoints); 
      if(NumberRegions==1) OneRegion->InitRegion(3,CNumberPoints,pvect,AllPoints,NumberPoints);
      else  OneRegion->InitRegion(InitRegionStructure,CNumberPoints,pvect,AllPoints,NumberPoints);
      //OneRegion->PrintModel();
      EveryRegion[i]=OneRegion; 
    } 
} 

 void MixtureRegions::LearningMixtureMeila()
 {
	 // Here the Mixture has been created and initialized we initial trees 
   int i,goahead;
 
  goahead = 1;
  Count = 0;
  MeanMixtProb = 0;
  //  AllPoints->Print();

    while ( Count<LearningSteps && goahead && MaxMixtProb>MeanMixtProb) 
	{
         
	  if(Count>0) FindCurrentCoefficients();
	  
	  FillProbabilities();
 	  FindGammaValues();
          NormalizeProb(); 
	  //UpdateForestProb(); 
          UpdateRegionsProb();	    
	  CalculateILikehood(Count); 

          goahead = (Count==0) || (LikehoodValues[Count]-BestLikehood> 0.5); //fabs(BestLikehood/double(10000)));
          if(Count==0) BestLikehood = LikehoodValues[0];
          else if (LikehoodValues[Count]-BestLikehood> 0) BestLikehood = LikehoodValues[Count];
	  
	  /*  
        for(i=0; i<NumberRegions;i++)
            {              
			 cout<<endl;
               cout<<"lambda"<<i<<" = "<<lambdavalues[i]<<endl;  
               cout<<"RegionProb"<<i<<" = "<<RegionProb[i]<<endl; 
		 
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


double MixtureRegions::CalculateILikehood(int step)
{ 	
 int i,j;
    double auxval,auxval1,sumprob;
	
	LikehoodValues[step] = 0;
        sumprob = 0;  
        for(i=0; i < NumberRegions; i++)  RegionProb[i] = 0;      
        //MeanMixtProb  =0 ;
         
        for(j=0; j < CNumberPoints; j++)
	{
         auxval1 = 0;
         for(i=0; i < NumberRegions; i++)  
	  {          
	   auxval  = (EveryRegion[i]->Prob(AllPoints->Ind(j),0));
           auxval1 += auxval*lambdavalues[i];
           RegionProb[i] += auxval; //Modificado
          }
	   
	 // if (auxval1>0) LikehoodValues[step]  += (PopProb[j]*(log(PopProb[j])-log(auxval1)));     //else  LikehoodValues[step]  -= (CNumberPoints*PopProb[j]*log(PopProb[j]));   

        if (auxval1>0) LikehoodValues[step]  += (NumberPoints*PopProb[j]*(log(auxval1)));   
         sumprob += auxval1;
         //cout<<" j "<<j<<"RealProb "<<PopProb[j]<<" prob "<<auxval<<" sumprob "<<sumprob<<" likeh "<<LikehoodValues[step]<<endl;
         }
         MeanMixtProb = sumprob;
	return LikehoodValues[step];
}


 
double MixtureRegions::CalculateOptimalLikehood()
{ 	
 int j;
 double Likehood;
 Likehood = 0;
         for(j=0; j < CNumberPoints; j++)  
            Likehood  -= PopProb[j]*log(PopProb[j]);            
	 return Likehood;
}

void MixtureRegions::SetPop(Popul* pop)
 { 
	 AllPoints = pop;
 }

void MixtureRegions::SetRegions(int pos, AbstractProbModel* tree)
 { 
	 EveryRegion[pos] = tree;
 }

void MixtureRegions::SamplingFromMixture(Popul *FinalPoints)
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
	 while ( (cutoff>tot) && (j+1<NumberRegions) )
	 { 
		 j++;
		 tot += lambdavalues[j];
     } 
     EveryRegion[j]->GenIndividual(FinalPoints,i,1);
   } 
}


void MixtureRegions::CalcMixtProb(int NRegions, double* lambdas,double* ExtraProbs)
{
	int i,j;
        double lambdaXprob;
    
	for(j=0; j <CNumberPoints+1; j++) ExtraProbs[j] = 0;
       
    for(i=0; i < NRegions; i++) 
     {
       for(j=0; j <CNumberPoints; j++) 
	 {      
	  lambdaXprob  = (EveryRegion[i]->Prob(AllPoints->Ind(j),0));           	      
          ExtraProbs[j] += lambdaXprob * lambdas[i];
         }         
     }   
}

void MixtureRegions::FillProbabilities()
{
	int i,j;
        double lambdaXprob;
        MeanMixtProb = 0;
       
	for(j=0; j <CNumberPoints+1; j++) Probabilities[NumberRegions][j] = 0;
        for(j=0; j <NumberRegions+1; j++) Probabilities[j][CNumberPoints] = 0; 


	//cout<<"The number of trees"<<NumberRegions<<endl;
    for(i=0; i < NumberRegions; i++) 
	{
          RegionProb[i] = 0;
     for(j=0; j <CNumberPoints; j++) 
	 {
	  
      // Equation Numb (3.15) in Melina's Thesis
	    lambdaXprob  = (EveryRegion[i]->Prob(AllPoints->Ind(j),0));           
	    //    cout<<i<<"  "<<j<<"  "<<lambdaXprob<<endl;
	  RegionProb[i] += lambdaXprob;
	  Probabilities[i][j] = PopProb[j]*(lambdaXprob)  * lambdavalues[i];        
          Probabilities[NumberRegions][j] += (Probabilities[i][j]);
         
	 }  
       MeanMixtProb += RegionProb[i]*lambdavalues[i]; 
      
     }
    //cout<<endl<<"Proba "<<endl;
    //PrintProb();
 }

void MixtureRegions::FillProbabilities(int k)
{
	int j;
        double lambdaXprob, temp;
       
        //MeanMixtProb -= RegionProb[k]*lambdavalues[k];

        MeanMixtProb = 0;
        RegionProb[k] = 0; 
      
        for(j=0; j <CNumberPoints; j++) 
	 {
     	    lambdaXprob  = (EveryRegion[k]->Prob(AllPoints->Ind(j),0)); 
            //cout<<k<<"  "<<j<<"  "<<lambdaXprob<<endl;
	    RegionProb[k] += lambdaXprob;
            temp = Probabilities[k][j];
	    Probabilities[k][j] = (lambdaXprob);  //* lambdavalues[k];   
            Probabilities[NumberRegions][j] -=  temp;
            Probabilities[NumberRegions][j] += Probabilities[k][j];            
         }  
	for(j=0; j <NumberRegions; j++) MeanMixtProb += RegionProb[j]*lambdavalues[j];   
 
       //MeanMixtProb += RegionProb[k]*lambdavalues[k];

	/* 
     for(i=0; i <NumberRegions+1; i++) 
       {
	 //cout<< RegionProb[i]<<" "<< MeanMixtProb<<endl;
      for(j=0; j <CNumberPoints+1; j++) 
	 {
           cout<<Probabilities[i][j]<<" ";
         }
	cout<<endl;
       }
				   */      
 }

void MixtureRegions::RemoveRegions()
{
 int i;
 for(i=0; i<NumberRegions;i++) delete EveryRegion[i];
}
