#ifndef __GUMDA_H 
#define __GUMDA_H 

#include "AbstractTree.h" 

class Gaussian {
 
       double *mean;
       double *std;
       double *trunc;
       double t1;
       double t2;
       double s2;
       double c;
       double xbar;
       double s;
       double mu;
       double sigma;
       double fc;
       double xc; 

}

 class G_UnivariateModel:public AbstractProbModel { 
	 public:     
	  
     double tval;
     double *AllProb;   // Univariate Probabilities of the individuals 

     Gaussian**  AllCondGaussians;
     Gaussian*   FullGaussian;
  
	   
	  UnivariateModel(int,int*,int,Popul*); 
          UnivariateModel(int,int*,int); 
          UnivariateModel(int); 
         virtual	 ~UnivariateModel(); 
         virtual void SetPrior(double,double,int); 
	 virtual double Prob(unsigned*); 
	 virtual void CalProb(); 
   	 virtual void CalProbFvect(double*); 
         virtual void CalProbFvect(Popul*,double*,int); 
         void ResetProb(); 
	 virtual void GenIndividual (Popul*,int); 
         virtual void GenPartialIndividual (Popul*,int,int,int*){}; 
	 virtual void UpdateModel(); 
	 //virtual void PopMutation(Popul*,int,int, double); 
	 //void IndividualMutation(Popul*,int,int,int);  
         virtual void RandParam(){};   
         double FindTruncGauss(double*); // Find the truncation parameter from the values


        }; 


#endif 

