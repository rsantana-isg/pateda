#include "Popul.h" 
#include "AbstractTree.h" 
 
class AdditiveConstraints{ 
	 public:     
	 
	 int typeconstraint; 
     int minunit; 
     int maxunit; 
	 int unitdim; 
	 int cantvars; 
	 int threshold; 
	 double* UnitValues; 
	 double**  unitprob; 
	 
   
	 AdditiveConstraints(int,int,int,int,int); 
	 ~AdditiveConstraints(); 
	 
 
	 void ResetProb(); 
	 void ResetUnitValues(); 
	 int Find_Unit_tobeset(); 
	 virtual void InitPop(Popul*); 
	 //virtual void InitPop(Popul* ); 
	 void InitUnitationValues(); 
     void InitUnivariateMarginals(); 
	 void GenInitIndividual (Popul*,int);	   
	 void GenInitPop(int, Popul*); 
	  
	}; 
 
  
class SConstraintUnivariateModel:public UnivariateModel, public AdditiveConstraints { 
	 public:     
	 double* AuxUnitValues; 
 
 	 SConstraintUnivariateModel(int,int*,int,Popul*,int,int,int); 
	 ~SConstraintUnivariateModel(); 
 
	 virtual void CalProb(); 
   	 virtual void CalProbFvect(double*){}; 
	 virtual double Prob(unsigned*); 
	 virtual void GenIndividual (Popul*,int);	   
	 void InitPop(Popul*); 
	 void PopMutation(Popul*,int,int, double); 
	 void IndividualMutation(Popul*,int,int,int); 
      
	}; 
 
class CConstraintUnivariateModel:public UnivariateModel, public AdditiveConstraints{ 
	 public:     
	 
	  
	  CConstraintUnivariateModel(int,int*,int,Popul*,int,int,int); 
	 ~CConstraintUnivariateModel(); 
	 
	 virtual void CalProb(); 
   	 virtual void CalProbFvect(double*){}; 
	 virtual double Prob(unsigned*); 
	 virtual void GenIndividual (Popul*,int); 
	 void InitPop(Popul*); 
	}; 
 
class ConstraintBinaryTreeModel:public BinaryTreeModel, public AdditiveConstraints{ 
	 public:
	   
	  ConstraintBinaryTreeModel(int,int*,int,Popul*,int,int); 
	   
	 ~ConstraintBinaryTreeModel(); 
	 virtual double Prob(unsigned*); 
	 virtual void CalProb(); 
	 virtual void GenIndividual (Popul*,int); 
	 void InitPop(Popul*); 
	 }; 
