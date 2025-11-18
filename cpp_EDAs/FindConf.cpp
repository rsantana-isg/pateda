#include "jtPartition.h" 
//#include "jtPart~1.h" 
 
int  FindMaxConf (pop* Mypop, jtree *jt,pop* otherpop,int cantconf ) 
 
 { 
   
  
     JtreeTable *Tree = new JtreeTable(jt->getNoVars(),jt->getNoCliques(),otherpop->get_card()); 
 
     Tree->convert(jt);  
      
     Tree->Compute(Mypop); 
         
     Tree->PassingFluxesTopDown(); 
 
     Tree->PassingFluxesBottomUp(); 
 
     Tree->FindBestConf(); 
 
     double  aux = Tree->GetMaxProb(); 
 
     JtPartition*  Pt = new JtPartition(cantconf,Tree->getNoVars(),Tree->getNoCliques()); 
        
     Pt->Add(Tree,aux); 
   
     Pt->Cycle(); 
 
     cantconf = (Pt->GetCantConf()+1 > otherpop->Getsize()) ? otherpop->Getsize() :Pt->GetCantConf()+1;  
  
     //printf("Popsizes %d , %d\n",cantconf ,otherpop->Getsize() );      
 
     Pt->SetPop(otherpop,cantconf); 
 
     //otherpop->print(); 
         
     delete Pt; 
     
     return cantconf; 
 
  } 
 
 
 
 
 
 
 
 
