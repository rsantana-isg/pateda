#include <iostream> 
#include <fstream> 
#include <math.h> 
#include <memory> 
#include <vector>
#include "InferenceClass.h"
#include "auxfunc.h"






//************************************************************************************
 
/*  
VectCandNode::VectCandNode(CandidateNode* vnode, VectCandNode* p, VectCandNode* n)
{
  content = vnode;
  previous = p;
  next = n;
}  

VectCandNode::~VectCandNode()
{
  delete content;
}


VectCandidate::VectCandidate()
{
 first =  0;
 current = 0;
 last  =  0;
}

void VectCandidate::push_back(CandidateNode* node)
{
  VectCandNode* auxptr;

  if (last ==  0)
    {
      auxptr = new VectCandNode(node,(VectCandNode*)0,(VectCandNode*)0);
      first =  auxptr;
      current =  auxptr;
      last =  auxptr; 
    } 
  else
    {
      auxptr = new VectCandNode(node,last,(VectCandNode*)0);
      last->next = auxptr;
      last =  auxptr;    
    } 
  //  cout<<"Insert "<<node->v<<"  "<<first<<"  "<<current<<" "<<last<<"  "<<last->previous<<"  "<<last->next<<endl;
}

void VectCandidate::begin()
 { 
   current = first;
 }

void VectCandidate::end()
 { 
   current = last;
 }


void VectCandidate::print()
{ 
  int i;
  i = 0;
  begin();
  while (current != 0)
    {
      cout<<i<<"-----------"<<current<<"  "<<current->content->v<<endl;
      current = current->next; 
      i++; 
    }

}

void VectCandidate::erase(VectCandNode* cnode)
 { 

  
   if(first == cnode && last == cnode)
     {
       first =  0; current = 0; last  =  0;
       //cout<<"After "<<"  "<<first<<"  "<<current<<" "<<last<<endl;
     }
   else
   {
    if(first == cnode) 
      {
        first =  cnode->next;
        cnode->next->previous = 0;
        current = first;
        delete cnode;
        return;
  
      }    
     else  
     {
       cnode->previous->next = cnode->next;     
     }

   if(last == cnode)
     {
        last = cnode->previous;
        current = last;
      }
   else
    { 
      cnode->next->previous = cnode->previous;
      current = cnode->next; 
    }   
   //cout<<"After "<<current->content->v<<"  "<<first<<"  "<<current<<" "<<last<<"  "<<current->previous<<"  "<<current->next<<endl;

   }
   
   delete cnode;
 }


int VectCandidate::empty()
 { 
   return(last==0);
 }

void VectCandidate::sort()
 { 
   CandidateNode* auxcandnode;  
   double bestval;
   int notend;
   VectCandNode* auxcurrent;
   VectCandNode* auxptr;

   VectCandidate* auxlist;
   auxlist = new VectCandidate();

   auxlist->begin();
   int i;
   i = 0;
   while (last !=  0)
   {
     begin();
     bestval = -1;
     notend = ! empty();

     while(notend)
      {
	//  cout<<current->content->v<<"  "<<bestval<<"  "<<notend<<" "<<first<<"  "<<current<<" "<<last<<endl;
       if (current->content->v > bestval)
	 {     
           bestval = current->content->v;
           auxcurrent = current;         
         }
       notend = (current != last);
       if (notend)  current = current->next; 
      }
     //cout<<i<<endl;
      auxlist->push_back(auxcurrent->content);
      erase(auxcurrent);
      i++;
   }
   //cout<<"Finished inserting "<<endl;
   //auxlist->print();
   //cout<<"Arrived here "<<notend<<endl;
   
   i = 0;
   auxlist->begin();
   while (auxlist->last != 0)
   {
   double valv = auxlist->current->content->v;
   int nod = auxlist->current->content->node;
   int sta = auxlist->current->content->state;
   int initB = auxlist->current->content->initBN;
   int ver = auxlist->current->content->verified;
   auxlist->erase(auxlist->current);     

     auxcandnode = new CandidateNode(valv,nod,sta,initB,ver);
     push_back(auxcandnode);    
     cout<<"Ordered "<<i<<" "<<auxcandnode->v<<"  "<<endl;
     i++;
  
    
     //auxptr = auxlist->current;
     //cout<<endl<<"Pass "<<i<<endl;
     //auxlist->print();
     //auxlist->current = auxptr; 
     // cout<<" The content is "<<auxlist->current->content->v<<endl;
   }
  
   delete auxlist;  
   cout<<"Finished Ok"<<endl;
 }



VectCandidate::~VectCandidate()
 {     
   begin();
   while (last !=  0)
   {
     erase(current);  
   }
 }

*/
// *****************************************************************************
// fillAdjMat
// *****************************************************************************



void  InferenceClass::fillAdjMat(unsigned int** Mat, vector<Nodes>& adjMat)
 {
   int i,j,k,neighNum_i;

  adjMat.resize(num_nodes);
  
  for (i=0; i<num_nodes; i++) 
    {
      neighNum_i = 0;
      for (j=0; j<num_nodes; j++) if(i != j) neighNum_i += Mat[i][j];
      //cout<<i<<" "<<neighNum_i<<endl;
      adjMat[i].resize(neighNum_i);
      k = 0;
      for (j=0; j<num_nodes; j++) 
         if (i != j && Mat[i][j]==1)
          {
	   adjMat[i][k] = j;
           //cout<<i<<"  "<<k<<" --->  "<<j<<endl;
           k++;
          }
     }
  // for(int i=0;i<num_nodes;i++)    cout<<i<<" "<< adjMat[i].size()<<endl;
  }



// *****************************************************************************
// CopyAdjMat
// *****************************************************************************

void  InferenceClass::CopyAdjMat(vector<Nodes>& Orig_adjMat, vector<Nodes>& adjMat)
 {
   int i,j;

 
  adjMat.resize(num_nodes);
  
  
  for (i=0; i<num_nodes; i++) 
    {
       adjMat[i].resize(Orig_adjMat[i].size());
       for (j=0; j<adjMat[i].size(); j++)  adjMat[i][j] = Orig_adjMat[i][j];     
     }

    
  //for(int i=0;i<num_nodes;i++)    cout<<i<<" "<< adjMat[i].size()<<endl;
  }



// *****************************************************************************
// fillLocalMat
// *****************************************************************************

void  InferenceClass::fillLocalMat(unsigned int* Card, double** LocalPot, MRF* mrf) 
{
  int i,xi;
  int const& N = mrf->N;
  MaxCardNumber = 0;

  // get V
  for (i=0; i<N; i++) 
    {
     mrf->V[i] = Card[i];
     if (mrf->V[i]>MaxCardNumber) MaxCardNumber = mrf->V[i];
    }

  mrf->initLocalPotentials();

  // fill localMat
  for (i=0; i<N; i++) 
  {
    for (xi=0; xi<mrf->V[i]; xi++) 
     {
      mrf->localMat[i][xi] = LocalPot[i][xi];
      //cout<<i<<" "<<xi<<" "<<mrf->localMat[i][xi]<<endl;
     }
    //cout<<endl;
    //cout<<endl;
  }  
}




// *****************************************************************************
// CopyLocalMat
// *****************************************************************************

void  InferenceClass::CopyLocalMat(MRF* Orig_mrf) 
{
  int i,xi;
  int const& N = main_mrf->N;
  
  // get V
  for (i=0; i<N; i++) 
    {
     main_mrf->V[i] = Orig_mrf->V[i];
    }

  main_mrf->initLocalPotentials();

  // fill localMat
  for (i=0; i<N; i++) 
  {
    for (xi=0; xi<main_mrf->V[i]; xi++) 
     {
       main_mrf->localMat[i][xi] = Orig_mrf->localMat[i][xi];
     }    
  }  
}



// *****************************************************************************
// fillPsiMat
// *****************************************************************************

void  InferenceClass::fillPsiMat(double*** Psi, MRF* mrf) 
{
  mrf->initPairPotentials();
  
  int const& N = mrf->N;
  int const* V = mrf->V;
  
   // fill lambdaMat
  for (int i=0; i<N; i++) 
   {
      for (int n=0; n<mrf->neighbNum(i); n++) 
       {
        int j = mrf->adjMat[i][n];
        if (i<j) 
         {
           PairPotentials pairPot_ij = new Potentials[V[i]];
 	   for (int xi=0; xi<V[i]; xi++) 
            {
	      pairPot_ij[xi] = new Potential[V[j]];
	      for (int xj=0; xj<V[j]; xj++) 
               {
		 //cout<<"i "<<i<<" j "<<j<<" xi "<<xi<<" xj "<<xj<<" "<<Psi[i][j][xj+xi*V[j]]<<endl;
		 //pairPot_ij[xi][xj] =  Psi[i][j][xi+xj*V[i]];
                   pairPot_ij[xi][xj] =  Psi[i][j][xj+xi*V[j]];
		 //Psi[ProteinContacts[0][j]][ProteinContacts[1][j]][k*RotNumber[ProteinContacts[1][j]]+l]); 
	       }
	     }
	    mrf->assignPairPotential(i, n, pairPot_ij);
	    for (int xi=0; xi<V[i]; xi++)   delete[] pairPot_ij[xi];
	    delete[] pairPot_ij;
	    pairPot_ij = 0;
           }
        }
    }
}



// *****************************************************************************
// CopyPsiMat
// *****************************************************************************

void  InferenceClass::CopyPsiMat(MRF* Orig_mrf) 
{
  main_mrf->initPairPotentials();
  
  int const& N = Orig_mrf->N;
  int const* V = Orig_mrf->V;
  
   // fill lambdaMat
  for (int i=0; i<N; i++) 
   {  
     for (int n=0; n<Orig_mrf->neighbNum(i); n++) 
       {
        int j = Orig_mrf->adjMat[i][n];
        if (i<j) 
         {
           PairPotentials pairPot_ij = new Potentials[V[i]];
 	   for (int xi=0; xi<V[i]; xi++) 
            {
	      pairPot_ij[xi] = new Potential[V[j]];
	      for (int xj=0; xj<V[j]; xj++) 
               {
	             pairPot_ij[xi][xj] =  Orig_mrf->lambdaMat[i][n][xi][xj];
	
	       }
	     }

	    main_mrf->assignPairPotential(i, n, pairPot_ij);
	    for (int xi=0; xi<V[i]; xi++)   delete[] pairPot_ij[xi];
	    delete[] pairPot_ij;
	    pairPot_ij = 0;
           }
        }
    }
}


// *****************************************************************************
// fillLambdaMat
// *****************************************************************************

void InferenceClass::fillLambdaMat(double** Lambda, PottsMRF* mrf) 
{
  mrf->initLambdaValues();
  int const& N = mrf->N; 

  // fill lambdaMat
  for (int i=0; i<N; i++) 
   {  
    for (int n=0; n<mrf->neighbNum(i); n++) 
     {
      mrf->lambdaValues[i][n] = Lambda[i][n];
     }
   }  
}




// *****************************************************************************
// CopyLambdaMat
// *****************************************************************************

void InferenceClass::CopyLambdaMat(PottsMRF* Orig_mrf) 
{
  ((PottsMRF*)main_mrf)->initLambdaValues();
    
  int const& N = main_mrf->N; 

  // fill lambdaMat
  for (int i=0; i<N; i++) 
   {  
    for (int n=0; n<main_mrf->neighbNum(i); n++) 
     {
      ((PottsMRF*)main_mrf)->lambdaValues[i][n] = Orig_mrf->lambdaValues[i][n];
     }
   }  
}




// *****************************************************************************
// fillInitialAssignment
// *****************************************************************************

void InferenceClass::fillInitialAssignment(int* initX) 
{
   // fill startX
  for (int i=0; i<num_nodes; i++)
 {
    startX[i] = initX[i];
  }
}



// *****************************************************************************
// fillRegions
// *****************************************************************************

void InferenceClass::fillRegions(int numRegs, int* regionsizes, int** AllInitRegions, RegionLevel& regions) 
{
  // fill regions
  int node;

  for (int i=0; i<numRegs; i++) 
   {
    int numNodes_i = regionsizes[i];
    Nodes nodes;
    nodes.clear();
    for (int n=0; n<numNodes_i; n++)
     {
      node = AllInitRegions[i][n];
      nodes.push_back(node);
     }
    Region reg;
    reg.assignNodes(nodes);
    if (!regions.addRegion(reg))  cout<<"one region cannot be a subset of another"<<endl;
  }
}

InferenceClass::InferenceClass(algorithmType algType,Model modl, int nnodes, unsigned int** Matrix, unsigned int *Card, double** LocalPot, double***Psi, double** Lambda, double temp)
    {
    
      model = modl;  
      algo_type = algType;
      num_nodes = nnodes;

      algorithm = 0;    
      singleBeliefs = 0;
      pairBeliefs = 0;

      assignInd = 0;
      bethe = 0;
      processor = 0;
      reg_mrf = 0;
      main_mrf = 0;
      regions = 0;
      allRegions = 0;
      allLevels = false;
      trw = false;
      temperature = temp;

      startX = 0;

 
       adjMatA  = new vector<Nodes>();
       fillAdjMat(Matrix,*adjMatA);  
   
      //  for(int i=0;i<num_nodes;i++)    cout<<i<<" "<< adjMat[i].size()<<endl;
  /*
 int i,j,k,neighNum_i;
  
  adjMat->resize(num_nodes);
  
  for (i=0; i<num_nodes; i++) 
    {
      neighNum_i = 0;
      for (j=0; j<num_nodes; j++) if(i != j) neighNum_i += Matrix[i][j];
      //cout<<i<<" "<<neighNum_i<<endl;
      adjMat[i].resize(neighNum_i);
      k = 0;
      for (j=0; j<num_nodes; j++) 
         if (i != j && Matrix[i][j]==1)
          {
	   adjMat[i][k] = j;
           k++;
          }
     }
       */
     
      potts_model = ((model==POTTS) || (algo_type==AT_WOLFF) || (algo_type==AT_SWENDSEN_WANG));
      monte_carlo = ((algo_type==AT_GIBBS) || (algo_type==AT_WOLFF) || (algo_type==AT_SWENDSEN_WANG) || (algo_type==AT_METROPOLIS));   
 
      // define the MRF
          
      if (potts_model)
      {
	 main_mrf = new PottsMRF(*adjMatA);
      }
      else 
      {
	main_mrf = new MRF(*adjMatA);
       
      }
      
  
     // get local potentials
      fillLocalMat(Card,LocalPot,main_mrf);

      
   // get pairwise potentials

  if (potts_model)
    {
     fillLambdaMat(Lambda, (PottsMRF*) main_mrf); 
    }
  else 
   {
    fillPsiMat(Psi, main_mrf);
   }
      
 // get tepmerature
   main_mrf->setTemperature(temperature);  
  
}
 


InferenceClass::InferenceClass(InferenceClass* InferenceEngine)
    {
   
      model = InferenceEngine->model;  
      algo_type = InferenceEngine->algo_type;
      num_nodes = InferenceEngine->num_nodes;
      MaxCardNumber = InferenceEngine->MaxCardNumber;
      temperature = InferenceEngine->temperature;

      algorithm = 0;    
      singleBeliefs = 0;
      pairBeliefs = 0;
      assignInd = 0;
      bethe = 0;
      processor = 0;
      reg_mrf = 0;
      main_mrf = 0;
      regions = 0;
      allRegions = 0;
      allLevels = false;
      trw = false;
      startX = 0;

    
      adjMatA  = new vector<Nodes>();
      CopyAdjMat(*InferenceEngine->adjMatA,*adjMatA);  
    
       /*
   int i,j;
    adjMat.resize(num_nodes);
    cout<<"Arrived in Here "<<adjMatA.size();  
    for (i=0; i<num_nodes; i++) 
     { 
       cout<<i<<" "<< InferenceEngine->adjMatA[i].size()<<endl;
       adjMatA[i].resize(InferenceEngine->adjMatA[i].size());
       for (j=0; j<InferenceEngine->adjMatA[i].size(); j++)  adjMatA[i][j] = InferenceEngine->adjMatA[i][j];     
     }
       */
 


      potts_model = InferenceEngine->potts_model;
      monte_carlo = InferenceEngine->monte_carlo ;
   
      
      // define the MRF
      
      if (potts_model)
      {
        main_mrf = new PottsMRF(*adjMatA);
      }
      else 
      {
        main_mrf = new MRF(*adjMatA);
        
      }
      
    
     // get local potentials
      CopyLocalMat(InferenceEngine->main_mrf);
      
   // get pairwise potentials


  if (potts_model)
    {
     CopyLambdaMat((PottsMRF*) InferenceEngine->main_mrf); 
    }
  else 
   {
    CopyPsiMat(InferenceEngine->main_mrf);
   }
 // get tepmerature
   main_mrf->setTemperature(InferenceEngine->temperature);  

  
}



void InferenceClass::SetMonteCarlo(int  burnT, int samplingI, int num_s, int* initassign)
{
   
  // For monte-carlo algorithms (gibbs, wolff, swendsen-wang), get
  // the initial state and the sampling parameters
 
    burningTime = burnT;
    samplingInterval = samplingI;
    num_samples = num_s;
    startX = new int[num_nodes];
    fillInitialAssignment(initassign);
}




void InferenceClass::SetLoopy(int maxI, SumOrMax SOM , Strategy  stra)
 {    
        // for loopy belief propagation:
	// get sum-or-max-flag and strategy
     
        maxIter = maxI;
        sumOrMax = SOM;
        strategy  = stra;

  
 }

void InferenceClass::SetGBP(int maxI, SumOrMax SOM , double alpha, bool tr, int numRegs, int* regionsizes, int** AllInitRegions)
{
  
        // for generalized belief propagation:
	// get regions, regions-adj (if given), sum-or-max-flag
	// and alpha

        maxIter = maxI;
        sumOrMax = SOM;
        gbp_alpha  = alpha;
        trw = tr;
       
      //************This implementation receives only level or regions 
	/*
	allLevels = (int)(mxGetScalar(prhs[8])) > 0;

	if (allLevels)
        {
	  allRegions = new vector<RegionLevel>();
	  allRegions->clear();
	  fillRegionLevels(prhs[7],*allRegions);
	}
       	else 
	*/
        {
	  regions = new RegionLevel();
	  fillRegions(numRegs, regionsizes,AllInitRegions, *regions); 
	}
 }
 

void InferenceClass::CreateAlgorithm()
{
  // **************************************************************************
  // create the algorithm
  // **************************************************************************

  algorithm = 0;
 
 

  switch (algo_type) {

    case AT_LOOPY: 

      algorithm = new Loopy(main_mrf,sumOrMax,strategy,maxIter);
      //cout<<algo_type<<" "<<main_mrf->N<<" "<<sumOrMax<<" "<<strategy<<"  "<<maxIter<<endl;
         
      break;

    case AT_GBP:
      if (allLevels) {
	processor = new GBPPreProcessor(allRegions, main_mrf, trw);
      }
      else {
	processor = new GBPPreProcessor(*regions, main_mrf, trw);
        processor->printAllRegions();
	regions->clear();
	delete regions;
	regions = 0;
      }
      
      reg_mrf = processor->getRegionMRF();
      assignInd = processor->getAssignTable();
      bethe = processor->getBethe();

      algorithm = new GBP(reg_mrf,assignInd,bethe,sumOrMax,gbp_alpha,maxIter);
      
      break;

    case AT_GIBBS:

      algorithm = new Gibbs(main_mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_WOLFF:

      algorithm = new Wolff((PottsMRF*)main_mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_SWENDSEN_WANG:

      algorithm = new SwendsenWang((PottsMRF*)main_mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_METROPOLIS:

      algorithm = new Metropolis(main_mrf,startX,burningTime,samplingInterval,num_samples);

      delete[] startX;
      startX = 0;
      
      break;

    case AT_MEAN_FIELD:

      algorithm = new MeanField(main_mrf,maxIter);

      break;
      
    default:

      cout<<"invalid algorithm type. possible values are: 0-loopy, 1-gbp, 2-gibbs, 3-wolff, 4-swendswen-wang, 5-metropolis, 6-mean-field"<<endl;
      break;
  }

}
  

void InferenceClass::MakeInferenceAlgorithm()
{

  // **************************************************************************
  // make inference
  // **************************************************************************
  
  beliefs = algorithm->inference(&converged);

 
  singleBeliefs = 0;
  pairBeliefs = 0;

  switch (algo_type) {
    
    case AT_LOOPY:
      
      	pairBeliefs = ((Loopy*)algorithm)->calcPairBeliefs();
	//cout<<"at init "<<pairBeliefs[0]<<"   "<<pairBeliefs[0][1]<<"   "<<pairBeliefs[0][1][0]<<"   "<<pairBeliefs[0][1][0][0]<<"  "<<endl;
   
      break;
      
    case AT_GBP:
      
      singleBeliefs = new double*[num_nodes];
      for (int i=0; i<num_nodes; i++) {
	singleBeliefs[i] = new double[main_mrf->V[i]];      
      }
      processor->extractSingle(beliefs,singleBeliefs,sumOrMax);

      //if (nlhs > 2) 
       {
	pairBeliefs = new double***[num_nodes];
	for (int i=0; i<num_nodes; i++) {
	  pairBeliefs[i] = new double**[main_mrf->neighbNum(i)];
	  for (int n=0; n<main_mrf->neighbNum(i); n++) {
	    pairBeliefs[i][n] = 0;
	    int j = main_mrf->adjMat[i][n];
	    if (i<j) {
	      pairBeliefs[i][n] = new double*[main_mrf->V[i]];
	      for (int xi=0; xi<main_mrf->V[i]; xi++) {
		pairBeliefs[i][n][xi] = new double[main_mrf->V[j]];
	      }
	    }
	  }
	}
	processor->extractPairs(beliefs,pairBeliefs,sumOrMax);
      }
      beliefs = singleBeliefs;
      
      break;
      
    default:
      
      break;
  }

  //cout<<algorithm<<"------- "<<singleBeliefs<<" "<<processor<<" "<<beliefs<<" "<<main_mrf<<"  "<<adjMat<<endl;
}




void InferenceClass::FindBest(unsigned* bestconf)
{
  int i,xi;  
  double maxbelief;

 
    for (i=0; i<main_mrf->N; i++) 
     {
      maxbelief = 0.0; 
      cout<<"--------  "<<i<<"-------------"<<endl; 
      for (xi=0; xi<main_mrf->V[i]; xi++) 
      {
	cout<<beliefs[i][xi]<<" ";
	if(maxbelief < beliefs[i][xi])
          {
	    maxbelief = beliefs[i][xi];
            bestconf[i] = xi;
          }
      }
      cout<<endl;
     }     
 
 
 }


void InferenceClass::FindMaxUnivariateMarginals(double* maxbelief, unsigned* maxconf)
{
  int i,xi;  
  
  //cout<<" It will enter here "<<endl;
     for (i=0; i<main_mrf->N; i++) 
     {
      maxbelief[i] = 0.0; 
      for (xi=0; xi<main_mrf->V[i]; xi++)
       { 
	 // cout<<beliefs[i][xi]<<" ";
        if(maxbelief[i] < beliefs[i][xi]) 
          {
             maxbelief[i] = beliefs[i][xi];
             maxconf[i] = xi;     
          } 
       } 
      //cout<<endl;
     }     
 
 }





/* OUTPUT RESULTS */

/*
void InferenceClass::CreateAlgorithm()
{
 
  // **************************************************************************
  // assign results to output argument (if given)
  // **************************************************************************

    //UNIVARIATE BELIEFS

      for (int xi=0; xi<main_mrf->V[i]; xi++) 
      {
	resBelPtr[xi] = beliefs[i][xi];
      }

     //BIVARIATE BELIEFS
	
       for (int i=0; i<num_nodes; i++) 
        {
         for (int n=0; n<main_mrf->neighbNum(i); n++) 
         {
	    int j = main_mrf->adjMat[i][n];
	    if (i<j) 
            {
	      for (int xi=0; xi<main_mrf->V[i]; xi++) 
              {
		for (int xj=0; xj<main_mrf->V[j]; xj++)
                {
		  resBelPtr[xi + xj*main_mrf->V[i]] = pairBeliefs[i][n][xi][xj];
		}
	      }	           
	    }
	  }	
         }
     }
*/


      
int InferenceClass::LoopyPropagation(int maxI, SumOrMax SOM , Strategy  stra)
{
  SetLoopy(maxI,SOM,stra);
  CreateAlgorithm();
  MakeInferenceAlgorithm();
  return converged;
}


void InferenceClass::ResolveTies(unsigned* maxAssign, int* cFlag, int MaxIter)
{
  double epsilon;
  double* maxbelief;
  unsigned*  auxvalues;
  int i,xi,vvar;
  int NumberTies;
  InferenceClass* InferenceEngine;
  int* TiedVarsIndices;
  unsigned** TiedValues;
  int posvar,posval; 
  int auxcount;
 
  NumberTies = 0;
  maxbelief =  new double[num_nodes]; // Maximum values univariate marginals
 
  TiedVarsIndices  = new int[num_nodes]; // Vars where maximum marginals are  reached at more than 1 value
  TiedValues = new unsigned*[num_nodes];    // Store configuration where max marginal is reached.
  auxvalues  = new unsigned[num_nodes+1];
  epsilon = exp(-250.0); 
  
  FindMaxUnivariateMarginals(maxbelief,maxAssign);

  //for (i=0; i<main_mrf->N; i++) cout<<i<<" "<<maxbelief[i]<<" "<<maxAssign[i]<<endl; 
  //for (xi=0;xi<main_mrf->V[54];xi++) cout<<xi<<" "<<main_mrf->localMat[54][xi]<<endl; 

  for (i=0; i<main_mrf->N; i++) 
     {
      TiedVarsIndices[i] = 0;
      for (xi=0; xi<main_mrf->V[i]; xi++) 
       {
	if(maxbelief[i] <= beliefs[i][xi] - epsilon)
          {     
	    //cout<<i<<" "<<xi<<" "<<maxbelief[i]<<endl;            
	      if(TiedVarsIndices[i]==0)  auxvalues[0] = 1; //First time the max is found
               else auxvalues[0]++;                    
	       auxvalues[TiedVarsIndices[i]+1] = xi;     //Values are stored in auxvalues
	       TiedVarsIndices[i]++;
           }  
        }  
      if (TiedVarsIndices[i] > 1)   //It has to be at least 1 
       {
         TiedValues[NumberTies] = auxvalues;  
         auxvalues = new unsigned[MaxCardNumber+1];
         NumberTies++;
       }
      else
	TiedValues[NumberTies] = (unsigned*) 0;   
     }
  
  //cout<<"Number of Ties is "<<NumberTies<<endl;
  // for (i=0; i<main_mrf->N; i++)   cout<<"  "<<TiedVarsIndices[i];
  // cout<<endl;
  
  if (NumberTies>0)     
    {
      //for (i=0; i<TiedVarsIndices[0]; i++)   cout<<"  "<<TiedValues[0][i];  
      // cout<<endl;
       posvar = randomint(NumberTies);  
       i = 0;
       auxcount = -1;
      while(i<num_nodes && auxcount<posvar)
	{
          if (TiedVarsIndices[i]>1) auxcount++; //Where is this variable
          i++; 
        }
      
      if(i<=num_nodes)
         { 
	   vvar = i-1;          
	   posval = TiedValues[posvar][randomint(TiedValues[posvar][0]) + 1]; //Which of its variables will be fixed
         }

        else cout<<"There must a mistake here"<<endl;
      //cout<<NumberTies<<"----"<<posvar<<"----- "<<auxcount<<"--------"<<i<<"------"<<vvar<<endl;
     
      
      InferenceEngine = new InferenceClass(this);     
      //for (xi=0;xi<InferenceEngine->main_mrf->V[vvar];xi++) cout<<xi<<" "<<InferenceEngine->main_mrf->localMat[vvar][xi]<<endl; 
      //for (xi=0;xi<main_mrf->V[54];xi++) cout<<xi<<" "<<main_mrf->localMat[54][xi]<<endl; 
      for (xi=0;xi<posval;xi++) InferenceEngine->main_mrf->localMat[vvar][xi] = epsilon;
      for (xi=posval+1;xi<InferenceEngine->main_mrf->V[vvar];xi++) InferenceEngine->main_mrf->localMat[vvar][xi] = epsilon;
      //cout<<" After "<<endl;
      //for (xi=0;xi<InferenceEngine->main_mrf->V[vvar];xi++) cout<<xi<<" "<<InferenceEngine->main_mrf->localMat[vvar][xi]<<endl;
      InferenceEngine->LoopyPropagation(MaxIter,MAX,SEQUENTIAL);
      
      InferenceEngine->ResolveTies(maxAssign,cFlag,MaxIter);   
      delete InferenceEngine;
    }
     
  delete[] maxbelief; 
  delete[] auxvalues; 
  delete[] TiedVarsIndices;
  for(i=0;i<NumberTies;i++) delete[] TiedValues[i];
  delete[] TiedValues;
}

/*
int InferenceClass::MaxConfigurationsLoopy(unsigned** bestconf, double* energies, InferenceClass* InferenceEngine, int numberconf,int MaxIter)
{
  unsigned* maxconf;  
  int  xi,cFlag,j,m,initBN,nAlgRun,Finish,notend,node,state;
  double epsilon, expectedE,nextV,tol,deltaE,energy_m,val;
  VectCandidate* newCandidates;
  CandidateNode* NextCand;
  NodeBestConf** ListEngines;

  newCandidates = new VectCandidate();
 
  epsilon = exp(-250.0);
  tol = exp(-6.0);
  nAlgRun = 0;  m = 0;
  ListEngines = new (NodeBestConf*)[2*numberconf+1];
  ListEngines[0] = new NodeBestConf(num_nodes, 0, InferenceEngine);
  maxconf = new unsigned[num_nodes];
  Finish = 0;


 while( m< numberconf && !Finish)
  {  
   
   cFlag = ListEngines[nAlgRun]->inf_ente->LoopyPropagation(MaxIter,MAX,SEQUENTIAL); //Loopy propagation is done
  
   if(cFlag!=-1) ListEngines[nAlgRun]->inf_ente->ResolveTies(maxconf, &cFlag,MaxIter);
  
   if(cFlag==-1) 
     {
       // cout<<"The algorithm failed to converge "<<endl;
       Finish = 1;
     }
   else  
    {      
     //Updates the value of the best energy of the best maximal config.
     ListEngines[nAlgRun]->SetBestConf(maxconf);
     energy_m = ListEngines[nAlgRun]->inf_ente->GetEnergy(maxconf);   
     //cout<<"The energy "<< energy_m<<endl;   
     ListEngines[nAlgRun]->best_energy =  energy_m;
   
     if (m>0  && abs(long(energy_m - expectedE))>tol ) //Expected energy differs, corect nextCand
     {
       val = energy_m -  ListEngines[0]->best_energy;
       //cout<<"not expected "<<energy_m<<" "<<expectedE<<endl;
       NextCand = new CandidateNode(val,NextCand->node,NextCand->state,NextCand->initBN,nAlgRun);
       newCandidates->push_back(NextCand);    
     }
     else 
     {
      m++;
      //ListEngines[nAlgRun]->PrintBestConf();
      for(j=0;j<num_nodes;j++)  bestconf[m-1][j] = maxconf[j];
      energies[m-1] = energy_m;
     }
     if(m==numberconf || nAlgRun==numberconf) Finish = 1;
    } 
  

 if(!Finish )
     {
      deltaE = energy_m - ListEngines[0]->best_energy;
      AddNewCandidates(newCandidates,ListEngines[nAlgRun]->inf_ente->beliefs,nAlgRun,maxconf,deltaE);   
      if (m>1) 
       {
        newCandidates->begin();
        int ndeliter = 0;
        notend = 1;
        while (notend) 
         {
           notend = (newCandidates->current != newCandidates->last);
	   if (newCandidates->current->content->initBN == initBN &&  newCandidates->current->content->verified==-1)
              {
                ndeliter++;                   
                newCandidates->erase(newCandidates->current);                                
	      }
           else if (notend) newCandidates->current = newCandidates->current->next;
       }  
        
        cFlag = ListEngines[initBN]->inf_ente->LoopyPropagation(MaxIter,MAX,SEQUENTIAL); //Loopy propagation in negatively constrained MRF
	if(cFlag!=-1) 
	  {            
            deltaE = ListEngines[initBN]->best_energy - ListEngines[0]->best_energy;
            AddNewCandidates(newCandidates,ListEngines[initBN]->inf_ente->beliefs,initBN,ListEngines[initBN]->best_conf,deltaE);
          }          	
       }
   
      if (newCandidates->empty()) 
       {
         cout<<"The list of candidates is empty"<<endl;
         Finish = 1;        
       }
     }

   if(! Finish)
    {
      newCandidates->begin();
      newCandidates->sort();
      newCandidates->begin();
      nextV = newCandidates->current->content->v;       
   
      while (newCandidates->current->content->verified>-1)  
         {
           m++;          
           cout<<m<<endl;
           for(j=0;j<num_nodes;j++)  bestconf[m-1][j] = ListEngines[newCandidates->current->content->verified]->best_conf[j];
           energies[m-1] =  ListEngines[newCandidates->current->content->verified]->best_energy;        	      
           newCandidates->erase(newCandidates->current);   
                  
           //newCandidates->begin();
           nextV =  newCandidates->current->content->v;    
          
         }
      //NextCand = ->assign(*cand_iter);
     


      node = newCandidates->current->content->node;
      state = newCandidates->current->content->state;
      initBN =  newCandidates->current->content->initBN;
      newCandidates->erase(newCandidates->current);
      expectedE = energies[0] + nextV; 
       //Lock in the best state
      nAlgRun++;    
      ListEngines[nAlgRun] = new NodeBestConf(num_nodes, 0);
      ListEngines[nAlgRun]->inf_ente = new InferenceClass(ListEngines[initBN]->inf_ente);  
      //cout<<"nAlgRun: "<<nAlgRun<<" node: "<<node<<" initBN: "<<initBN<<" state: "<<state<<" expectedE: "<<expectedE<<" ptr 1-2  "<< ListEngines[nAlgRun]->inf_ente->main_mrf->localMat<<" "<< ListEngines[initBN]->inf_ente->main_mrf->localMat<<endl;
     
      for (xi=0;xi<state;xi++) ListEngines[nAlgRun]->inf_ente->main_mrf->localMat[node][xi] = epsilon;
      for (xi=state+1;xi<ListEngines[nAlgRun]->inf_ente->main_mrf->V[node];xi++) ListEngines[nAlgRun]->inf_ente->main_mrf->localMat[node][xi] = epsilon;
      ListEngines[nAlgRun]->initBN = initBN;
      ListEngines[initBN]->inf_ente->main_mrf->localMat[node][state] = epsilon; 
     } 
  }
 
 delete[] maxconf;
 delete newCandidates;
 for(j=1;j<nAlgRun+1;j++)  delete ListEngines[j]; // The first Inference object is not deleted
 delete[] ListEngines;
 // cout<<"Finished Ok "<<endl;
 return m;
}


*/

int InferenceClass::MaxConfigurationsLoopy(unsigned** bestconf, double* energies, InferenceClass* InferenceEngine, int numberconf,int MaxIter)
{
  int xi,cFlag,j;
  unsigned* maxconf;  
  int m,initBN,nAlgRun,Finish;
  double epsilon;
  double expectedE,nextV,tol;
  int state,node;
  double deltaE,energy_m,val;
  vector<CandidateNode>* newCandidates;
  vector<CandidateNode>::iterator cand_iter;
  CandidateNode* NextCand;
  //  CandidateNode* AuxCand;
  NodeBestConf** ListEngines;

  NextCand = new CandidateNode(0.0,0,0,0,0);
  newCandidates  = new vector<CandidateNode>();
  newCandidates->clear();
  


  epsilon = exp(-250.0);
  tol = exp(-6.0);

  nAlgRun = 0;  m = 0;
  ListEngines = new NodeBestConf* [2*numberconf+1];
  ListEngines[0] = new NodeBestConf(num_nodes, 0, InferenceEngine);

  // EVALUAR AQUI PARA VER SI PSI GUARDA LOS VALORES APROPIADOS, O SI HAN LLEGADO CORRECTAMENTE EN InferenceEngine

  maxconf = new unsigned[num_nodes];
  // The class (this) has been initialized with all the parameters

 Finish = 0;


 while( m< numberconf && !Finish)
  {  
   
   cFlag = ListEngines[nAlgRun]->inf_ente->LoopyPropagation(MaxIter,MAX,SEQUENTIAL); //Loopy propagation is done

     // CAMBIAN LOS VALORES DESPUES DE LA PROPAGACIO, COMO CAMBIAN? 
   //cout<<"Resolve ties 0"<<endl;
   if(cFlag!=-1) ListEngines[nAlgRun]->inf_ente->ResolveTies(maxconf, &cFlag,MaxIter);
   //cout<<"Resolve ties 1"<<endl;
   if(cFlag==-1) 
     {
        cout<<"The algorithm failed to converge "<<endl;
       Finish = 1;
     }
   else  
    {
      
     //Updates the value of the best energy of the best maximal config.
     ListEngines[nAlgRun]->SetBestConf(maxconf);
     energy_m = ListEngines[nAlgRun]->inf_ente->GetEnergy(maxconf);   
     //cout<<"The energy "<< energy_m<<endl;
   
     ListEngines[nAlgRun]->best_energy =  energy_m;
   
     if (m>0  && abs(long(energy_m - expectedE))>tol ) //Expected energy differs, corect nextCand
     {
       val = energy_m -  ListEngines[0]->best_energy;
       //cout<<"not expected "<<energy_m<<" "<<expectedE<<endl;
       newCandidates->push_back(CandidateNode(val,NextCand->node,NextCand->state,NextCand->initBN,nAlgRun));    
     }
     else 
     {
      m++;
      //ListEngines[nAlgRun]->PrintBestConf();
      for(j=0;j<num_nodes;j++)  bestconf[m-1][j] = maxconf[j];
      energies[m-1] = energy_m;
      // Finish = 1; //CAMBIADO
     }
     if(m==numberconf || nAlgRun==numberconf) Finish = 1;
    } 

   

 if(!Finish )
     {
      deltaE = energy_m - ListEngines[0]->best_energy;
      AddNewCandidates(newCandidates,ListEngines[nAlgRun]->inf_ente->beliefs,nAlgRun,maxconf,deltaE);
     
      if (m>1) 
       {
        cand_iter = newCandidates->begin();
        int ndeliter = 0;
        while (cand_iter != newCandidates->end()) 
         {
	   if ( (*cand_iter).initBN == initBN && (*cand_iter).verified==-1)
              {
                ndeliter++;
                //AuxCand  = &*cand_iter;
                //cout<<ndeliter<<" del "<<AuxCand->node<<" "<<AuxCand->state<<endl;    
                //delete AuxCand;                   
                //cout<<ndeliter<<" now "<<cand_iter->node<<" "<<cand_iter->state<<" "<<cand_iter->v<<" "<<cand_iter->initBN<<endl;       
                newCandidates->erase(cand_iter);                                
	      }
           else cand_iter++;   //ELSE HAS BEEN ADDED           
	  
         }

	
	 //cand_iter = newCandidates->begin();
	 //cout<<"To Check"<<initBN<<endl;
	 //while (cand_iter != newCandidates->end()) 
      	 // {
         //   cout<<cand_iter->node<<" "<<cand_iter->state<<" "<<cand_iter->v<<" "<<cand_iter->initBN<<endl;          
	 //  cand_iter++;
         // }
	

        cFlag = ListEngines[initBN]->inf_ente->LoopyPropagation(MaxIter,MAX,SEQUENTIAL); //Loopy propagation in negatively constrained MRF
	
        if(cFlag!=-1) 
	  {            
            deltaE = ListEngines[initBN]->best_energy - ListEngines[0]->best_energy;
            AddNewCandidates(newCandidates,ListEngines[initBN]->inf_ente->beliefs,initBN,ListEngines[initBN]->best_conf,deltaE);
          }   
       	// cout<<"constrained propagation-- 1 "<<endl;      
       }
   
      if (newCandidates->empty()) 
       {
         cout<<"The list of candidates is empty"<<endl;
         Finish = 1;        
       }
     }




   if(! Finish)
    {
      cand_iter = newCandidates->begin();
      //  while (cand_iter != newCandidates->end()) 
      //	  {
      //      cout<<cand_iter->node<<" "<<cand_iter->state<<" "<<cand_iter->v<<" "<<cand_iter->initBN<<endl;          
      //      cand_iter++;
      //    }

      sort(newCandidates->begin(), newCandidates->end());
      cand_iter = newCandidates->begin();
   
    
      // cout<<cand_iter->node<<" "<<cand_iter->state<<endl;    

       nextV = (*cand_iter).v;
   
    
      while ((*cand_iter).verified>-1)  
         {
           m++;
          
           for(j=0;j<num_nodes;j++)  bestconf[m-1][j] = ListEngines[(*cand_iter).verified]->best_conf[j];
           energies[m-1] =  ListEngines[(*cand_iter).verified]->best_energy;
          
	      
           newCandidates->erase(cand_iter);
            
         
           cand_iter = newCandidates->begin();
           nextV = (*cand_iter).v;
          
         }
  
      NextCand->assign(*cand_iter);
      node = (*cand_iter).node;
      state = (*cand_iter).state;
      initBN =  (*cand_iter).initBN;
      newCandidates->erase(cand_iter);
      expectedE = energies[0] + nextV; 
    
      //Lock in the best state

      nAlgRun++;

      
      ListEngines[nAlgRun] = new NodeBestConf(num_nodes, 0);
      ListEngines[nAlgRun]->inf_ente = new InferenceClass(ListEngines[initBN]->inf_ente);  
    
      //   cout<<"nAlgRun: "<<nAlgRun<<" node: "<<node<<" initBN: "<<initBN<<" state: "<<state<<" expectedE: "<<expectedE<<" ptr 1-2  "<< ListEngines[nAlgRun]->inf_ente->main_mrf->localMat<<" "<< ListEngines[initBN]->inf_ente->main_mrf->localMat<<endl;
     
      for (xi=0;xi<state;xi++) ListEngines[nAlgRun]->inf_ente->main_mrf->localMat[node][xi] = epsilon;
   
   for (xi=state+1;xi<ListEngines[nAlgRun]->inf_ente->main_mrf->V[node];xi++) ListEngines[nAlgRun]->inf_ente->main_mrf->localMat[node][xi] = epsilon;
    
      //ListEngines[nAlgRun]->inf_ente->main_mrf->localMat[node][state] =  ListEngines[initBN]->inf_ente->main_mrf->localMat[node][state];
     
      ListEngines[nAlgRun]->initBN = initBN;
      ListEngines[initBN]->inf_ente->main_mrf->localMat[node][state] = epsilon; 
       //for (xi=0;xi<ListEngines[initBN]->inf_ente->main_mrf->V[node];xi++) cout<<xi<<" "<<ListEngines[initBN]->inf_ente->main_mrf->localMat[node][xi]<<endl;
       //for (xi=0;xi<ListEngines[nAlgRun]->inf_ente->main_mrf->V[node];xi++) cout<<xi<<" "<<ListEngines[nAlgRun]->inf_ente->main_mrf->localMat[node][xi]<<endl;
     
     
    } 
  }


 delete[] maxconf;
 
 
 newCandidates->clear();
 delete newCandidates;
 delete NextCand;
 

 for(j=1;j<nAlgRun+1;j++) 
   {

    delete ListEngines[j]; // The first Inference object is not deleted
  
   } 

 delete[] ListEngines;
 // cout<<"Finished Ok "<<endl;
 return m;
}





InferenceClass::~InferenceClass()
{
  // **************************************************************************
  // free memory
  // **************************************************************************


  //cout<<algorithm<<" "<<singleBeliefs<<" "<<processor<<" "<<beliefs<<" "<<main_mrf<<"  "<<endl;
 
 

 if( algorithm != 0)
   {
     delete algorithm;
     algorithm = 0;
   }


  if (singleBeliefs != 0) {
    for (int i=0; i<num_nodes; i++) {
      delete[] singleBeliefs[i];
    }

    delete[] singleBeliefs;    
    singleBeliefs = 0;
  }

 

  /*
   if (pairBeliefs[0] != 0 && algo_type == AT_GBP) 
   cout<<pairBeliefs[0]<<"   "<<pairBeliefs[0][1]<<"   "<<pairBeliefs[0][1][0]<<"   "<<pairBeliefs[0][1][0][0]<<"  "<<endl;
    for (int i=0; i<main_mrf->N; i++) {
      for (int n=0; n<main_mrf->neighbNum(i); n++) {
	if (pairBeliefs[i][n] != 0) {
	  for (int xi=0; xi<main_mrf->V[i]; xi++) {
            delete[] pairBeliefs[i][n][xi];        
	  }
    	  delete[] pairBeliefs[i][n];
	}
      }
      delete[] pairBeliefs[i];
    }
    delete[] pairBeliefs;
    pairBeliefs = 0;
  }
  */

  if (processor != 0) {
    delete processor;
    processor = 0;
  }
 
  /*
if (beliefs != 0) {
    for (int i=0; i<num_nodes; i++) {
      cout<<i<<" "<<beliefs[i][0]<<endl;
       delete[] beliefs[i];
    }
    delete[] beliefs;    
    beliefs = 0;
  }
  */

  delete main_mrf;
  main_mrf = 0;


 /*
 for(int i=0;i<num_nodes;i++)    cout<<i<<" "<< adjMat[i].size()<<endl;
        for (int i=0; i<num_nodes; i++) {
        adjMat[i].clear();
    }
      adjMat.clear();
 */  
   

  //   cout<<algorithm<<" "<<singleBeliefs<<" "<<processor<<" "<<beliefs<<" "<<main_mrf<<endl;
}


double InferenceClass::GetEnergy(unsigned* assign)
  {
  
   double E;
  
   E = main_mrf->getEnergy(assign);
   //cout<< "It is  evaluating energy "<<E<<endl;

   /*
 
   for(j=0;j<num_nodes;j++)
    {
         E = E - log(main_mrf->localMat[j][assign[j]] + epsilon);
       
      for(k=0;k<neighbNum(j);k++)
      {
        k = adjMat[i][n]
	cout<<j<<" "<<k<<" "<<assign[j]<<" "<<assign[k]<<" "<<E<<endl;
       	 E = E - log(main_mrf->pairPotential(j,k,assign[j],assign[k]) +epsilon);    
      }
    }    

   */
   return E;
   
  }

/*
void InferenceClass::AddNewCandidates(VectCandidate* newCandidates, double** ubeliefs, int n, unsigned* res, double deltaE)
{
  double** normbelief;
  unsigned int i,xi;
  double sumbelief,val,epsilon,eps2;
  CandidateNode* auxCandidate;

 
  int ncand;
  ncand = 0;
  eps2 = exp(-10.0); //This is the maximum increment in the energy before stopping the Max Conf (SANTANA)
  epsilon = exp(-200.0);
  normbelief = new double*[main_mrf->N];

  for (i=0; i<main_mrf->N; i++) 
     {         
      normbelief[i] = new double[main_mrf->V[i]];
      sumbelief = 0.0; 
      for (xi=0; xi<main_mrf->V[i]; xi++)  sumbelief += ubeliefs[i][xi];
      for (xi=0; xi<main_mrf->V[i]; xi++) 
       {
         ubeliefs[i][xi] /=  sumbelief;     

         //val = -log(ubeliefs[i][xi]) + log(ubeliefs[i][res[i]]) + deltaE;
         if(xi != res[i]  && ubeliefs[i][xi]>epsilon ) //&& val>eps2
	   {
             val = -log(ubeliefs[i][xi]) + log(ubeliefs[i][res[i]]) + deltaE;
             ncand++;              
             auxCandidate = new CandidateNode(val,i,xi,n,-1);
             newCandidates->push_back(auxCandidate);              
           }
	}
       delete[] normbelief[i];
     } 
   delete[] normbelief;
}

*/

void InferenceClass::AddNewCandidates(vector<CandidateNode>* newCandidates, double** ubeliefs, int n, unsigned* res, double deltaE)
{
  double** normbelief;
  unsigned int i,xi;
  double sumbelief,val,epsilon,eps2;
  CandidateNode auxCandidate;
 
  int ncand;
  ncand = 0;
  eps2 = exp(-10.0); //This is the maximum increment in the energy before stopping the Max Conf (SANTANA)
  epsilon = exp(-200.0);
  normbelief = new double*[main_mrf->N];

  for (i=0; i<main_mrf->N; i++) 
     {         
      normbelief[i] = new double[main_mrf->V[i]];
      sumbelief = 0.0; 
      for (xi=0; xi<main_mrf->V[i]; xi++)  sumbelief += ubeliefs[i][xi];
      for (xi=0; xi<main_mrf->V[i]; xi++) 
       {
         ubeliefs[i][xi] /=  sumbelief;     

         //val = -log(ubeliefs[i][xi]) + log(ubeliefs[i][res[i]]) + deltaE;
         if(xi != res[i]  && ubeliefs[i][xi]>epsilon ) //&& val>eps2
	   {
             val = -log(ubeliefs[i][xi]) + log(ubeliefs[i][res[i]]) + deltaE;
             ncand++;
            
	     //  cout<<ncand<<" "<<n<<" "<<i<<" "<<xi<<" "<<ubeliefs[i][xi]<<" "<<val<<endl;
	     // cout<<endl;
             auxCandidate.setvalues(val,i,xi,n,-1);
             newCandidates->push_back(auxCandidate); 
             //newCandidates->push_back(CandidateNode(val,i,xi,n,-1)); 
           }
	}
      //cout<<ncand<<" "<<i<<" "<<endl;
      delete[] normbelief[i];
     } 
   delete[] normbelief;

   //cout<<"Number of candidates "<<ncand<<endl;
}
 




//**************************************************

NodeBestConf::NodeBestConf(int nnodes, int marca, double b_energy, unsigned* best_c, InferenceClass* InfEnte)
{
  int i;
  number_nodes = nnodes;
  initBN = marca;
  best_energy = b_energy;
  best_conf = new unsigned[number_nodes]; 
  for (i=0;i<number_nodes;i++) best_conf[i] = best_c[i];
  inf_ente = InfEnte;
}


NodeBestConf::NodeBestConf(int nnodes, int marca, InferenceClass* InfEnte)  
{
  number_nodes = nnodes;
  best_conf = new unsigned[number_nodes]; 
  initBN = marca;
  inf_ente = InfEnte;
}

NodeBestConf::NodeBestConf(int nnodes, int marca)  
{
  number_nodes = nnodes;
  best_conf = new unsigned[number_nodes]; 
  initBN = marca;
  inf_ente = (InferenceClass*)0;
}

void NodeBestConf::SetBestConf(unsigned* bconf)  
{
  int i;
 
  for(i=0;i<number_nodes;i++) 
   {
   
     best_conf[i] = bconf[i];
   }

}


void NodeBestConf::PrintBestConf()  
{
  int i;
  for(i=0;i<number_nodes;i++) cout<<best_conf[i]<<" ";
  cout<<endl;
}


NodeBestConf::~NodeBestConf()
{
 
  delete[] best_conf;
  delete inf_ente;  

}

//*****************************************************************

void CandidateNode::setvalues(double val,int i, int xi, int n, int ver)
{
  v = val;
  node = i;
  state = xi;
  initBN = n;
  verified = ver;
}


CandidateNode::CandidateNode()
{

}

CandidateNode::CandidateNode(double valv, int nod, int sta, int initB, int ver)
{
  v = valv;
  node = nod;
  state = sta;
  initBN = initB;
  verified = ver;
}


void CandidateNode::assign(CandidateNode CandN)
{
  v = CandN.v;
  node = CandN.node;
  state = CandN.state;
  initBN = CandN.initBN;
  verified = CandN.verified;
}

 bool operator<(const CandidateNode& x, const CandidateNode& y) { return(x.v < y.v); }
 bool operator==(const CandidateNode& x, const  CandidateNode& y) { return(x.v == y.v); }



/*


  MRF(unsigned int** Mat, int num_nodes):adjMat(vector< Nodes,allocator<Nodes> > )
 {
   int i,j,k,neighNum_i;
   //vector<Nodes,allocator<Nodes> > hadjMat;
    
   //adjMat = *(new  vector<Nodes,allocator<Nodes> >);
  adjMat.resize(num_nodes);
  
  for (i=0; i<num_nodes; i++) 
    {
      neighNum_i = 0;
      for (j=0; j<num_nodes; j++) if(i != j) neighNum_i += Mat[i][j];
      adjMat[i].resize(neighNum_i);
      k = 0;
      for (j=0; j<num_nodes; j++) 
         if (i != j && Mat[i][j]==1)
          {
	   adjMat[i][k] = j;
           k++;
          }
     }
    
    N = adjMat.size();
    V = new int[N];
      for (int i=0; i<N; i++) {
      V[i] = 0;
    }
    lambdaMat = 0;
    localMat = 0;

    T = 1.0;
   
 }

*/
