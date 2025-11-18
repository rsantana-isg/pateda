// BayesianNetwork.cpp: implementation of the CBayesianNetwork class.
//

#include "EDA.h"
#include "BayesianNetwork.h"
#include "UndirectedGraph.h"
#include <math.h>
#include <limits.h>
#include <stdlib.h>
#include <stdio.h>
#include "Chi.h"

#ifdef MPI
  #include "EDAMPI.h"
#endif

#define MAXPARENTSBN 21 
extern int generaciones; 

CBayesianNetwork::CBayesianNetwork()
{
    int i,j;

    // An arcless Bayesian network is created. All nodes have
    // no parents.
    m_parents = new bool*[IND_SIZE];
    for(i=0;i<IND_SIZE;i++)
    {
        m_parents[i] = new bool[IND_SIZE];
        for(j=0;j<IND_SIZE;j++) m_parents[i][j] = false;
    }

    // Thus, there are no paths among nodes.
    m_paths = new int*[IND_SIZE];
    for(i=0;i<IND_SIZE;i++)
    {
        m_paths[i] = new int[IND_SIZE];

        for(j=0;j<IND_SIZE;j++)
            if(i==j) m_paths[i][j] = 1;
            else m_paths[i][j] = 0;
    }

    // The probability distribution will be uniform.
    m_probs = new double*[IND_SIZE];
    for(i=0;i<IND_SIZE;i++)
    {
        m_probs[i] = new double[STATES[i]-1];
        // The final probability is found subtracting
        // all the others to 1.

        for(j=0;j<STATES[i]-1;j++) 
            m_probs[i][j] = 1/(double)STATES[i];
    }

    // The order is trivial.
    m_order = new int[IND_SIZE];
    for(i=0;i<IND_SIZE;i++) m_order[i] = i;

    // The ordered nodes can be found from m_order.
    m_ordered = new int[IND_SIZE];
    for(i=0;i<IND_SIZE;i++) m_ordered[m_order[i]] = i;

    // memory for m_A is allocated.
    m_A = new long double*[IND_SIZE];
    for(i=0;i<IND_SIZE;i++)
    {
        m_A[i] = new long double[IND_SIZE];
        for(j=0;j<IND_SIZE;j++) m_A[i][j] = INT_MIN;
    }

    // Matrix to record the actual state of the Bayesian network.
    m_ActualMetric = new double[IND_SIZE];

}

CBayesianNetwork::~CBayesianNetwork()
{
    int i;

    for(i=0;i<IND_SIZE;i++) delete [] m_probs[i];
    delete [] m_probs;

    delete [] m_order;
    delete [] m_ordered;

    for(i=0;i<IND_SIZE;i++) delete [] m_parents[i];
    delete [] m_parents;

    for(i=0;i<IND_SIZE;i++) delete [] m_A[i];
    delete [] m_A;

    for(i=0;i<IND_SIZE;i++) delete [] m_paths[i];
    delete [] m_paths;

    delete [] m_ActualMetric;
}

CIndividual * CBayesianNetwork::Simulate()
{
    int i, j, v, variable;
    CIndividual * individual = new CIndividual;
    int * genes = individual->Genes();

    //Variables to modify the simulation step
    int *Indiv = new int[STATES[0]]; //we assume that, when using PLS_ALL_VALUES_x,
                     //the number of states of all 
                     //variables is always the same
    for (i=0; i<STATES[0]; i++)
        Indiv[i]=0;
    int variables_left = IND_SIZE;
    int values_left = STATES[0];
    double P_indiv, K;
    double sum;

    //table to store all the probabilities and access them more easily
    int max_states=0; //variable to compute which is the maximum number of states of any of the variables
        for(i=0; i<IND_SIZE; i++) if(max_states<STATES[i]) max_states=STATES[i];
    double *Prob_table = new double [max_states];

    // The individual will be generated simulating its
    // genes according to the order they have in the
    // Bayesian network.
    for(variable=0;variable<IND_SIZE;variable++)
    {
        double which = double(double(rand())/RAND_MAX);

        //calculate the probabilities and store them in the probability table
        double * probs = Probabilities(variable,genes);

        sum=0.0;
        for (i=0; i<STATES[variable]-1; i++) {
            Prob_table[i] = probs[i];
            sum += probs[i];
        }
        Prob_table[STATES[variable]-1]=1-sum; //prob. of last variable = 1 - sum of the rest

        //Modify the probabilities to get correct individuals
        switch (SIMULATION) {
        
        case PLS:               
            break; //PLS simple, no modifications

        case PLS_CORRECT:               
            break; //PLS simple, modifications later

        case PLS_ALL_VALUES_1:
            if (variables_left==values_left) {
                P_indiv=0.0;
                for (i=0; i<STATES[variable]; i++) 
                    if (Indiv[i]>0) 
                        P_indiv += Prob_table[i];
                for (i=0; i<STATES[variable]; i++) {
                    if (Indiv[i]>0) 
                        Prob_table[i] = 0.0;
                    else 
                        Prob_table[i] *= 1/(1-P_indiv);
                }
            }
            break;
        
        case PLS_ALL_VALUES_2:  
            if (values_left>0) {

                //calculate P_indiv
                P_indiv=0.0;
                for (i=0; i<STATES[variable]; i++) 
                    if (Indiv[i]>0) 
                        P_indiv += Prob_table[i];

                //Change the probabilities of all values
                if (variables_left==values_left) {
                    for (i=0; i<STATES[variable]; i++) {
                        if (Indiv[i]>0) 
                            Prob_table[i]=0.0;
                        else 
                            Prob_table[i] *= 1/(1-P_indiv);
                    }
                } else {
                    //calculate K, the constant to divide the probabilities
                    K=int((IND_SIZE - variables_left)/(variables_left - values_left))+1;
                    //K = round(K);

                    for (i=0; i<STATES[variable]; i++) {
                        if (Indiv[i]>0) 
                            Prob_table[i] /= K;
                        else 
                            Prob_table[i] *= (K - P_indiv)/(K*(1-P_indiv));
                    }
                }
            }
            break;
        }
        
        int j;
        for(j=0;j<(STATES[m_ordered[variable]]-1) && which>Prob_table[j];j++)
            which -= Prob_table[j];

        //Correct the problem of the rounding in the computer
        if (Prob_table[j]==0.0) {
            if (j==0) //we will select the next variable that has a prob!=0
                do {j++;} while (Prob_table[j]==0.0);
            else //we will select the previous variable that has a prob!=0
                do {j--;} while (Prob_table[j]==0.0);
        }

        genes[m_ordered[variable]] = j;
        variables_left--;
        if (Indiv[j]==0) values_left--;
        Indiv[j]++;

    }

    //Correct the individual if necessary and requested
    if ((SIMULATION==PLS_CORRECT)&&(values_left>0)) {
        while (values_left>0) {
            //select a random position that contains a variable already appeared at least twice
            do {j=rand()%IND_SIZE;} while (Indiv[genes[j]]<2); 
            
            //substitute this variable by next the missing value.
            for (v=0; v<STATES[variable]; v++) {if (Indiv[v]==0) break;}
            Indiv[genes[j]]--;
            genes[j]=v;
            Indiv[v]++;
            values_left--;
        }
    }

    //Code neccessary only for Endika's program. Please, keep it commented otherwise
    //it keeps record of the correctness of the individual generated.
    switch(values_left) {
    case 0: POPCORRECT++; break;
    case 1: POPMISS1++; break;
    case 2: POPMISS2++; break;
    case 3: POPMISS3++; break;
    default: POPMISSMORE++;
    }

    delete [] Indiv;
    delete [] Prob_table;

    individual->m_values_left = values_left;
    return individual;
}



void CBayesianNetwork::Learn(int **&cases,double **&values,double sel_total)
{
    // Learn the structure
    switch(LEARNING)
    {
    case UMDA: 
        // No learning required.
        break;
    case EBNA_B:
      //SCORE = BIC_SCORE;
        LearnEBNAB(cases);
        break;
    case EBNA_LOCAL:
      //SCORE = BIC_SCORE;
        LearnEBNALocal(cases);
        break;
    case EBNA_PC:
        LearnPC(cases);
        break;
    case EBNA_K2:
        SCORE = K2_SCORE;
        LearnEBNALocal(cases);
        break;
    case PBIL:
        // No learning required.
        break;
    case BSC:
        // No learning required.
        break;
    case TREE:
        LearnTree(cases);
        break;
    case MIMIC:
        LearnMIMIC(cases);
        break;
    }

    
    // Learn the probabilities
    LearnProbabilities(cases,values,sel_total);
}

double * CBayesianNetwork::Probabilities(int ordered_node, 
                                         int * instance)
{
    // First the configuration of the parent nodes is
    // found.
    int parent_configuration = 0;
    int i;
    int nparents=0;

    for(i=0;i<ordered_node;i++)
        if(m_parents[m_ordered[ordered_node]][m_ordered[i]]) 
        {
	  nparents++;
	  if (nparents > MAXPARENTSBN) {
	    //	    cerr << "variable " << i << " has more than " << MAXPARENTSBN << " parents! Truncating to " << MAXPARENTSBN << "." << endl;
	    break;
	  } else {
            parent_configuration *= STATES[m_ordered[i]];
            parent_configuration += instance[m_ordered[i]];
	  }
        }

    // Then the conditional probability corresponding to that
    // parent configuration is returned.
    return &(m_probs[m_ordered[ordered_node]]
                    [parent_configuration*(STATES[m_ordered[ordered_node]]-1)]);
}


void CBayesianNetwork::LearnProbabilities(int **&cases,double **&values,double sel_total)
{
    long i,j,k;
    int ** nijk=NULL;
    int nparents;

    for(i=0;i<IND_SIZE;i++)
    {
        // Calculate the number of parent configurations of i.
        long no_j = 1;
        nparents = 0;

        for(j=0;j<IND_SIZE;j++) {
	  if(m_parents[i][j]) {
              nparents++;
              if (nparents > MAXPARENTSBN) {
		cerr << "variable " << i << " has more than " << MAXPARENTSBN << " parents! Truncating to " << MAXPARENTSBN << "." << endl;
                break;
	      } else {
		no_j *= STATES[j];
	      } 
	  }
	  //if (i==8) 
	  //  {cout << "no_j: " << no_j << " j:" << j << " i: " << i << endl;}
        }
                
        // Allocate memory for all nijk-s.
        nijk=NULL;
        nijk = new (int (* [no_j]));
        //cout << endl << "kk1 " << nijk;// << nijk[0];
        /*nijk = (int **) realloc (nijk, no_j*sizeof(int *));*/
        //cout << " kk2 " << nijk << " " << nijk[0] << endl;
        for(j=0;j<no_j;j++) 
        {
            nijk[j] = new int[STATES[i]];
            for(k=0;k<STATES[i];k++) nijk[j][k] = 0;
        }
        /*for(j=0;j<no_j;j++) 
        {
            nijk[j] = (int *)calloc(STATES[i], sizeof(int));
            for(k=0;k<STATES[i];k++) nijk[j][k] = 0;
        }
        cout << " kk3 " << nijk << " " << nijk[0] << endl;*/

        // All nijk-s are calculated.
        for(j=0;j<SEL_SIZE;j++)
        {
            long parent_configuration = 0;
            nparents = 0;
            for(int parent=0;parent<IND_SIZE;parent++) {
                if(m_parents[i][parent])
                {
		  nparents++;
		  if (nparents > MAXPARENTSBN ) {
		    //		    cerr << "variable " << i << " has more than " << MAXPARENTSBN << " parents! Truncating to " << MAXPARENTSBN << "." << endl;
		    break;
		  } else {
                    parent_configuration *= STATES[parent];
                    parent_configuration += cases[j][parent];
		  }
		}
	    }
	    //            if (i==8) 
	    //	      cout << "parent_configuration: " << parent_configuration << " j:" << j << " i: " << i << endl;
	      
            nijk[parent_configuration][cases[j][i]]++;
        }

        // All probabilities are recalculated (as Buntine).    
        if (LEARNING != PBIL)
        { 
            delete [] m_probs[i];
            m_probs[i] = new double[no_j*(STATES[i]-1)];
        }

        // PBIL needs the probabilities of the prior population.
        // Other learning algorithms do not need them.
        for(j=0;j<no_j;j++)
        {
            int nij = 0;
            for(k=0;k<STATES[i];k++) nij += nijk[j][k];

            double BDeu = (double)1/STATES[i]/no_j;

            // Also when the BN is induced by EDA, conditional
            // probabilities are calculated in univariated=UMDA form
            if ((LEARNING != PBIL) && (LEARNING != BSC))
            {
                for(k=0;k<STATES[i]-1;k++)
                    m_probs[i][j*(STATES[i]-1)+k] =
                        (double)(nijk[j][k]+BDeu)/
                        (double)(nij+STATES[i]*BDeu);
            }

            if (LEARNING == PBIL)
            {
                for(k=0;k<STATES[i]-1;k++)
                    m_probs[i][j*(STATES[i]-1)+k] = 
                        (1.0 - ALPHA_PBIL) * (m_probs[i][j*(STATES[i]-1)+k]) + 
                        ALPHA_PBIL * ((double)(nijk[j][k])/(double)(nij));
            }

            if (LEARNING == BSC)
            {
                for(k=0;k<STATES[i]-1;k++)
                    m_probs[i][j*(STATES[i]-1)+k] = values[i][k]/sel_total;     
            }
            
        }

        // Memory used for nijk-s is released.

        //cout << "ya!";
        for(j=0;j<no_j;j++) delete [] nijk[j];
        delete [] nijk;
        /*for(j=1;j<no_j;j++) 
            free(nijk[j]);
        free(nijk)*/;
    }
}

/******************************************************
 ******************************************************
                        EBNA
 ******************************************************
 ******************************************************/


void CBayesianNetwork::LearnEBNAB(int **&cases)
{
    InitBayesianNetwork();

    LearnEBNALocal(cases);
}

void CBayesianNetwork::LearnEBNALocal(int **&cases)
{
int TimesArcsChanged=0;

 #ifdef VERBOSE
    cout << "Starting BIC computation... " << endl;
 #endif
    CalculateA(cases);
 #ifdef VERBOSE
    cout << "End of BIC computation. " << endl;
 #endif

    double max;

    do
    {
    TimesArcsChanged++; //To count the number of arcs removed/added

        // Find the arc which modification most increases 
        // the BIC metric of the Bayesian network.

        int max_i = 0;
        int max_j = 0;
        max = INT_MIN;

        for(int i=0;i<IND_SIZE;i++)
            for(int j=0;j<IND_SIZE;j++)
                if(m_paths[j][i]==0 && m_A[i][j]>max)
                {
                    // In order to modify the arc j->i 
                    // there cannot be a path from i to
                    // j.

                    max_i = i;
                    max_j = j;
                    max = m_A[i][j];
                }

        // If the BIC metric of the Bayesian network can
        // be improved the arc is modified.

        if(max>0)
        {
            if(!m_parents[max_i][max_j])
            {
                // If there is no arc j->i, it is added.

                //ARC_LOG(max_i,max_j,true);

                Add(max_i,max_j);
            }
            else
            {
                // If there is an arc j->i, it is removed.

                //ARC_LOG(max_i,max_j,false);

                Remove(max_i,max_j);
            }


 #ifdef VERBOSE
	    cout << "Arc: " << max_i << "->" << max_j << " old Metric:" << m_ActualMetric[max_i] << " new:" << m_ActualMetric[max_i] + m_A[max_i][max_j] << endl;
 #endif
            //Actualize the Actual BIC or K2 metric
            m_ActualMetric[max_i] += m_A[max_i][max_j];
            
            CalculateANode(max_i,cases);
        }

 #ifdef VERBOSE
	cout << "looking for best arc - max: " << max << endl;
 #endif

    }
    while(max>0);

#ifdef VERBOSE
    //Show m_A
    for (int ii=0; ii<IND_SIZE; ii++) {
      for (int jj=0; jj<IND_SIZE; jj++)
    cout << m_A[ii][jj] << " ";
      cout << endl;
    }
    cout << endl;
#endif

#ifdef VERBOSE
    cout << endl << "end of looking for best arc " << endl;
#endif

    //CAMBIO HECHO POR ROBERTO  // cout << "Iterations looking for best arc: " << TimesArcsChanged << endl;

}



/******************************************************
 ******************************************************
                         TREE
 ******************************************************
 ******************************************************/

void CBayesianNetwork::LearnTree(int **&cases)
{int i,j,l,r;

    int withnocycles = 0;
    int max_i = 0;
    int max_j = 0;
    double max_mi = 0.0;
    bool no_info = false; // to check if the arc with the best MI has 
                  // MI = 0.0--> in that case, no continue 
                  // learning the tree structure.   

    /* a up triangular matrix which contains the MI (X,Y) values*/
    /* of the variables of the program*/
    double **mi = new double*[IND_SIZE];
    for (j=0;j<IND_SIZE;j++)
    {
        mi[j] = new double[IND_SIZE];
        for (int k=0;k<IND_SIZE;k++)
            mi[j][k] = (double)0.0;
    }

    /* a boolean double matrix containing if there is an*/
    /* undirected path between two nodes */
    /* between i and i there exists an undirected path*/
    bool ** m_upaths = new bool*[IND_SIZE];
    for (j=0;j<IND_SIZE;j++)
    {
        m_upaths[j] = new bool[IND_SIZE];
        for (l=0;l<IND_SIZE;l++)
            m_upaths[j][l] = false;     
    }

    for (l=0;l<IND_SIZE;l++)
        m_upaths[l][l] = true;

    for (i=0;i<IND_SIZE;i++)
    {   
        for (j=i;j<IND_SIZE;j++)
        {
          if (j>i)
			mi[i][j] = ComputeMI(i,j,cases);
        }
    }

	/* a boolean vector to know whether the variable has a parent or not
	bool *m_hasparent = new bool[IND_SIZE];
	for (l=0;l<IND_SIZE;l++)
		m_hasparent[l] = false;     
    */
	
	InitBayesianNetwork();
    
    /*search for the biggest MI and go inducing the BN=Tree of*/
    /*IND_SIZE-1 arcs*/
    while((withnocycles<(IND_SIZE-1)) && (no_info == false))
    {
        max_mi = (double)0.0;
	        
        for (i=0;i<IND_SIZE;i++)
        {
            for (j=i+1;j<IND_SIZE;j++) //no undirected path between i and j
            {
                if (j>i && m_upaths[j][i]==false && m_upaths[i][j]==false && mi[i][j]>=max_mi)
                {
                    max_i = i;
                    max_j = j;
                    max_mi = mi[i][j];
                }
            }
        }

        /* The best arc has MI = 0 and the tree learn process must be stopped
        if (max_mi == (double)0.0)
            no_info = true;         
        */
        mi[max_i][max_j] = (double)(0.0);

        
        // check if the MI of the 'best' arc is not 0.0;-->in this case, no continue learning the tree.
        // check if the edge can be added not to create cycles:
        // no directed paths from i to j, j is no parent of i;
        // no already arc j-->i.
        if ((no_info==false) && (!m_parents[max_i][max_j]) && (m_paths[max_j][max_i]==0))
        {
            withnocycles++;

            // From j to i: j <--> i 
            m_parents[max_i][max_j] = true;
			m_parents[max_j][max_i] = true;
			//m_hasparent[max_i]=true;
            //cout << "Added arete: " << max_j << " <-> " << max_i << endl;

            // The undirected paths of the Bayesian 
            // network are updated. The nodes that can be accesed 
            // from i, they can be now be accesed from j, and viceversa. 
            m_upaths[max_i][max_j]=true;
            m_upaths[max_j][max_i]=true;

            for (l=0;l<IND_SIZE;l++)
            {
                if ((m_upaths[l][max_i]==true) || (m_upaths[max_i][l]==true))
                {
                    m_upaths[l][max_j]=true;
                    m_upaths[max_j][l]=true;
		    // Update the undirected paths of inderectly implicated nodes.
                    for (r=0;r<IND_SIZE;r++)
                    {
                        if ((m_upaths[r][max_i]==true) || (m_upaths[max_i][r]==true))
                        {   
                            m_upaths[l][r]=true;
                            m_upaths[r][l]=true;
                        }
                    }
                }
                if ((m_upaths[l][max_j]==true) || (m_upaths[max_j][l]==true))
                {
                    m_upaths[l][max_i]=true;
                    m_upaths[max_i][l]=true;
		    // Update the undirected paths of inderectly implicated nodes.
                    for (r=0;r<IND_SIZE;r++)
                    {
                        if ((m_upaths[r][max_j]==true) || (m_upaths[max_j][r]==true))
                        {   
                            m_upaths[l][r]=true;
                            m_upaths[r][l]=true;
                        }
                    }
                }
            }
            
            /* The paths of the Bayesian network are updated.
            for(i=0;i<IND_SIZE;i++)
                for(j=0;j<IND_SIZE;j++)
                   if((m_paths[max_j][j]>0) && (m_paths[i][max_i]>0))
                        m_paths[i][j]+=m_paths[max_j][j]*m_paths[i][max_i];
	    
            // Update the ordering of the nodes.
            if(m_order[max_i]<m_order[max_j])
            {
                // The order of max_i, its descendants, max_j
                // and its ancestors must be updated.
                int jump_j = 0; // How many positions the ancestors of max_j are moved.

                // Calculate how many positions the
                // descendants of max_i must be moved.
                int jump_i = 0;
		
                for(k=m_order[max_i];k<=m_order[max_j];k++)
                    if(m_paths[max_j][m_ordered[k]]>0) jump_i++;

                // Update the order of the nodes between
                // max_i and max_j (both included).
                for(k=m_order[max_i];k<=m_order[max_j];k++) 
                    if(m_paths[max_j][m_ordered[k]]>0)
                    {
                        m_order[m_ordered[k]] += jump_j;
                        jump_i--;
                    }
                    else 
                    {
                        m_order[m_ordered[k]] += jump_i;
                        jump_j--;
                    }

                // Update the ordered nodes.
                for(k=0;k<IND_SIZE;k++) m_ordered[m_order[k]] = k;
	      
			}*/
        } /*del if de m_paths, m_parents*/
    }

	//Poner bien el m_parents para crear la estructura de arbol
	//Siempre elegimos el Xi con el i más pequeño como root.
	//Dar dirección a los arcos
	int *m_lastdirected = new int[IND_SIZE];
	for (i=0; i<IND_SIZE; i++) m_lastdirected[i] = false;

	int root = 0; //we fix always the root to be X0
	for (i=0; i<IND_SIZE; i++) {
		if (i==root) continue;
		if (m_parents[i][root]==true) {
			m_parents[root][i]=false;
			m_lastdirected[i]=true;
			Add(i,root);
			withnocycles--;
		}
	}

	int num_directed;
	do {
		num_directed = 0;
		for (int parent=0; parent<IND_SIZE; parent++) {
			if (m_lastdirected[parent]==false) continue;
			for (i=0; i<IND_SIZE; i++) {
				if (i==parent) continue;
				if ((m_parents[i][parent]==true)&&(m_parents[parent][i]==true)) {
					m_parents[parent][i]=false;
					m_lastdirected[i]=true;
					num_directed++;
					Add(i,parent);
					withnocycles--;
				}
			}
		}
	} while (num_directed>0 && withnocycles>0);

	if (withnocycles>0) cerr << "The structure is a Forest!!!!!" << endl;

    /*Show mi
	cout << "max_i:" << max_i << " max_j:" << max_j << endl;
	/    for (int ii=0; ii<IND_SIZE; ii++) {
      for (int jj=0; jj<IND_SIZE; jj++)
    cout << mi[ii][jj] << " ";
      cout << endl;
    }
    cout << endl;*/
       
    /*Show m_order
cout << "m_order:";
    for (int ii=0; ii<IND_SIZE; ii++) {
      cout << m_order[ii] << " ";
    }
    cout << endl;
    //Show m_ordered
cout << "m_ordered:";
    for (ii=0; ii<IND_SIZE; ii++) {
      cout << m_ordered[ii] << " ";
    }
    cout << endl;
    //Show m_upaths
cout << "m_paths:";
    for (ii=0; ii<IND_SIZE; ii++) {
      for (int jj=0; jj<IND_SIZE; jj++)
    cout << m_paths[ii][jj] << " ";
      cout << endl;
    }
    cout << endl;
    //Show m_parents
cout << "m_parents:";
    for (int ii=0; ii<IND_SIZE; ii++) {
      for (int jj=0; jj<IND_SIZE; jj++)
    cout << m_parents[ii][jj] << " ";
      cout << endl;
    }
    cout << endl;*/

    //free memory
    for (i=0;i<IND_SIZE;i++) 
        delete [] mi[i];
    delete [] mi;
    for (i=0;i<IND_SIZE;i++) 
        delete [] m_upaths[i];
    delete [] m_upaths; 
    delete [] m_lastdirected; 
    //delete [] m_hasparent;
}

double CBayesianNetwork::ComputeMI(int bat,int bi,int **&cases)
{
    double pbat = (double)0.0;
    double pbi = (double)0.0;
    double pbatbi = (double)0.0;
    double mi = (double)0.0;

    for (int i=0;i<STATES[i];i++) /*para bat*/
    {
        for (int j=0;j<STATES[j];j++)  /*para bi*/
        {
            pbat = (double)0.0; pbi = (double)0.0; pbatbi = (double)0.0;
            
            for (int r=0;r<SEL_SIZE;r++)
            {
                if (cases[r][bat]==i)
                    pbat = pbat + (double)1.0;
                if (cases[r][bi]==j)
                    pbi = pbi + (double)1.0;
                if ((cases[r][bat]==i) && (cases[r][bi]==j))
                    pbatbi = pbatbi + (double)1.0;
            }
            
            pbat = pbat/(double)(SEL_SIZE);
            pbi = pbi/(double)(SEL_SIZE);
            pbatbi = pbatbi/(double)(SEL_SIZE); 
            
            if (pbatbi!=(double)0.0 && pbat!=(double)0.0 && pbi!=(double)0.0)
                mi += pbatbi * (log((pbatbi)/(pbat*pbi)));
            
        }
    }

    return(mi);
}



/******************************************************
 ******************************************************
                        MIMIC
 ******************************************************
 ******************************************************/


void CBayesianNetwork::Add(int node, int parent)
{int i,j,k;
    // Add the arc.
    m_parents[node][parent] = true;

    // The paths of the Bayesian network
    // are updated.

    for(i=0;i<IND_SIZE;i++)
        for(j=0;j<IND_SIZE;j++)
            if(m_paths[parent][j]>0 && m_paths[i][node]>0)
                m_paths[i][j]+=m_paths[parent][j]*m_paths[i][node];

    // Update the ordering of the nodes.

    if(m_order[node]<m_order[parent])
    {
        // The order of node, its descendants, parent
        // and its ancestors must be updated.

        int jump_parent = 0;// How many positions the
                            // ancestors of parent are moved.

        // Calculate how many positions the
        // descendants of node must be moved.
        int jump_node = 0;
        for(k=m_order[node];k<=m_order[parent];k++)
            if(m_paths[parent][m_ordered[k]]>0) jump_node++;

        // Update the order of the nodes between
        // node and parent (both included).

        for(k=m_order[node];k<=m_order[parent];k++) 
            if(m_paths[parent][m_ordered[k]]>0)
            {
                m_order[m_ordered[k]] += jump_parent;
                jump_node--;
            }
            else 
            {
                m_order[m_ordered[k]] += jump_node;
                jump_parent--;
            }

        // Update the ordered nodes.
        for(k=0;k<IND_SIZE;k++) m_ordered[m_order[k]] = k;
    }
}

double CBayesianNetwork::Entropy(int node,int **&cases)
{int x,i;
    int * n_x = new int[STATES[node]];

    for(x=0;x<STATES[node];x++) n_x[x] = 0;

    for(i=0;i<SEL_SIZE;i++)
        n_x[cases[i][node]]++;

    double entropy = 0;
    for(x=0;x<STATES[node];x++)
        if(n_x[x]>0) entropy += n_x[x]*log((double)n_x[x]);

    entropy /= -SEL_SIZE;
    entropy += log((double)SEL_SIZE);

    delete [] n_x;

    return entropy;
}


double CBayesianNetwork::Entropy(int node, int parent, int **&cases)
{int x,i,y;
    int ** n_xy = new int*[STATES[node]];
    for(x=0;x<STATES[node];x++) 
        n_xy[x] = new int[STATES[parent]];
    
    for(x=0;x<STATES[node];x++)
        for(int y=0;y<STATES[parent];y++)
            n_xy[x][y] = 0;

    for(i=0;i<SEL_SIZE;i++)
        n_xy[cases[i][node]][cases[i][parent]]++;

    double entropy = 0;
    for(x=0;x<STATES[node];x++)
    {
        double entropy_y = 0;
        int n_x = 0;

        for(y=0;y<STATES[parent];y++)
            if(n_xy[x][y]>0)
            {
                entropy_y += n_xy[x][y]*log((double)n_xy[x][y]);
                n_x += n_xy[x][y];
            }

        entropy -= entropy_y;
        if(n_x>0) entropy += n_x*log((double)n_x);
    }

    entropy /= -SEL_SIZE;
    entropy -= log(double(SEL_SIZE));

    for(x=0;x<STATES[node];x++) delete [] n_xy[x];
    delete [] n_xy;

    return entropy;
}

void CBayesianNetwork::LearnMIMIC(int **&cases)
{int i;
    InitBayesianNetwork();

    bool * linked = new bool[IND_SIZE];
    for(i=0;i<IND_SIZE;i++) linked[i] = false;

    int min_i = -1;
    double min_entropy_i = INT_MAX;
    for(i=0;i<IND_SIZE;i++)
    {
        double entropy_i = Entropy(i,cases);

        if(entropy_i<min_entropy_i)
        {
            min_i = i;
            min_entropy_i = entropy_i;
        }
    }

    linked[min_i] = true;
    int head = min_i;

    for(i=IND_SIZE-2;i>=0;i--)
    {
        int min_j = -1;
        double min_entropy_j = INT_MAX;
        for(int j=0;j<IND_SIZE;j++)
            if(!linked[j])
            {
                double entropy_j = Entropy(j,head,cases);

                if(entropy_j<min_entropy_j)
                {
                    min_j = j;
                    min_entropy_j = entropy_j;
                }
            }

        Add(min_j,head);
        linked[min_j] = true;
        head = min_j;
    }

    delete [] linked;
}

    


void CBayesianNetwork::InitBayesianNetwork()
{int i,j;

    for(i=0;i<IND_SIZE;i++)
    {
        m_order[i] = i;
        m_ordered[i] = i;

        for(j=0;j<IND_SIZE;j++)
        {
            m_parents[i][j] = false;
            
            if(i==j) m_paths[i][j] = 1;
            else m_paths[i][j] = 0;
        }
    }   
}

void CBayesianNetwork::CalculateA(int **&cases)
{

/*
 #ifdef VERBOSE
    cout << "begin of Parallel BIC!" << endl;
 #endif

#ifdef MPI
    MPI_ParallelBIC(this, IND_SIZE);
#else
    m_paralelo.ParallelBIC(this, IND_SIZE);
#endif

 #ifdef VERBOSE
    cout << "end of Parallel BIC!" << endl;
 #endif
*/

    for(int node=0;node<IND_SIZE;node++) {
        switch(SCORE)
        {
        case BIC_SCORE:
            m_ActualMetric[node] = BIC(node,cases);
            break;
        case K2_SCORE:
            m_ActualMetric[node] = K2(node,cases);
            break;
        }
        CalculateANode(node,cases);
 #ifdef VERBOSE
	cout << "node " << node << " done!" << endl;
 #endif

    }        
  }

void CBayesianNetwork::CalculateANode(int node, int **&cases)
{
#ifdef PARALLEL
        m_cases = cases;
  #ifdef MPI
        MPI_ParallelCalculateANode(this, IND_SIZE, node, cases);
  #else
        m_paralelo.ParallelCalculateANode(this, IND_SIZE, node);
  #endif
#else

  /*We record the actual state of the metric value. then it will be compared
  //to the new one for all the nodes.
    double old_metric;
    switch(SCORE)
    {
    case BIC_SCORE:
        old_metric = BIC(node,cases);
        break;
    case K2_SCORE:
        old_metric = K2(node,cases);
        break;
    }
*/
    for(int i=0;i<IND_SIZE;i++)
        if ((i!=node)) //&&(m_paths[i][node]==0))
        {
            m_parents[node][i] = !m_parents[node][i];

            double new_metric;
            switch(SCORE)
            {
            case BIC_SCORE:
                new_metric = BIC(node,cases);
                break;
            case K2_SCORE:
                new_metric = K2(node,cases);
                break;
            }

            m_parents[node][i] = !m_parents[node][i];
            m_A[node][i] = new_metric - m_ActualMetric[node];
        }
        else m_A[node][i] = INT_MIN;
#endif

}

double CBayesianNetwork::BIC(int node, int **&cases)
{int j,k;

    // Calculate the number of parent configurations.

    int no_j = 1;
    for(j=0;j<IND_SIZE;j++) 
        if(m_parents[node][j]) no_j *= STATES[j];

    // Allocate memory for all nijk-s.

    int ** nijk = new int*[no_j];
    for(j=0;j<no_j;j++)
    {
        nijk[j] = new int[STATES[node]];

        for(k=0;k<STATES[node];k++) nijk[j][k] = 0;
    }

    // Calculate all nijk-s.

    for(j=0;j<SEL_SIZE;j++)
    {
        // Find the parent configuration for the j-th case.

        int parent_configuration = 0;
        for(int parent=0;parent<IND_SIZE;parent++)
            if(m_parents[node][parent])
            {
                parent_configuration *= STATES[parent];
                parent_configuration += cases[j][parent];
            }

        // Update the corresponding nijk.

        nijk[parent_configuration][cases[j][node]]++;
    }


    // Calculate the BIC value.

    double bic = 0;
    for(j=0;j<no_j;j++)
    {
        int nij = 0;
        for(k=0;k<STATES[node];k++)
        {
            nij += nijk[j][k];
    
            //For rounding problems...
            if (nijk[j][k]!=0)
              bic += nijk[j][k]*log(nijk[j][k]);
        }

        //For rounding problems...
        if (nij!=0)
          bic -= nij*log(nij);
    }

///////////////////////////////
    bic -= log(SEL_SIZE)*(STATES[node]-1)*no_j/2;
///////////////////////////////
//    bic -= log(SEL_SIZE)*no_j/2;


    // Free the memory allocated for the nijk-s.

    for(j=0;j<no_j;j++)
        delete [] nijk[j];
    delete [] nijk;

    return bic;
}

//Function only to use with the Parallel version
double CBayesianNetwork::deltaBIC(int node, int nparent, int **&cases)
{int j,k;

    // Calculate the number of parent configurations.
    int no_j = 1;
    for(j=0;j<IND_SIZE;j++)
      if (j!=nparent) { 
        if(m_parents[node][j]) no_j *= STATES[j];
      }
      else {
        if(!(m_parents[node][nparent])) no_j *= STATES[nparent];
      }

    // Allocate memory for all nijk-s.
    int ** nijk = new int*[no_j];
    for(j=0;j<no_j;j++)
    {
        nijk[j] = new int[STATES[node]];

        for(k=0;k<STATES[node];k++) nijk[j][k] = 0;
    }

    // Calculate all nijk-s.
    for(j=0;j<SEL_SIZE;j++)
    {
        // Find the parent configuration for the j-th case.
        int parent_configuration = 0;
        for(int parent=0;parent<IND_SIZE;parent++)
          if (parent!=nparent) { 
            if(m_parents[node][parent])
            {
                parent_configuration *= STATES[parent];
                parent_configuration += cases[j][parent];
            } 
          }
          else {
            if(!(m_parents[node][nparent]))
            {
                parent_configuration *= STATES[nparent];
                parent_configuration += cases[j][nparent];
            }
          }
        // Update the corresponding nijk.
        nijk[parent_configuration][cases[j][node]]++;
    }

    // Calculate the BIC value.
    double deltabic = 0;
    for(j=0;j<no_j;j++)
    {
        int nij = 0;
        for(k=0;k<STATES[node];k++)
        {
            nij += nijk[j][k];
    
            //For rounding problems...
            if (nijk[j][k]!=0)
              deltabic += nijk[j][k]*log(nijk[j][k]);
        }

        //For rounding problems...
        if (nij!=0)
          deltabic -= nij*log(nij);
    }

    deltabic -= log(SEL_SIZE)*no_j/2;


    // Free the memory allocated for the nijk-s.
    for(j=0;j<no_j;j++)
        delete [] nijk[j];
    delete [] nijk;

    return deltabic;
}


void CBayesianNetwork::Remove(int node, int parent)
{
    // Remove the arc.

    m_parents[node][parent] = false;

    // The paths of the Bayesian network
    // are updated.

    for(int i=0;i<IND_SIZE;i++)
        for(int j=0;j<IND_SIZE;j++)
            if(m_paths[parent][j]>0 && m_paths[i][node]>0)
                m_paths[i][j]-=m_paths[parent][j]*m_paths[i][node];
}

/******************************************************
 ******************************************************
                        EBNA PC
 ******************************************************
 ******************************************************/

void CBayesianNetwork::LearnPC(int **&cases)
{int i,x,y;

    InitBayesianNetwork();

    // First a complete undirected graph is constructed.

    CUndirectedGraph ugraph;

    // Then d-separations are checked, arcs removed and
    // separation sets stored.

    CSet ** sepset = new CSet *[IND_SIZE];
    for(i=0;i<IND_SIZE;i++)
        sepset[i] = new CSet[IND_SIZE];

    bool loop;
    int n = 0;
    do
    {
        loop = false;
        for(x=0;x<IND_SIZE;x++)
            for(y=x+1;y<IND_SIZE;y++)
            {
                CSet adjacents;
                ugraph.Adjacents(x,adjacents);

                if(adjacents.Contains(y) &&
                   adjacents.Cardinality()>=n)
                {
                    loop = true;

                    adjacents -= y;

                    CSet subset;
                    bool stop = false;

                    for(adjacents.FirstSubset(n,subset);
                        !subset.IsNull() && !stop;
                        adjacents.NextSubset(n,subset))
                    {
                        if(DSeparate(x,y,subset,cases))
                        {
                        ugraph.Remove(x,y);
                            sepset[x][y] += subset;
                            sepset[y][x] += subset;
                            stop = true;
                        }
                    }
                }
            }

        n++;
    }
    while(loop);

    // Finally, the edges are oriented. 

    CSet * oriented = new CSet[IND_SIZE];   
    // oriented[i] stores the set consisting of the origin 
    // nodes of the edges that have been already oriented 
    // towards node i.

    CSet * not_oriented = new CSet[IND_SIZE];
    // not_oriented[i] stores the set consisting of the
    // nodes which have an edge which node i and has not been 
    // oriented yet.

    for(x=0;x<IND_SIZE;x++)
        for(y=x+1;y<IND_SIZE;y++)
            if(ugraph.Adjacents(x,y))
            {
                not_oriented[x] += y;
                not_oriented[y] += x;
            }
    // First, the v-structures are detected and the edges are 
    // oriented according to them.

    for(x=0;x<IND_SIZE;x++)
        for(y=0;y<IND_SIZE;y++)
            for(int z=0;z<IND_SIZE;z++)
                if(x!=y && y!=z && x!=z &&
                   ugraph.Adjacents(x,y) &&
                   ugraph.Adjacents(y,z) &&
                   !ugraph.Adjacents(x,z) &&
                   !sepset[x][z].Contains(y))
                {
                    if(!m_parents[x][y])
                    {
                        Add(y,x);
                        not_oriented[x] -= y;
                        not_oriented[y] -= x;
                        oriented[y] += x;
                    }

                    if(!m_parents[z][y])
                    {
                        Add(y,z);
                        not_oriented[y] -= z;
                        not_oriented[z] -= y;
                        oriented[y] += z;
                    }
                }

    for(i=0;i<IND_SIZE;i++) delete [] sepset[i];
    delete [] sepset;

    // Then the remaining edges are oriented according to
    // a maximum cardinality search ordering.
    int next;
    do
    {
        next = -1;
        int next_oriented = 0;

        for(i=0;i<IND_SIZE;i++)
            if(oriented[i].Cardinality()>next_oriented &&
               not_oriented[i].Cardinality()>0)
            {
                next = i;
                next_oriented = oriented[i].Cardinality();
            }

        if(next!=-1) 
            for(int i=0;i<IND_SIZE;i++)
                if(not_oriented[next].Contains(i))
                {
                    Add(i,next);
                    not_oriented[next] -= i;
                    not_oriented[i] -= next;
                    oriented[i] += next;
                }
    }
    while(next!=-1);

    delete [] oriented;
    delete [] not_oriented; 
}
 
bool CBayesianNetwork::DSeparate(int node1, int node2, 
                                 CSet & set, int **&cases)
{int i,x,y,z;

    // node1 and node2 are separated by set if node1 and
    // node2 are separated given any possible configuration
    // of set, therefore an independence test is done
    // for each configuration of set. The independence
    // test is performed using the chi square distribution.
    
    // First we calculate all required frequencies.

    int r_z = set.Configurations();

    int *** n_xyz = new int**[STATES[node1]];

    int * n_z = new int[r_z];
   
    for(z=0;z<r_z;z++) n_z[z] = 0;

    for(x=0;x<STATES[node1];x++)
    {
        n_xyz[x] = new int*[STATES[node2]];

        for(y=0;y<STATES[node2];y++)
        {
            n_xyz[x][y] = new int[r_z];

            for(z=0;z<r_z;z++) n_xyz[x][y][z] = 0;
        }
    }

    for(i=0;i<SEL_SIZE;i++){
        n_xyz[cases[i][node1]][cases[i][node2]][set.Configuration(cases[i])]++;
	n_z[set.Configuration(cases[i])]++;
    }

    int ** n_xz = new int*[STATES[node1]];

    for(x=0;x<STATES[node1];x++)
    {
        n_xz[x] = new int[r_z];

        for(z=0;z<r_z;z++) 
        {
            n_xz[x][z] = 0;

            for(y=0;y<STATES[node2];y++)
                n_xz[x][z] += n_xyz[x][y][z];
        }
    }

    int ** n_yz = new int*[STATES[node2]];

    for(y=0;y<STATES[node2];y++)
    {
        n_yz[y] = new int[r_z];

        for(z=0;z<r_z;z++) 
        {
            n_yz[y][z] = 0;

            for(x=0;x<STATES[node1];x++)
                n_yz[y][z] += n_xyz[x][y][z];
        }
    }

    bool connected=false;
    double chi = 0;	

    for(x=0;x<STATES[node1];x++)
      for(y=0;y<STATES[node2];y++)
 	for(z=0;z<r_z;z++){
	   //cout<<"n_xyz["<<x<<"]["<<y<<"]["<<z<<"]="<<n_xyz[x][y][z]<<endl;
           //cout<<"n_xz["<<x<<"]["<<z<<"]="<<n_xz[x][z]<<endl;
           //cout<<"n_yz["<<y<<"]["<<z<<"]="<<n_yz[y][z]<<endl;
           //cout<<"n_z["<<z<<"]="<<n_z[z]<<endl;

	  double Pxyz = double (n_xyz[x][y][z])/double(SEL_SIZE);
	  double Pxydz =double (n_xyz[x][y][z])/double(n_z[z]);
	  double Pxdz = double (n_xz[x][z])/double(n_z[z]);
	  double Pydz = double (n_yz[y][z])/double(n_z[z]);
	  //cout<<"Pxyz="<<Pxyz<<".Pxydz="<<Pxydz<<".Pxdz="<<Pxdz<<".Pydz="<<Pydz<<endl;	
	  if (Pxyz!=0)
	    chi += Pxyz*log(Pxydz/(Pxdz*Pydz));
	}	   

	chi *= 2*SEL_SIZE;

    if(chi>critchi(ALPHA_EBNAPC,(STATES[node1]-1)*(STATES[node2]-1)*r_z))
      connected = true;


/* This is the old version
    // Then an independence test is performed for each
    // configuration of set.

    bool connected = false;
    for(z=0;z<r_z && !connected;z++)
    {
        double chi = -1;

        for(x=0;x<STATES[node1];x++)
            for(y=0;y<STATES[node2];y++)
                if(n_xz[x][z]!=0 && n_yz[y][z]!=0)
                    chi += double(n_xyz[x][y][z]*n_xyz[x][y][z])/
                           double(n_xz[x][z]*n_yz[y][z]);
        chi *= SEL_SIZE;
        // If the chi statistics is higher than the critical
        // chi value then the independence assumption is
        // rejected.

        if(chi>critchi(ALPHAPC,(STATES[node1]-1)*(STATES[node2]-1)))
            connected = true;
    }
*/
    // Finally all requested memory is released.

    delete [] n_z;

    for(x=0;x<STATES[node1];x++)
    {
        for(y=0;y<STATES[node2];y++)
            delete [] n_xyz[x][y];

        delete [] n_xyz[x];
    }
    delete [] n_xyz;

    for(x=0;x<STATES[node1];x++) delete [] n_xz[x];
    delete [] n_xz;

    for(y=0;y<STATES[node2];y++) delete [] n_yz[y];
    delete [] n_yz;

    return !connected; 
}

/******************************************************
 ******************************************************
                        EBNA K2
 ******************************************************
 ******************************************************/

double CBayesianNetwork::K2(int node, int **&cases)
{int j;

    // Calculate the number of parent configurations.
    int no_j = 1;
    for(j=0;j<IND_SIZE;j++) 
        if(m_parents[node][j]) no_j *= STATES[j];

    // Allocate memory for all nijk-s.

    int ** nijk = new int*[no_j];
    for(j=0;j<no_j;j++)
    {
        nijk[j] = new int[STATES[node]];

        for(int k=0;k<STATES[node];k++) nijk[j][k] = 0;
    }

    // Calculate all nijk-s.

    for(j=0;j<SEL_SIZE;j++)
    {
        // Find the parent configuration for the j-th case.

        int parent_configuration = 0;
        for(int parent=0;parent<IND_SIZE;parent++)
            if(m_parents[node][parent])
            {
                parent_configuration *= STATES[parent];
                parent_configuration += cases[j][parent];
            }

        // Update the corresponding nijk.

        nijk[parent_configuration][cases[j][node]]++;
    }


    // Calculate the K2 value.

    double k2 = 0;
    for(j=0;j<no_j;j++)
    {
        int nij = 0;
        for(int k=0;k<STATES[node];k++)
        {
            nij += nijk[j][k];
            k2 += log_fact(nijk[j][k]);
        }
    
        k2 += log_fact(STATES[node]-1);
        k2 -= log_fact(nij+STATES[node]-1);
    }

    k2 -= log((double)SEL_SIZE)*no_j/2;

    // Free the memory allocated for the nijk-s.

    for(j=0;j<no_j;j++)
        delete [] nijk[j];
    delete [] nijk;

    return k2;
}

double CBayesianNetwork::deltaK2(int node, int nparent, int **&cases)
{int j;

    // Calculate the number of parent configurations.
    int no_j = 1;
    for(j=0;j<IND_SIZE;j++) 
      if(j!=nparent) {
        if(m_parents[node][j]) no_j *= STATES[j];
      }
      else {
        if(!(m_parents[node][nparent])) no_j *= STATES[nparent];
      }

    // Allocate memory for all nijk-s.
    int ** nijk = new int*[no_j];
    for(j=0;j<no_j;j++)
    {
        nijk[j] = new int[STATES[node]];

        for(int k=0;k<STATES[node];k++) nijk[j][k] = 0;
    }

    // Calculate all nijk-s.
    for(j=0;j<SEL_SIZE;j++)
    {
        // Find the parent configuration for the j-th case.
        int parent_configuration = 0;
        for(int parent=0;parent<IND_SIZE;parent++)
          if (parent != nparent) {
            if(m_parents[node][parent])
            {
                parent_configuration *= STATES[parent];
                parent_configuration += cases[j][parent];
            }
          }
          else {
            if(!(m_parents[node][nparent]))
            {
                parent_configuration *= STATES[nparent];
                parent_configuration += cases[j][nparent];
            }
          }
        // Update the corresponding nijk.
        nijk[parent_configuration][cases[j][node]]++;
    }


    // Calculate the K2 value.
    double k2 = 0;
    for(j=0;j<no_j;j++)
    {
        int nij = 0;
        for(int k=0;k<STATES[node];k++)
        {
            nij += nijk[j][k];
            k2 += log_fact(nijk[j][k]);
        }
    
        k2 += log_fact(STATES[node]-1);
        k2 -= log_fact(nij+STATES[node]-1);
    }

    k2 -= log((double)SEL_SIZE)*no_j/2;

    // Free the memory allocated for the nijk-s.
    for(j=0;j<no_j;j++)
        delete [] nijk[j];
    delete [] nijk;

    return k2;
}


double CBayesianNetwork::log_fact(int n)
{
    double value = 0;
    for(int i=2;i<n;i++) value += log((double)i);

    return value;
}

/*********************************************************************
 *********************************************************************
 *********************************************************************
                  FUWCTIONS TO WORK WITH LEDA GRAPHS
 *********************************************************************
 *********************************************************************
 ********************************************************************/


//#include <LEDA/graph.h>
//#include <string.h>
// It saves in a file the Bayesian Network in the form of a LEDA graph.
void CBayesianNetwork::SaveBayesianNetworkLEDA(char *filename)
{
  /* I delete this part as it does not compile it
  GRAPH<string, int> G;
  int i,j;
  char str1[20], str[20];
  node n1, n2;
  edge e1,e2;

  for(i=0; i<IND_SIZE; i++) {
    n1 = G.new_node();
    sprintf(str1, "%d", i+1);
    strcpy(str, "X");
    strcat(str, str1); 
    G.assign(n1,  str);
  }
  i=0;
  forall_nodes(n1,G) {
    j=0;
    forall_nodes(n2,G) {
        if ((i!=j)&&(m_parents[i][j])) {
        e1 = G.new_edge(n1,n2);
        G.assign(e1, 1);
    }
        j++;
    }
    i++;    
  }
  G.write(filename);
  */
  ofstream fgraph;
  int i,j, kont;

  fgraph.open(filename);
  if (!fgraph) {
    cerr << "Could not open file " << filename << ". Ignoring this file." << endl;
    return;
  }
  fgraph << "LEDA.GRAPH" << endl << "string" << endl << "int" << endl << IND_SIZE << endl;
  for(i=0; i<IND_SIZE; i++) {
    fgraph << "|{X" << i+1 << "}|" << endl;
  }
  kont=0;
  for(i=0; i<IND_SIZE; i++) {
     for(j=0; j<IND_SIZE; j++) {
       if ((i!=j)&&(m_parents[i][j])) kont++;
     }
  }
  fgraph << kont << endl;
  for(i=0; i<IND_SIZE; i++) {
     for(j=0; j<IND_SIZE; j++) {
       if ((i!=j)&&(m_parents[i][j])) 
     fgraph << j+1 << " " << i+1 << " 0 " << "|{1}|" << endl;
     }
  }
  fgraph.close();
}
