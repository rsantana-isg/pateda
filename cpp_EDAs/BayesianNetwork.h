// BayesianNetwork.h: interface for the CBayesianNetwork class.
//
// Description:
//      The class which handles Bayesian networks.
//
// Author:  Ramon Etxeberria
// Date:    1999-09-25
//
// Note:
//      The information about the problem must be initialized
//      before any  Bayesian network is created.

#ifndef _BAYESIAN_NETWORK_
#define _BAYESIAN_NETWORK_

#include "Individual.h"
#include "Set.h"

#ifdef PARALLEL
    #include "Parallel.h"
#endif

class CBayesianNetwork  
{
public:
    // The constructor. It creates an arcless Bayesian
    // network which represents a uniform probability
    // distribution.
    CBayesianNetwork();

    // The destructor.
    virtual ~CBayesianNetwork();

    // It learns the Bayesian network from the given cases.
    void Learn(int ** & cases,double ** & values,double sel_total);

    // It creates a new individual simulating the Bayesian
    // network.
    CIndividual * Simulate();

    // Function to save the learned structure in the form of a graph
    // that can be read with the LEDA libraries. This function isa called from
    // the class CSolution once the structure is build.
    void SaveBayesianNetworkLEDA(char *filename);

    // Matrixes storing the improvement of arc modifications.
    // m_A[i][j] represents the gain which could be obtained
    // when adding/removing the arc j->i when it is absent/present.
    // m_ActualMetric[node] stores the actual value of the score of 
    // the Bayesian network
    long double ** m_A;
    double *m_ActualMetric;

    // The BIC metric for the local structure (EBNA BIC) represented by
    // m_parents[node].
    double BIC(int node, int **&cases);

    // Function to execute the K2 metric for the BN (EBNA K2)
    double K2(int node, int **&cases);
    double log_fact(int n);

    //Definitions for parallelizing EBNA algorithms
#ifdef PARALLEL
    CParallel m_paralelo;
#endif
    int ** m_cases;
    double deltaBIC(int node, int nparent, int **&cases);
    double deltaK2(int node, int nparent, int **&cases);

private:
    // Adjacency matrix representing the structure of
    // the Bayesian network. If m_parents[i][j] is true
    // the there is an arc j->i in the Bayesian network.
    bool ** m_parents;

    // The conditional probability distributions of the
    // bayesian network. m_probs[i][j*(r_i-1)+k] represents the
    // probability of i being in its k-th state conditiones
    // to its parent nodes being in their j-th configuration 
    // (r_i is the number of possible states that i can take).
    double ** m_probs;

    // The matrix which stores the number of paths between
    // two nodes. m_paths[i][j] represents the number of
    // paths from j to i. m_paths[i][i] is always 1.
    int ** m_paths;

    // The topologycal order of the nodes. m_order[i]
    // represents the topologycal order of i.
    int * m_order;

    // The nodes of the Bayesian network ordered according
    // to their topologycal order.
    int * m_ordered;

    // It calculates the m_A matrix of the current Bayesian
    // networks.
    void CalculateA(int **&cases);

    // It calculates the m_A[node] values of the current
    // Bayesian network.
    void CalculateANode(int node,int **&cases);

    // It sets the structure of the Bayesian network to
    // an arcless graph.
    void InitBayesianNetwork();

    // It learns the structure of the Bayesian network using
    // the local search algorithm.
    void LearnEBNALocal(int ** & cases);

    // It learns the structure of the Bayesian network using
    // the B algorithm.
    void LearnEBNAB(int ** & cases);

    // It learn the structure of the Bayesian network as
    // in the PC algorithm.
    void LearnPC(int **& cases);

    // It learng the structure of the Tree by the MWST method 
    // of Chow and Liu.
    void LearnTree(int**&cases);

    // It learns the structure of the Bayesian network using
    // the MIMIC algorithm.
    void LearnMIMIC(int **&cases);

    // The empirical entropy of node given parent.
    double Entropy(int node, int parent, int **&cases);

    // The empirical entropy of node.
    double Entropy(int node, int **&cases);

    // It adds the arc parent->node to the Bayesian network
    // and updates all the internal data.
    void Add(int node, int parent);

    // It removes the arc parent->node from the Bayesian
    // network and updates all the internal data.
    void Remove(int node, int parent);

    // Compute the MI value between two variables in selected individuals
    double ComputeMI(int bat,int bi,int**&cases);

    // It calculates the probabilities of the Bayesian network.
    void LearnProbabilities(int **& cases,double **&values,double sel_total);

    // It returns the conditional probability distribution
    // of the orderede_node-th gene conditioned to the
    // states that its parent nodes have in instance.
    double * Probabilities(int ordered_node,int * instance);

    // It returns true if node1 and node2 are d-separated by
    // set according to the given cases. (for EBNA PC)
    bool DSeparate(int node1, int node2, 
                   CSet & set, int **&cases);

};

#endif
