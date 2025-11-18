// UndirectedGraph.h: interface for the CUndirectedGraph class.
//
// Description:
//		The class which handles undirected graphs.
//
// Author:	Ramon Etxeberria
// Date:	1999-11-29

#ifndef _UNDIRECTED_GRAPH_
#define _UNDIRECTED_GRAPH_

#include "Set.h"

class CUndirectedGraph  
{
public:

	// The constructor. It creates a complete graph.
	CUndirectedGraph();

	// The destructor.
	virtual ~CUndirectedGraph();

	// It return the adjacency set of node.
	void Adjacents(int node,CSet & adjacents);

	// It returns true if both nodes are adjacents.
	bool Adjacents(int node1, int node2);

	// It adds the edge node1--node2.
	void Add(int node1, int node2);

	// It removes the edge node1--node2.
	void Remove(int node1, int node2);

protected:

	// m_edges[i][j] is true when the edge i--j is
	// present. Obviously m_edges[i][j]==m_edges[j][i].
	bool ** m_edges;
};

#endif
