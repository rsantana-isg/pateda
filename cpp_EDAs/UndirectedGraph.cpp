// UndirectedGraph.cpp: implementation of the CUndirectedGraph class.
//

#include "UndirectedGraph.h"
#include "EDA.h"

CUndirectedGraph::CUndirectedGraph()
{int i,j;
	m_edges = new bool*[IND_SIZE];
	for(i=0;i<IND_SIZE;i++) 
		m_edges[i] = new bool[IND_SIZE];

	for(i=0;i<IND_SIZE;i++)
		for(j=0;j<IND_SIZE;j++)
			if(i==j) m_edges[i][j] = false;
			else m_edges[i][j] = true;
}

CUndirectedGraph::~CUndirectedGraph()
{
	for(int i=0;i<IND_SIZE;i++) delete [] m_edges[i];
	delete [] m_edges;
}

void CUndirectedGraph::Add(int node1,int node2)
{
	m_edges[node1][node2] = true;
	m_edges[node2][node1] = true;
}

void CUndirectedGraph::Remove(int node1,int node2)
{
	m_edges[node1][node2] = false;
	m_edges[node2][node1] = false;
}

bool CUndirectedGraph::Adjacents(int node1,int node2)
{
	return m_edges[node1][node2];
}

void CUndirectedGraph::Adjacents(int node,CSet & adjacents)
{
	adjacents.FirstSubset(0,adjacents);

	for(int i=0;i<IND_SIZE;i++)
		if(m_edges[node][i]) adjacents += i;
}

