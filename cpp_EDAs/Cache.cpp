// Cache.cpp: implementation of the CCache class.
//

#include "EDA.h"
#include "Cache.h"
#include "Problem.h"
#include <stdio.h>
#include <limits.h>

CCache::CCache()
: m_state(0),m_value(INT_MIN),
  m_other_states(NULL),m_other_genes(NULL)
{
}

CCache::~CCache()
{
	delete m_other_states;
	delete m_other_genes;
}


CCache * CCache::NextGene(int state)
{
	if(m_other_genes==NULL)
	{
		m_other_genes = new CCache(state);
		return m_other_genes;
	}
	else 
	{
		CCache * aux;
		for(aux = m_other_genes;
		    aux->m_state!=state && aux->m_other_states!=NULL;
		    aux = aux->m_other_states);

		if(aux->m_state==state) return aux;
		else
		{
			aux->m_other_states = new CCache(state);
			return aux->m_other_states;
		}
	}
}

CCache::CCache(int state)
: m_state(state),m_value(INT_MIN),
  m_other_states(NULL),m_other_genes(NULL)
{
}

double CCache::Value(int * genes)
{
	CCache * aux = this;

	for(int i=0;i<IND_SIZE;i++)
		aux = aux->NextGene(genes[i]);

	if(aux->m_value==INT_MIN) aux->m_value = Metric(genes);

	return aux->m_value;
}
