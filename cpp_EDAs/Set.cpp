// Set.cpp: implementation of the CSet class.
//

#include "EDA.h"
#include "Set.h"
#include <stdlib.h>

CSet::CSet()
{
	m_members = new bool[IND_SIZE];
	for(int i=0;i<IND_SIZE;i++) m_members[i] = false;
}

CSet::~CSet()
{
	delete [] m_members;
}

int CSet::Cardinality()
{
	int card = 0;

	for(int i=0;i<IND_SIZE;i++)
		if(m_members[i]) card++;

	return card;
}

bool CSet::Contains(int element)
{
	return m_members[element];
}

void CSet::operator+=(int element)
{
	m_members[element] = true;
}

void CSet::operator-=(int element)
{
	m_members[element] = false;
}

void CSet::operator+=(CSet & set)
{
	for(int i=0;i<IND_SIZE;i++)
		m_members[i] = m_members[i] || set.m_members[i];
}

bool CSet::IsNull()
{
	return m_members==NULL;
}

void CSet::FirstSubset(int n,CSet &subset)
{
	int count = 0; 

	for(int i=0;i<IND_SIZE;i++)
		if(m_members[i] && count<n)
		{
			subset.m_members[i] = true;
			count++;
		}
		else subset.m_members[i] = false;
}

void CSet::NextSubset(int n,CSet &subset)
{
	if(LastSubset(n,subset)) 
	{
		delete [] subset.m_members;
		subset.m_members = NULL;
		return;
	}

	bool next;
	int moved = 1;
	int max_subset = IND_SIZE-1;
	do
	{
		next = false;

		int last_subset;
		for(last_subset=max_subset;
			!subset.m_members[last_subset];
			last_subset--);

		int last_set;
		for(last_set=max_subset;
			!m_members[last_set];
			last_set--);

		if(last_subset==last_set) 
		{
			max_subset = last_subset - 1;
			moved++;
			next = true;
		}
		else max_subset = last_subset;
	}
	while(next);
	
	subset.m_members[max_subset] = false;

	for(int i=max_subset+1;i<IND_SIZE;i++)
		if(m_members[i] && moved>0) 
		{
			subset.m_members[i] = true;
			moved--;
		}
		else subset.m_members[i] = false;		
}

bool CSet::LastSubset(int n,CSet &subset)
{
	bool is_last = true;

	int count = n;
	for(int i=IND_SIZE-1;count>0;i--)
		if(m_members[i])
		{
			count--;
			is_last = is_last && subset.m_members[i];
		}

	return is_last;
}

ostream & operator<<(ostream & os,CSet & set)
{
	for(int i=0;i<IND_SIZE;i++)
		if(set.m_members[i]) os << i << " ";

	return os;
}

int CSet::Configurations()
{
	int confs = 1;

	for(int i=0;i<IND_SIZE;i++)
		if(m_members[i]) confs *= STATES[i];

	return confs;
}
	

int CSet::Configuration(int * configuration)
{
	int conf = 0;
	long offset = 1;

	for(int i=0;i<IND_SIZE;i++)
		if(m_members[i])
		{
			conf += configuration[i] * offset;
			offset *= STATES[i];
		}

	return conf;
}
