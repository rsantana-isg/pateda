// Individual.cpp: implementation of the CIndividual class.
//

#include "EDA.h"
#include "Individual.h"
#include "Cache.h"
#include "Problem.h"
#include <limits.h>

// The object storing the values of all the individuals
// that have been created during the execution of the
// program. 
static CCache CACHE;

CIndividual::CIndividual()
: m_value(INT_MIN)
{
	m_genes = new int[IND_SIZE];
}

CIndividual::~CIndividual()
{
	delete [] m_genes;
}

double CIndividual::Value()
{
	// The INT_MIN value indicates that the value of
	// the individual has not been calculated yet. In
	// that case it is calculated and stored before
	// it is returned.
	if(m_value==INT_MIN)
		if(CACHING) m_value = CACHE.Value(m_genes);
		else m_value = Metric(m_genes);

	// If the penalization has to be applied to the individual
	// do it now.
	if (SIMULATION==PENALIZATION)
		m_value /= (m_values_left+1);

	return m_value;
}

int * CIndividual::Genes()
{
	return m_genes;
}

ostream & operator<<(ostream & os,CIndividual * & individual)
{
	os << individual->Value() << " ";

        os << individual->m_genes[0];
	for(int i=1;i<IND_SIZE;i++)
		os << "," << individual->m_genes[i];
        os << ".";

	return os;
}

istream & operator>>(istream & is,CIndividual * & individual)
{
  char k; //to avoid intermediate characters such as ,.

  is >> individual->m_value;
  is >> individual->m_genes[0];
  for(int i=1;i<IND_SIZE;i++)
    is >> k >> individual->m_genes[i];
  is >> k;

  return is;
}



