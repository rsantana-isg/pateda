// Population.cpp: implementation of the CPopulation class.
//

#include "Population.h"


CPopulation::CPopulation()
: m_individual(NULL), m_next(NULL), m_prev(NULL)
{
	// m_next will be the head of the list and
	// m_prev the tail.
}

CPopulation::~CPopulation()
{
	if(m_next!=NULL) delete m_next;
}

POSITION CPopulation::GetHeadPosition()
{
	return m_next;
}

void CPopulation::RemoveTail()
{
	// If the list is empty the function will fail.

	if(m_next==m_prev)
	{
		// There is only one individual in the list.

		delete m_prev;
		m_next = NULL;
		m_prev = NULL;
	}
	else
	{
		// Delete the node at the tail and update the links.

		CPopulation * last_but_one = m_prev->m_prev;
		last_but_one->m_next = NULL;

		delete m_prev;
		m_prev = last_but_one;
	}
}

POSITION CPopulation::GetNext(POSITION pos)
{
	if(pos==NULL) return NULL;
	else return pos->m_next;
}

CIndividual * & CPopulation::GetAt(POSITION pos)
{
	// If pos is NULL the function will fail.

	return pos->m_individual;
}

CIndividual * & CPopulation::GetHead()
{
	// If the list is empty  the function will fail.

	return m_next->m_individual;
}

CIndividual * & CPopulation::GetTail()
{
	// If the list is empty the function will fail.

	return m_prev->m_individual;
}

void CPopulation::AddTail(CIndividual *individual)
{
	if(m_next==NULL)
	{
		// The list is empty.

		m_prev = new CPopulation;
		m_prev->m_individual = individual;

		m_next = m_prev;
	}
	else
	{
		m_prev->m_next = new CPopulation;
		m_prev->m_next->m_prev = m_prev;
		m_prev = m_prev->m_next;
		m_prev->m_individual = individual;
	}
}

void CPopulation::InsertBefore(POSITION pos, CIndividual *individual)
{
	if(pos==NULL) return; // No position to insert.
	else if(m_next==pos)
	{
		// Insert at the head of the list.

		m_next->m_prev = new CPopulation;
		m_next = m_next->m_prev;
		m_next->m_next = pos;
		m_next->m_individual = individual;
	}
	else
	{
		POSITION aux = pos->m_prev;

		aux->m_next = new CPopulation;
		aux->m_next->m_individual = individual;
		aux->m_next->m_prev = aux;
		aux->m_next->m_next = pos;

		pos->m_prev = aux->m_next;
	}
}

void CPopulation::AddToPopulation(CIndividual * individual)
{
	POSITION pos;

	for(pos=GetHeadPosition();
	    pos!=NULL && individual->Value()<GetAt(pos)->Value();
	    pos = GetNext(pos));

	if(pos==NULL)AddTail(individual);
	else InsertBefore(pos,individual);
}




ostream & operator<<(ostream & os,CPopulation & population)
{
  POSITION pos;
  os  << " pop.size: " << POP_SIZE << " indiv.length: " << IND_SIZE << endl;      
  for(pos = population.GetHeadPosition(); pos!=NULL; pos = population.GetNext(pos))
		os << population.GetAt(pos) << endl;
  return os;
}

istream & operator>>(istream & is,CPopulation & population)
{
  char TEXT[50]; //To avoid unwanted characters  , str2[15];
  is >> TEXT >> POP_SIZE >> TEXT >> IND_SIZE;      
  for(int i=0; i<POP_SIZE; i++) {
    CIndividual * individual = new CIndividual;
    is >> individual;
    population.AddToPopulation(individual);
  }
  return is;
}

void CPopulation::ExportPopulation(char * FileName)
{
	ofstream file;

	file.open(FileName);

	file  << *this;
	/*" pop.size: " << POP_SIZE << " indiv.length: " << IND_SIZE << endl;      
	for(POSITION pos = GetHeadPosition(); pos!=NULL; pos = GetNext(pos))
		file << GetAt(pos) << endl;

		file.close();*/
}

void CPopulation::ImportPopulation(char * FileName)
{
  /*i've never tried this function!!!!!*/
	ifstream file;

	file.open(FileName);

	file  >> *this;
	/*" pop.size: " << POP_SIZE << " indiv.length: " << IND_SIZE << endl;      
	for(POSITION pos = GetHeadPosition(); pos!=NULL; pos = GetNext(pos))
		file << GetAt(pos) << endl;

		file.close();*/
}

