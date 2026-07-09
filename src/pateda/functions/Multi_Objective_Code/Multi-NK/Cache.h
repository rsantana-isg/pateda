// Cache.h: interface for the CCache class.
//
// Description:
//		The structure for storing individual's values. Its use
//		is recommended for small individuals whose value requires
//		a lot of computation. 
//
// Author:	Ramon Etxeberria
// Date:	1999-09-25
//
// Note:
//		Use with care. It requires a lot of memory. Use only
//		a single global object to store all the values.

#ifndef _CACHE_
#define _CACHE_

class CCache  
{
public:
	// The constructor.
	CCache();

	// The destructor. It frees all the used memory.
	virtual ~CCache();

	// It returns the value of the given genes. If the
	// value is not stored in the structure yet, it is
	// calculated and stored for further use.
	double Value(int * genes);

private:
	// Another constructor. m_state is initialized with
	// the given state.
	CCache(int state);

	// The value of the individual corresponding to the
	// current node.
	double m_value;

	// The state to which the value corresponds.
	int m_state;

	// The next gene of the individual.
	CCache * m_other_genes;

	// If m_state does not correspond with the state
	// of the individual being looked for it must
	// be searched in this variable.
	CCache * m_other_states;

	// It moves to the individual with the given state
	// as its next one. If there is not such an individual
	// it will be created. It is used to iterate through
	// the structure.
	CCache * NextGene(int state);
};

#endif
