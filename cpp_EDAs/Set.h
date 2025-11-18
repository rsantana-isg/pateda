// Set.h: interface for the CSet class.
//
// Description:
//		The class which handles sets.
//
// Note:
//		It can only contain numbers from 0 to IND_SIZE-1.
//
// Author:	Ramon Etxeberria
// Date:	1999-11-29

#ifndef _SET_
#define _SET_

#include <iostream.h>

class CSet  
{
public:

	// The constructor. It creates a null set.
	CSet();

	// The destructor.
	virtual ~CSet();

	// It return true if the element is in the set, false
	// otherwise.
	bool Contains(int element);

	// It adds the element to the set.
	void operator+=(int element);

	// It removes the element from the set.
	void operator-=(int element);

	// It adds the elements in set to the current set.
	void operator+=(CSet & set);

	// It returns the 'first' subset of n elements.
	void FirstSubset(int n,CSet & subset);

	// It returns the 'next' subset of n elements. It
	// does not check whether the given subset has n
	// elements. If that is the case then the result
	// is unpredictable. If there is no next subset
	// a null set is returned.
	void NextSubset(int n,CSet & subset);

	// It return true if the given subset is the last
	// subset available with n elements.
	bool LastSubset(int n,CSet & subset);

	// It returns true if the set is null, false
	// otherwise.
	bool IsNull();

	// It returns the number of elements in the set.
	int Cardinality();

	// It returns the number of configurations that
	// the variables in the set can take.
	int Configurations();

	// Given a posible configuration of the set it
	// return its index number.
	int Configuration(int * configuration);

	// It prints the elements of the set.
	friend ostream & operator<<(ostream & os, CSet & set);

protected:
	
	// m_member[i] will be true if i is in the set,
	// false otherwise. If m_members is NULL it
	// it represents a null (incorrect) set.
	bool * m_members;
};

#endif
