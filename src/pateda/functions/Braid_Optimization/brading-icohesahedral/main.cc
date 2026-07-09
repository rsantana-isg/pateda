/*****************************************************************************
*
* TQC Project: Topological Quantum Compiling/Computation
*
* main program: hashing with the icosahedral group
*
* Copyright (C) 2009 by Michele Burrello <burrello@sissa.it>
*                       Giuseppe Mussardo <mussardo@sissa.it>
*                       Xin Wan <xinwan@zimp.zju.edu.cn>
*                       Haitan Xu <haitanxu@yahoo.com.cn>
*
*
* This software is part of the TQC libraries, published under the TQC
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1.0 or (at your option) any
* later version.
* 
* You should have received a copy of the TQC Library License along with
* the TQC Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://sites.google.com/site/braidanyons/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE AND
* NON-INFRINGEMENT. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR ANYONE
* DISTRIBUTING THE SOFTWARE BE LIABLE FOR ANY DAMAGES OR OTHER
* LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT
* OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
* IN THE SOFTWARE.
*
*****************************************************************************/

#include <math.h>
#include <assert.h>
#include <getopt.h>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
using namespace boost;

#include "braid.h"
#include "icosahedral_group.cc"

int Level = 2;
int Niter = 3;

double Ax = 0.0;
double Ay = 0.0;
double Az = 1.0;
double Theta = -0.5;

void usage(int argc, char* argv[]) {
  cout << endl;
  cout << "This program implements the topological quantum hashing algorithm\n";
  cout << "using the icosahedral group, as presented in sequence presented\n";
  cout << "in Fig. 3 in arXiv:0903.1497. The sequence is\n";
  cout << endl;
  cout << "{2,2,1,1,2,2,2,2,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,\n";
  cout << " -1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,\n";
  cout << " 1,1,2,2,2,2,2,2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,\n";
  cout << " 2,2,1,1,-2,-2,1,1,1,1,1,1,-2,-2,1,1,2,2,-1,-1,2,2,2,2,\n";
  cout << " -2,-2,-2,-2,-2,-2,-1,-1,2,2,1,1,1,1,1,1,-2,-2,1,1,1,1,-2,-2}\n";
  cout << endl;
  cout << "where 1,2 denote sigma_{1,2} and -1,-2 denote sigma_{1,2}^{-1}\n";
  cout << endl;
  cout << "The targeted gate (iZ) has a unitary matrix representation\n";
  cout << "                              i    0\n";
  cout << "                              0   -i\n";
  cout << "The searching result (up to a pure phase) is\n"; 
  cout << "    -0.00040257 + i               -0.000729767 - 0.000534672i\n";
  cout << "     0.000729767 - 0.000534672i   -0.00040257 - i\n";
  cout << "The distance to the targeted gate is 0.0009902\n";
  cout << endl;
  cout << "We define a unitary matrix U(x, y, z; theta) by a rotation \n";
  cout << "axis R = (x, y, z), which does not have to be normalized, and\n";
  cout << "the rotating angle theta (in units of 2 pi). The unitary matrix\n";
  cout << "is then given by\n";
  cout << "    cos(theta pi / 2.0) I - i sin(theta pi / 2.0) (R dot Sigma)\n";
  cout << "where Sigma_{x,y,z} are Pauli matrices.\n";
  cout << endl;

  cout << "Usage: " << argv[0] 
       << " -h "
       << "-i braidLength " 
       << "-l level "
       << "-s braidSequence "
       << "-u x,y,z,theta"
       << endl;

  cout << endl;
  cout << "Examples:\n";
  cout << "(1) Th default example (as for Fig.3 of arXiv:0903.1497)\n";
  cout << "    " << argv[0] << endl;
  cout << endl;
  cout << "(2) Involve an additional processor with an L = 44 braid\n";
  cout << "representation of the icosahedral group. \n";
  cout << "    " << argv[0] << " -l 3" << endl;
  cout << endl;
  cout << "(3) Construct braid to approach U(x, y, z; theta)\n";
  cout << "    " << argv[0] << " -u 1.0,2.0,3.0,0.3" << endl;
  cout << "The unitary matrix rotates an angle of (0.3 pi) about the axis\n";
  cout << "(1.0, 2.0, 3.0).\n";
  cout << endl;
  cout << "(4) Search for the braid representation for the icosahedral\n";
  cout << "group with a fixed length L = 8.\n";
  cout << "    " << argv[0] << " -i 8" << endl;
  cout << endl;
  cout << "(5) Print out the unitary matrix representation of a braid\n";
  cout << "sequence, e.g., {1,1,2,2,-1,-1,-1,-1}.\n";
  cout << "    " << argv[0] << " -s 1,1,2,2,-1,-1,-1,-1" << endl;
  cout << endl;

  
  return;
}

int init(int argc, char* argv[]) {
  int c, info;
  int errflg = 0;
  int braid_length;
  vector<UMatrix> group;
  vector<Braid> braids_to_search;

  while ((c = getopt(argc, argv, "hi:l:s:u:")) != EOF) 
    switch (c) {
    case 'h':
      usage(argc, argv);
      exit(0);
    case 'i':
      // Find the braid representation with certain length
      braid_length = atoi(optarg);
      icosahedral_unitary_matrices(group);
      info = weave_brute_force_vector_search(braid_length, group, braids_to_search, 1);
      exit(1);
      break;
    case 'l':
      Level = atoi(optarg);
      if (Level > 3) {
	cout << "Set to the maximum level of 3." << endl;
	Level = 3;
      }
      break;
    case 's':
      // Split the string into tokens ( use ',' as delimiters )
      // We need copies of the input only, and adjacent tokens are compressed
      {
	vector<std::string> ResultCopy;
	split(ResultCopy, optarg, is_any_of(","), token_compress_on);
	int* seq = new int [ResultCopy.size()];
	for(unsigned int nIndex = 0; nIndex < ResultCopy.size(); nIndex++)
	  seq[nIndex] = atoi(ResultCopy[nIndex].c_str());
	Braid b(ResultCopy.size(),seq);
	cout << b << "a b | theta: ";
	b.print_abt();
	cout << ">>> After simplification, " << endl;
	b.simplify();
	cout << b << "a b | theta: ";
	b.print_abt();
	exit(0);
      }
      break;
    case 'u':
      // read parameter string x,y,z,theta for a unitary matrix
      //   that rotate theta*pi around axis (x,y,z)
      {
	vector<std::string> ResultCopy;
	split(ResultCopy, optarg, is_any_of(","), token_compress_on);
	if (ResultCopy.size() == 4) {
	  Ax = atof(ResultCopy[0].c_str());
	  Ay = atof(ResultCopy[1].c_str());
	  Az = atof(ResultCopy[2].c_str());
	  Theta = atof(ResultCopy[3].c_str());
	} else {
	  errflg++;
	}
      }
      break;
    default:
      errflg++;
      break;
    }
  
  if (errflg) {
    usage(argc, argv);
    exit(1);
  }

  return 0;
}

int main(int argc, char* argv[]) {

  int c = init(argc, argv);

  cout << "Preparing the icosahedral group ..." << endl;
  vector<UMatrix> group;
  icosahedral_unitary_matrices(group);

  
  int inverse_group[60];
  get_icosahedral_inverse(inverse_group, group);
  int comp_table[60][60];
  get_icosahedral_map(comp_table, group);
  
  Braid braid_to_search;
  double d;
  
  // three-level search of the target gate
  cout << "Loading the L = 8 braid representation ..." << endl;
  int n_iter = 3;
  vector<Braid> pseudo_group_8;
  braid_representation_8(pseudo_group_8);
  vector<Braid> composite_braids_8;
  form_mesh(n_iter, pseudo_group_8, composite_braids_8, inverse_group, comp_table);
  
  cout << "Loading the L = 24 braid representation ..." << endl;
  int close = 1;
  vector<Braid> pseudo_group_24;
  braid_representation_24(pseudo_group_24);


  /* ADDED BY ROBERTO TO TEST ERROR OF THE 24 AND 44 REPRESENTATIONS */
  vector<Braid> pseudo_group_44;
  braid_representation_44(pseudo_group_44);

  cout<<"Here "<<group[0]<<" "<<endl;
  cout<<"Here 1: "<<pseudo_group_24[0]<<endl;

  for(int kk=0;kk<60;kk++)
    {
      cout<<24<<" "<<kk<<" ";
     aux_print_target_braid(group[kk], pseudo_group_24[kk]);
     cout<<44<<" "<<kk<<" ";
     aux_print_target_braid(group[kk], pseudo_group_44[kk]);
    }

  cout<<"FINISH Here: "<<endl;
  vector<Braid> composite_braids_24;
  form_mesh(n_iter, pseudo_group_24, composite_braids_24, inverse_group, comp_table, close);
  
  //vector<Braid> pseudo_group_44;
  vector<Braid> composite_braids_44;



  if (Level > 2) { 
    cout << "Loading the L = 44 braid representation ..." << endl;
    braid_representation_44(pseudo_group_44);
    form_mesh(n_iter, pseudo_group_44, composite_braids_44, inverse_group, comp_table, close);
  }
  
  double ax = Ax;
  double ay = Ay;
  double az = Az;
  double theta = Theta;
  UMatrix target_u;
  int info = 1;
  while (info) {
    target_u.set_axis_angle(ax, ay, az, 2.0 * M_PI * theta);
    cout << endl << ">>> Targeted gate:" << target_u << endl;
    braid_to_search.clear();
    search_within_mesh(target_u, braid_to_search, composite_braids_8);
    cout << ">>> After preprocessor (L = 8), " << endl;
    print_target_braid(target_u, braid_to_search);
    search_within_mesh(target_u, braid_to_search, composite_braids_24);
    cout << ">>> After main processor (L = 24), " << endl;
    print_target_braid(target_u, braid_to_search);
    if (Level > 2) { 
      search_within_mesh(target_u, braid_to_search, composite_braids_44);
      cout << ">>> After a third processor (L = 44), " << endl;
      print_target_braid(target_u, braid_to_search);
    }
    cout << ">>> After simplification, " << endl;
    braid_to_search.simplify();
    print_target_braid(target_u, braid_to_search);

    cout << "Try yourself with any unitary matrix as a target." << endl;
    cout << "Input rotation axis (x y z):" << endl;
    scanf("%lf %lf %lf", &ax, &ay, &az);
    cout << "Input rotation angle (in units of Pi):" << endl;
    scanf("%lf", &theta);
  }

  return 0;
}

