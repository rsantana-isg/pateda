/*****************************************************************************
*
* TQC Project: Topological Quantum Compiling/Computation
*
* Icosahedral group library: matrix and braid representations 
*
* Copyright (C) 2009 by Michele Burrello <burrello@sissa.it>
*                       Giuseppe Mussardo <mussardo@sissa.it>
*                       Xin Wan <xinwan@zimp.zju.edu.cn>
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
#include <iostream>
#include <iomanip>
#include <fstream>

typedef long long int INT;
// typedef int INT;

int form_mesh(int n_iter, vector<Braid> pseudo_group, vector<Braid>& composite_braids, int inverse_group[60], int comp_table[60][60], int close = 0) {
  int n_group = pseudo_group.size();
  int MAX_INT = static_cast<int> (pow(static_cast<double>(n_group), static_cast<double>(n_iter)) + 0.5);
  Braid b1;

  INT c, bit, index;
  Braid b;
  composite_braids.clear();
  for (int i = 0; i < MAX_INT; i++) {
    b.clear();
    c = i;
    index = 0;
    for (int ibit = 0; ibit < n_iter; ibit++) {
      bit = c % n_group;
      b += pseudo_group[bit];
      index = comp_table[index][bit];
      c /= n_group;
    }
    if (close)
      b += pseudo_group[inverse_group[index]];
    composite_braids.push_back(b);
  }
  return 0;
} 

int search_within_mesh(UMatrix target, Braid& braid_to_search, vector<Braid>& composite_braids) {
  double MIN_ERROR = 1.0e-14; // to screen for machine errors
  Braid b0 = braid_to_search;
  Braid b1;
  int i_target;
  double d;
  double d_min = 10.0;
  UMatrix transformed_target;
  transformed_target = ~(braid_to_search.rep()) * target;
  for (int i = 0; i < composite_braids.size(); i++) {
    d = composite_braids[i].distance(transformed_target);
    if (d < d_min) {
      d_min = d;
      i_target = i;
    }
  }
  braid_to_search += composite_braids[i_target];
  return 0;
}

void print_target_braid(UMatrix target_u, Braid result_b) {
  cout << result_b;
  cout << "Canonical format:";
  result_b.rep().print_ab_format();
  cout << "Distance to the targeted gate: " << result_b.distance(target_u) << endl;
  cout << endl;
  fflush(stdout);
  return;
}

void aux_print_target_braid(UMatrix target_u, Braid result_b) {
  cout << result_b.distance(target_u) << endl;
  return;
}

int icosahedral_group_search_with_map(int n_iter, vector<Braid> pseudo_group, UMatrix target, Braid& braid_to_search, int inverse_group[60], int comp_table[60][60]) {
  int n_group = pseudo_group.size();
  int MAX_INT = static_cast<int> (pow(static_cast<double>(n_group), static_cast<double>(n_iter)) + 0.5);
  double MIN_ERROR = 1.0e-14; // to screen for machine errors
  Braid b0 = braid_to_search;
  Braid b1;
  UMatrix unity;

  INT c, bit, index;
  Braid b;
  double d;
  double d_min = 10.0;
  for (int i = 0; i < MAX_INT; i++) {
    b.clear();
    c = i;
    index = 0;
    for (int ibit = 0; ibit < n_iter; ibit++) {
      bit = c % n_group;
      b += pseudo_group[bit];
      index = comp_table[index][bit];
      c /= n_group;
    }
    b += pseudo_group[inverse_group[index]];
    if (b.distance(unity) > MIN_ERROR) {
      b1 = b0;
      b1 += b;
      d = b1.distance(target);
      if (d < d_min) {
	d_min = d;
	braid_to_search = b1;
      }
    }
  }

  return 0;
}

int icosahedral_group_search(int n_iter, vector<Braid> pseudo_group, UMatrix target, Braid& braid_to_search) {
  int n_group = pseudo_group.size();
  int MAX_INT = static_cast<int> (pow(static_cast<double>(n_group), static_cast<double>(n_iter)) + 0.5);
  double MIN_ERROR = 1.0e-14; // to screen for machine errors
  Braid b0 = braid_to_search;
  Braid b1;
  UMatrix unity;

  INT c, bit;
  Braid b;
  double d;
  double d_min = 10.0;
  for (int i = 0; i < MAX_INT; i++) {
    b.clear();
    c = i;
    for (int ibit = 0; ibit < n_iter; ibit++) {
      bit = c % n_group;
      b += pseudo_group[bit];
      c /= n_group;
    }
    if (b.distance(unity) > MIN_ERROR) {
      b1 = b0;
      b1 += b;
      d = b1.distance(target);
      if (d < d_min) {
	d_min = d;
	braid_to_search = b1;
      }
    }
  }

  return 0;
}

int weave_brute_force_vector_search(int braid_length, vector<UMatrix> target_vector, vector<Braid>& braid_vector, int print = 0) {
  int MAX_LEN = sizeof(INT) * 8;
//   cout << "Maximum length in the current implementation: " << MAX_LEN << endl;
//   cout << braid_length << endl;
  assert(braid_length < MAX_LEN);
  double MIN_ERROR = 1.0e-14; // to screen for machine errors
  
  int n_target = target_vector.size();

  braid_vector.clear();
  Braid braid; // empty braid
  for (int i_target = 0; i_target < n_target; i_target++)
    braid_vector.push_back(braid);

  INT MAX_INT = 1;
  MAX_INT <<= braid_length;
  vector<int> seq;
  double MIN_DISTANCE = 10.0; // larger than any possible distance 
  vector<double> distance_vector(n_target, MIN_DISTANCE);
  double d;
  INT c, bit;
  for (INT i = 0; i < MAX_INT; i++) {
    seq.clear();
    c = i;
    for (int ibit = 0; ibit < braid_length / 2; ibit++) {
      bit = c % 4 + 1;
      if ((ibit > 0) && (bit == 5 - seq.back())) {
	c = -1;
	break;
      }
      seq.push_back(bit);
      seq.push_back(bit);
      c /= 4;
    }
    if (c != -1) { 
      Braid b(seq.size(), &seq[0]);
//       cout << b << "Distance: " << b.distance(target) << endl;
      for (int i_target = 0; i_target < n_target; i_target++) {
	d = b.distance(target_vector[i_target]);
	if ((d < distance_vector[i_target]) && (d > MIN_ERROR)) { 
	  // d > MIN_ERROR is to prevent a complicated trivial braid that is nothing but identity
	  braid_vector[i_target] = b;
	  distance_vector[i_target] = d;
	}
      }
    }
  }
  
  if (print)
    for (int i_target = 0; i_target < n_target; i_target++)
      print_target_braid(target_vector[i_target], braid_vector[i_target]);

  return 0;
}

int icosahedral_unitary_matrices(vector<UMatrix>& icosahedral, int print = 0) {
  UMatrix u;
  
  icosahedral.push_back(u.set_axis_angle(0.0, 0.0, 1.0, 0.0));

  double tau = (sqrt(5.0) + 1.0) / 2.0;
  double phi = 0.4 * M_PI;
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0, tau, phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0, -tau, phi));
  icosahedral.push_back(u.set_axis_angle(0.0, -1.0, tau, phi));
  icosahedral.push_back(u.set_axis_angle(0.0, -1.0, -tau, phi));
  icosahedral.push_back(u.set_axis_angle(tau, 0.0, 1.0, phi));
  icosahedral.push_back(u.set_axis_angle(-tau, 0.0, 1.0, phi));
  icosahedral.push_back(u.set_axis_angle(tau, 0.0, -1.0, phi));
  icosahedral.push_back(u.set_axis_angle(-tau, 0.0, -1.0, phi));
  icosahedral.push_back(u.set_axis_angle(1.0, tau, 0.0, phi));
  icosahedral.push_back(u.set_axis_angle(1.0, -tau, 0.0, phi));
  icosahedral.push_back(u.set_axis_angle(-1.0, tau, 0.0, phi));
  icosahedral.push_back(u.set_axis_angle(-1.0, -tau, 0.0, phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0, tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0, -tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(0.0, -1.0, tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(0.0, -1.0, -tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(tau, 0.0, 1.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-tau, 0.0, 1.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(tau, 0.0, -1.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-tau, 0.0, -1.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(1.0, tau, 0.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(1.0, -tau, 0.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-1.0, tau, 0.0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-1.0, -tau, 0.0, 2.0 * phi));

  icosahedral.push_back(u.set_axis_angle(0.0, 0.0, 1.0, M_PI));
  icosahedral.push_back(u.set_axis_angle(1.0, 1.0 / tau, tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(-1.0, 1.0 / tau, tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(1.0, -1.0 / tau, tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(-1.0, -1.0 / tau, tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(1.0, 0.0, 0.0, M_PI));
  icosahedral.push_back(u.set_axis_angle(tau, 1.0, 1.0 / tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(tau, -1.0, 1.0 / tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(tau, 1.0, -1.0 / tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(tau, -1.0, -1.0 / tau, M_PI));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0, 0.0, M_PI));
  icosahedral.push_back(u.set_axis_angle(1.0 / tau, tau, 1.0, M_PI));
  icosahedral.push_back(u.set_axis_angle(-1.0 / tau, tau, 1.0, M_PI));
  icosahedral.push_back(u.set_axis_angle(1.0 / tau, tau, -1.0, M_PI));
  icosahedral.push_back(u.set_axis_angle(-1.0 / tau, tau, -1.0, M_PI));

  phi = 2.0 * M_PI / 3.0;
  icosahedral.push_back(u.set_axis_angle(tau, 0.0, 1.0 + 2.0 * tau, phi));
  icosahedral.push_back(u.set_axis_angle(-tau, 0.0, 1.0 + 2.0 * tau, phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, tau, 0, phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, -tau, 0, phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0 + 2.0 * tau, tau, phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0 + 2.0 * tau, -tau, phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, phi));
  icosahedral.push_back(u.set_axis_angle(-1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, phi));
  icosahedral.push_back(u.set_axis_angle(-1.0 - 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, phi));
  icosahedral.push_back(u.set_axis_angle(tau, 0.0, 1.0 + 2.0 * tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-tau, 0.0, 1.0 + 2.0 * tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, tau, 0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, -tau, 0, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0 + 2.0 * tau, tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(0.0, 1.0 + 2.0 * tau, -tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(1.0 + 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi));
  icosahedral.push_back(u.set_axis_angle(-1.0 - 2.0 * tau, -1.0 - 2.0 * tau, 1.0 + 2.0 * tau, 2.0 * phi));

  if (print)
    for (int i = 0; i < icosahedral.size(); i++) 
      icosahedral[i].print_abt();
  
  return 0;
}

int get_icosahedral_inverse(int inverse_group[60], vector<UMatrix> icosahedral, int print = 0) {
  double MIN_ERROR = 1.0e-14; // to screen for machine errors
  UMatrix unity;
  double distance;
  for (int i = 0; i < 60; i++) {
    for (int j = 0; j < 60; j++) {
      if ((icosahedral[i] * icosahedral[j]).distance(unity) < MIN_ERROR) 
	inverse_group[i] = j;
    }
  }
  if (print) {
    for (int i = 0; i < 60; i++) {
      cout << inverse_group[i] << " ";
    }
    cout << endl;
  }
  return 0;
}

int get_icosahedral_map(int comp_table[60][60], vector<UMatrix> icosahedral, int print = 0) {
  UMatrix product;
  double MIN_ERROR = 1.0e-14; // to screen for machine errors
  for (int i = 0; i < 60; i++) 
    for (int j = 0; j < 60; j++) {
      comp_table[i][j] = -1;
      product = icosahedral[i] * icosahedral[j];
      for (int k = 0; k < 60; k++) {
	if (product.distance(icosahedral[k]) < MIN_ERROR) 
	  comp_table[i][j] = k;
      }
    }
  if (print) {
    for (int i = 0; i < 60; i++) {
      for (int j = 0; j < 60; j++) {
	cout << comp_table[i][j] << " ";
      }
      cout << endl;
    }
  }
  return 0;
}

void push_back_braid_from_sequence(int length, int* seq, vector<Braid>& vector_braid) {
  Braid b(length, seq);
  vector_braid.push_back(b);
  return;
}

void braid_representation_8(vector<Braid>& vector_braid) {
  vector_braid.clear();
  int length = 8;

  int seq1[] = {-2,-2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq1, vector_braid);
  int seq2[] = {-2,-2,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq2, vector_braid);
  int seq3[] = {2,2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq3, vector_braid);
  int seq4[] = {-1,-1,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq4, vector_braid);
  int seq5[] = {1,1,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq5, vector_braid);
  int seq6[] = {-2,-2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq6, vector_braid);
  int seq7[] = {-1,-1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq7, vector_braid);
  int seq8[] = {1,1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq8, vector_braid);
  int seq9[] = {2,2,-1,-1,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq9, vector_braid);
  int seq10[] = {-2,-2,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq10, vector_braid);
  int seq11[] = {1,1,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq11, vector_braid);
  int seq12[] = {2,2,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq12, vector_braid);
  int seq13[] = {-1,-1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq13, vector_braid);
  int seq14[] = {-2,-2,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq14, vector_braid);
  int seq15[] = {2,2,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq15, vector_braid);
  int seq16[] = {2,2,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq16, vector_braid);
  int seq17[] = {-2,-2,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq17, vector_braid);
  int seq18[] = {-2,-2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq18, vector_braid);
  int seq19[] = {2,2,2,2,2,2,2,2};
  push_back_braid_from_sequence(length, seq19, vector_braid);
  int seq20[] = {-2,-2,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq20, vector_braid);
  int seq21[] = {2,2,-1,-1,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq21, vector_braid);
  int seq22[] = {-1,-1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq22, vector_braid);
  int seq23[] = {-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq23, vector_braid);
  int seq24[] = {1,1,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq24, vector_braid);
  int seq25[] = {1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq25, vector_braid);
  int seq26[] = {-1,-1,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq26, vector_braid);
  int seq27[] = {-2,-2,-2,-2,1,1,2,2};
  push_back_braid_from_sequence(length, seq27, vector_braid);
  int seq28[] = {-1,-1,2,2,2,2,2,2};
  push_back_braid_from_sequence(length, seq28, vector_braid);
  int seq29[] = {2,2,1,1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq29, vector_braid);
  int seq30[] = {1,1,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq30, vector_braid);
  int seq31[] = {-2,-2,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq31, vector_braid);
  int seq32[] = {1,1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq32, vector_braid);
  int seq33[] = {-1,-1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq33, vector_braid);
  int seq34[] = {2,2,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq34, vector_braid);
  int seq35[] = {-1,-1,-1,-1,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq35, vector_braid);
  int seq36[] = {2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq36, vector_braid);
  int seq37[] = {1,1,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq37, vector_braid);
  int seq38[] = {-2,-2,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq38, vector_braid);
  int seq39[] = {2,2,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq39, vector_braid);
  int seq40[] = {-1,-1,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq40, vector_braid);
  int seq41[] = {-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq41, vector_braid);
  int seq42[] = {2,2,1,1,1,1,2,2};
  push_back_braid_from_sequence(length, seq42, vector_braid);
  int seq43[] = {1,1,1,1,1,1,2,2};
  push_back_braid_from_sequence(length, seq43, vector_braid);
  int seq44[] = {2,2,1,1,1,1,1,1};
  push_back_braid_from_sequence(length, seq44, vector_braid);
  int seq45[] = {-2,-2,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq45, vector_braid);
  int seq46[] = {2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq46, vector_braid);
  int seq47[] = {-1,-1,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq47, vector_braid);
  int seq48[] = {-2,-2,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq48, vector_braid);
  int seq49[] = {2,2,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq49, vector_braid);
  int seq50[] = {-1,-1,-2,-2,1,1,2,2};
  push_back_braid_from_sequence(length, seq50, vector_braid);
  int seq51[] = {1,1,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq51, vector_braid);
  int seq52[] = {-2,-2,-1,-1,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq52, vector_braid);
  int seq53[] = {-2,-2,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq53, vector_braid);
  int seq54[] = {-1,-1,-1,-1,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq54, vector_braid);
  int seq55[] = {2,2,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq55, vector_braid);
  int seq56[] = {-2,-2,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq56, vector_braid);
  int seq57[] = {2,2,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq57, vector_braid);
  int seq58[] = {1,1,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq58, vector_braid);
  int seq59[] = {1,1,2,2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq59, vector_braid);
  int seq60[] = {-2,-2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq60, vector_braid);

  return;
}

void braid_representation_24(vector<Braid>& vector_braid) {
  vector_braid.clear();
  int length = 24;

  int seq1[] = {-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq1, vector_braid);
  int seq2[] = {1,1,1,1,-2,-2,1,1,2,2,1,1,2,2,-1,-1,-2,-2,-2,-2,1,1,2,2};
  push_back_braid_from_sequence(length, seq2, vector_braid);
  int seq3[] = {-1,-1,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1,2,2,2,2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq3, vector_braid);
  int seq4[] = {2,2,1,1,-2,-2,-2,-2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq4, vector_braid);
  int seq5[] = {-2,-2,-1,-1,2,2,2,2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq5, vector_braid);
  int seq6[] = {2,2,1,1,-2,-2,-1,-1,2,2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq6, vector_braid);
  int seq7[] = {2,2,1,1,-2,-2,1,1,1,1,1,1,-2,-2,1,1,2,2,-1,-1,2,2,2,2};
  push_back_braid_from_sequence(length, seq7, vector_braid);
  int seq8[] = {-2,-2,-1,-1,2,2,-1,-1,-1,-1,-1,-1,2,2,-1,-1,-2,-2,1,1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq8, vector_braid);
  int seq9[] = {-2,-2,-1,-1,2,2,1,1,-2,-2,-1,-1,2,2,1,1,2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq9, vector_braid);
  int seq10[] = {-1,-1,-2,-2,-1,-1,-2,-2,1,1,1,1,1,1,2,2,1,1,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq10, vector_braid);
  int seq11[] = {2,2,-1,-1,2,2,1,1,2,2,1,1,1,1,1,1,-2,-2,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq11, vector_braid);
  int seq12[] = {1,1,2,2,1,1,2,2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq12, vector_braid);
  int seq13[] = {-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,2,2,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq13, vector_braid);
  int seq14[] = {2,2,1,1,-2,-2,1,1,1,1,1,1,2,2,-1,-1,2,2,1,1,1,1,1,1};
  push_back_braid_from_sequence(length, seq14, vector_braid);
  int seq15[] = {-2,-2,-1,-1,2,2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq15, vector_braid);
  int seq16[] = {1,1,1,1,1,1,2,2,-1,-1,2,2,1,1,1,1,1,1,-2,-2,1,1,2,2};
  push_back_braid_from_sequence(length, seq16, vector_braid);
  int seq17[] = {-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,2,2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq17, vector_braid);
  int seq18[] = {2,2,2,2,-1,-1,-2,-2,-2,-2,1,1,1,1,-2,-2,-2,-2,-1,-1,2,2,2,2};
  push_back_braid_from_sequence(length, seq18, vector_braid);
  int seq19[] = {1,1,-2,-2,1,1,2,2,-1,-1,2,2,-1,-1,2,2,2,2,1,1,1,1,2,2};
  push_back_braid_from_sequence(length, seq19, vector_braid);
  int seq20[] = {-1,-1,2,2,-1,-1,-2,-2,1,1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq20, vector_braid);
  int seq21[] = {-2,-2,-2,-2,1,1,2,2,2,2,-1,-1,-1,-1,2,2,2,2,1,1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq21, vector_braid);
  int seq22[] = {2,2,-1,-1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq22, vector_braid);
  int seq23[] = {-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq23, vector_braid);
  int seq24[] = {-2,-2,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,1,1,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq24, vector_braid);
  int seq25[] = {1,1,1,1,2,2,1,1,2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq25, vector_braid);
  int seq26[] = {1,1,2,2,-1,-1,-2,-2,1,1,-2,-2,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq26, vector_braid);
  int seq27[] = {-2,-2,1,1,2,2,1,1,1,1,-2,-2,1,1,-2,-2,1,1,2,2,2,2,2,2};
  push_back_braid_from_sequence(length, seq27, vector_braid);
  int seq28[] = {-2,-2,1,1,1,1,-2,-2,1,1,1,1,1,1,2,2,-1,-1,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq28, vector_braid);
  int seq29[] = {2,2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,2,2,-1,-1,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq29, vector_braid);
  int seq30[] = {-2,-2,-2,-2,-2,-2,-1,-1,2,2,1,1,1,1,1,1,-2,-2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq30, vector_braid);
  int seq31[] = {-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq31, vector_braid);
  int seq32[] = {-2,-2,-2,-2,1,1,2,2,1,1,2,2,2,2,1,1,2,2,1,1,1,1,2,2};
  push_back_braid_from_sequence(length, seq32, vector_braid);
  int seq33[] = {2,2,2,2,-1,-1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq33, vector_braid);
  int seq34[] = {2,2,2,2,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,1,1,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq34, vector_braid);
  int seq35[] = {-2,-2,-2,-2,1,1,1,1,1,1,2,2,2,2,-1,-1,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq35, vector_braid);
  int seq36[] = {-2,-2,1,1,2,2,2,2,2,2,-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq36, vector_braid);
  int seq37[] = {1,1,2,2,2,2,2,2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq37, vector_braid);
  int seq38[] = {1,1,-2,-2,-2,-2,1,1,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2};
  push_back_braid_from_sequence(length, seq38, vector_braid);
  int seq39[] = {-1,-1,2,2,2,2,-1,-1,-2,-2,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq39, vector_braid);
  int seq40[] = {-1,-1,-2,-2,-2,-2,-2,-2,-1,-1,2,2,2,2,1,1,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq40, vector_braid);
  int seq41[] = {1,1,-2,-2,1,1,1,1,1,1,2,2,2,2,2,2,-1,-1,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq41, vector_braid);
  int seq42[] = {2,2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq42, vector_braid);
  int seq43[] = {1,1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,2,2,1,1,2,2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq43, vector_braid);
  int seq44[] = {-2,-2,1,1,1,1,2,2,1,1,2,2,-1,-1,-2,-2,-2,-2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq44, vector_braid);
  int seq45[] = {-2,-2,-1,-1,-2,-2,1,1,-2,-2,1,1,-2,-2,-2,-2,1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq45, vector_braid);
  int seq46[] = {2,2,1,1,2,2,-1,-1,2,2,-1,-1,2,2,2,2,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq46, vector_braid);
  int seq47[] = {-1,-1,2,2,1,1,1,1,1,1,-2,-2,-1,-1,2,2,-1,-1,-1,-1,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq47, vector_braid);
  int seq48[] = {-2,-2,-1,-1,-1,-1,-1,-1,2,2,-1,-1,-2,-2,1,1,1,1,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq48, vector_braid);
  int seq49[] = {-1,-1,-1,-1,2,2,1,1,1,1,-2,-2,-1,-1,2,2,-1,-1,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq49, vector_braid);
  int seq50[] = {1,1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq50, vector_braid);
  int seq51[] = {-1,-1,2,2,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,-2,-2,1,1,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq51, vector_braid);
  int seq52[] = {-2,-2,1,1,2,2,1,1,-2,-2,1,1,1,1,-2,-2,1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq52, vector_braid);
  int seq53[] = {2,2,-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,2,2,2,2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq53, vector_braid);
  int seq54[] = {-1,-1,2,2,-1,-1,2,2,2,2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq54, vector_braid);
  int seq55[] = {-1,-1,-2,-2,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq55, vector_braid);
  int seq56[] = {1,1,2,2,1,1,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq56, vector_braid);
  int seq57[] = {2,2,1,1,1,1,1,1,-2,-2,1,1,2,2,-1,-1,-1,-1,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq57, vector_braid);
  int seq58[] = {1,1,-2,-2,-1,-1,-1,-1,-1,-1,2,2,1,1,-2,-2,1,1,1,1,1,1,2,2};
  push_back_braid_from_sequence(length, seq58, vector_braid);
  int seq59[] = {-1,-1,2,2,1,1,1,1,-2,-2,1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq59, vector_braid);
  int seq60[] = {1,1,1,1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,1,1,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq60, vector_braid);

  return;
}

void braid_representation_32(vector<Braid>& vector_braid) {
  vector_braid.clear();
  int length = 32;

  int seq1[] = {1,1,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2,1,1,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq1, vector_braid);
  int seq2[] = {-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1,1,1,2,2,2,2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq2, vector_braid);
  int seq3[] = {1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,2,2,1,1,2,2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq3, vector_braid);
  int seq4[] = {1,1,2,2,-1,-1,2,2,2,2,1,1,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq4, vector_braid);
  int seq5[] = {-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,2,2,1,1,2,2,1,1,-2,-2,1,1,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq5, vector_braid);
  int seq6[] = {2,2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq6, vector_braid);
  int seq7[] = {1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq7, vector_braid);
  int seq8[] = {-1,-1,2,2,1,1,1,1,2,2,1,1,1,1,2,2,-1,-1,-2,-2,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq8, vector_braid);
  int seq9[] = {2,2,2,2,2,2,1,1,-2,-2,1,1,2,2,-1,-1,2,2,-1,-1,2,2,1,1,2,2,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq9, vector_braid);
  int seq10[] = {-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,2,2,1,1,1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq10, vector_braid);
  int seq11[] = {1,1,2,2,1,1,1,1,1,1,2,2,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq11, vector_braid);
  int seq12[] = {1,1,2,2,1,1,1,1,1,1,2,2,-1,-1,2,2,2,2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq12, vector_braid);
  int seq13[] = {-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,-1,-1,2,2,1,1,1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq13, vector_braid);
  int seq14[] = {-1,-1,2,2,2,2,-1,-1,2,2,1,1,1,1,-2,-2,1,1,1,1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq14, vector_braid);
  int seq15[] = {1,1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,2,2,1,1,1,1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq15, vector_braid);
  int seq16[] = {-1,-1,-1,-1,2,2,-1,-1,-1,-1,-2,-2,1,1,1,1,-2,-2,1,1,1,1,2,2,-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq16, vector_braid);
  int seq17[] = {1,1,1,1,-2,-2,1,1,1,1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq17, vector_braid);
  int seq18[] = {-1,-1,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq18, vector_braid);
  int seq19[] = {2,2,2,2,1,1,2,2,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq19, vector_braid);
  int seq20[] = {1,1,1,1,2,2,-1,-1,2,2,1,1,-2,-2,1,1,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq20, vector_braid);
  int seq21[] = {1,1,2,2,1,1,1,1,1,1,2,2,-1,-1,2,2,2,2,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq21, vector_braid);
  int seq22[] = {1,1,1,1,2,2,-1,-1,-1,-1,2,2,1,1,1,1,2,2,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq22, vector_braid);
  int seq23[] = {-1,-1,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,2,2,1,1,1,1,2,2,-1,-1,-1,-1,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq23, vector_braid);
  int seq24[] = {-1,-1,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,2,2,-1,-1,-1,-1,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq24, vector_braid);
  int seq25[] = {1,1,1,1,2,2,-1,-1,-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq25, vector_braid);
  int seq26[] = {1,1,2,2,-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,1,1,1,1,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq26, vector_braid);
  int seq27[] = {1,1,2,2,-1,-1,2,2,1,1,2,2,2,2,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq27, vector_braid);
  int seq28[] = {-1,-1,2,2,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq28, vector_braid);
  int seq29[] = {1,1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,2,2,2,2,1,1,2,2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq29, vector_braid);
  int seq30[] = {-2,-2,1,1,2,2,2,2,2,2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq30, vector_braid);
  int seq31[] = {-2,-2,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,2,2,2,2,2,2,2,2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq31, vector_braid);
  int seq32[] = {1,1,1,1,-2,-2,1,1,-2,-2,1,1,1,1,2,2,1,1,1,1,2,2,1,1,2,2,-1,-1,2,2,2,2};
  push_back_braid_from_sequence(length, seq32, vector_braid);
  int seq33[] = {2,2,2,2,-1,-1,2,2,1,1,2,2,1,1,1,1,2,2,1,1,1,1,-2,-2,1,1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq33, vector_braid);
  int seq34[] = {2,2,2,2,-1,-1,2,2,1,1,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq34, vector_braid);
  int seq35[] = {-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,2,2,1,1,-2,-2,1,1,1,1,2,2,-1,-1,2,2,2,2};
  push_back_braid_from_sequence(length, seq35, vector_braid);
  int seq36[] = {1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,-1,-1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq36, vector_braid);
  int seq37[] = {2,2,1,1,2,2,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq37, vector_braid);
  int seq38[] = {2,2,2,2,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq38, vector_braid);
  int seq39[] = {-2,-2,-2,-2,1,1,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,2,2,1,1,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq39, vector_braid);
  int seq40[] = {-2,-2,-1,-1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2,1,1,1,1,1,1,2,2,-1,-1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq40, vector_braid);
  int seq41[] = {1,1,1,1,1,1,1,1,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq41, vector_braid);
  int seq42[] = {1,1,1,1,1,1,1,1,-2,-2,1,1,1,1,-2,-2,1,1,2,2,1,1,-2,-2,1,1,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq42, vector_braid);
  int seq43[] = {1,1,2,2,2,2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq43, vector_braid);
  int seq44[] = {-1,-1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2,-2,-2,1,1,2,2,1,1,2,2,-1,-1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq44, vector_braid);
  int seq45[] = {1,1,-2,-2,1,1,2,2,-1,-1,-1,-1,2,2,1,1,1,1,2,2,2,2,1,1,2,2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq45, vector_braid);
  int seq46[] = {-1,-1,2,2,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq46, vector_braid);
  int seq47[] = {-1,-1,2,2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,2,2,2,2,1,1,-2,-2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq47, vector_braid);
  int seq48[] = {1,1,-2,-2,1,1,-2,-2,1,1,2,2,2,2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq48, vector_braid);
  int seq49[] = {-1,-1,-2,-2,-2,-2,-1,-1,2,2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq49, vector_braid);
  int seq50[] = {1,1,-2,-2,1,1,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,2,2,-1,-1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq50, vector_braid);
  int seq51[] = {1,1,2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq51, vector_braid);
  int seq52[] = {1,1,2,2,-1,-1,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq52, vector_braid);
  int seq53[] = {1,1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq53, vector_braid);
  int seq54[] = {-1,-1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq54, vector_braid);
  int seq55[] = {-1,-1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq55, vector_braid);
  int seq56[] = {1,1,2,2,-1,-1,2,2,1,1,2,2,2,2,1,1,1,1,2,2,-1,-1,-1,-1,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq56, vector_braid);
  int seq57[] = {-1,-1,2,2,-1,-1,2,2,-1,-1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2,2,2,1,1,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq57, vector_braid);
  int seq58[] = {1,1,-2,-2,1,1,1,1,2,2,2,2,-1,-1,2,2,1,1,-2,-2,-2,-2,-1,-1,2,2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq58, vector_braid);
  int seq59[] = {-1,-1,2,2,-1,-1,-2,-2,1,1,2,2,2,2,1,1,1,1,1,1,1,1,-2,-2,1,1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq59, vector_braid);
  int seq60[] = {1,1,2,2,2,2,1,1,-2,-2,1,1,1,1,1,1,1,1,2,2,2,2,1,1,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq60, vector_braid);

  return;
}

void braid_representation_40(vector<Braid>& vector_braid) {
  vector_braid.clear();
  int length = 40;

  int seq1[] = {2,2,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq1, vector_braid);
  int seq2[] = {1,1,1,1,2,2,2,2,-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq2, vector_braid);
  int seq3[] = {-1,-1,-1,-1,-2,-2,-2,-2,1,1,1,1,2,2,1,1,2,2,-1,-1,2,2,1,1,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,2,2};
  push_back_braid_from_sequence(length, seq3, vector_braid);
  int seq4[] = {-2,-2,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq4, vector_braid);
  int seq5[] = {2,2,2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,1,1,1,1,2,2,-1,-1,2,2,1,1,2,2,1,1,1,1,-2,-2,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq5, vector_braid);
  int seq6[] = {-2,-2,1,1,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq6, vector_braid);
  int seq7[] = {-2,-2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,1,1,1,1,2,2,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq7, vector_braid);
  int seq8[] = {2,2,1,1,2,2,2,2,2,2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,2,2,2,2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq8, vector_braid);
  int seq9[] = {2,2,-1,-1,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq9, vector_braid);
  int seq10[] = {-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,2,2,-1,-1,2,2,1,1,1,1,1,1,2,2,-1,-1,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq10, vector_braid);
  int seq11[] = {-1,-1,-1,-1,-1,-1,-1,-1,2,2,1,1,1,1,1,1,2,2,-1,-1,2,2,2,2,2,2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq11, vector_braid);
  int seq12[] = {1,1,2,2,1,1,1,1,1,1,2,2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,1,1,1,1,1,1};
  push_back_braid_from_sequence(length, seq12, vector_braid);
  int seq13[] = {1,1,1,1,1,1,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-2,-2,-1,-1,2,2,1,1,1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq13, vector_braid);
  int seq14[] = {-1,-1,2,2,2,2,1,1,-2,-2,1,1,2,2,1,1,2,2,2,2,1,1,1,1,2,2,1,1,1,1,-2,-2,1,1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq14, vector_braid);
  int seq15[] = {1,1,-2,-2,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq15, vector_braid);
  int seq16[] = {-1,-1,-2,-2,-2,-2,1,1,-2,-2,1,1,1,1,2,2,1,1,1,1,2,2,2,2,1,1,2,2,1,1,-2,-2,1,1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq16, vector_braid);
  int seq17[] = {1,1,2,2,2,2,-1,-1,2,2,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq17, vector_braid);
  int seq18[] = {1,1,1,1,2,2,1,1,2,2,-1,-1,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,1,1,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq18, vector_braid);
  int seq19[] = {1,1,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,2,2,2,2,1,1,2,2,1,1,-2,-2,-1,-1,2,2,2,2,1,1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq19, vector_braid);
  int seq20[] = {-1,-1,2,2,1,1,1,1,2,2,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq20, vector_braid);
  int seq21[] = {-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,2,2,1,1,-2,-2,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,-1,-1,-1,-1,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq21, vector_braid);
  int seq22[] = {1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,-2,-2,-2,-2,-1,-1,2,2,2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq22, vector_braid);
  int seq23[] = {2,2,1,1,2,2,2,2,2,2,-1,-1,-2,-2,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq23, vector_braid);
  int seq24[] = {-1,-1,2,2,1,1,2,2,1,1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,2,2,2,2,1,1,-2,-2,-2,-2,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq24, vector_braid);
  int seq25[] = {-2,-2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,2,2,2,2,1,1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,2,2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq25, vector_braid);
  int seq26[] = {-2,-2,-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,2,2,-1,-1,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq26, vector_braid);
  int seq27[] = {2,2,2,2,2,2,1,1,-2,-2,-1,-1,-2,-2,1,1,2,2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq27, vector_braid);
  int seq28[] = {2,2,2,2,1,1,1,1,2,2,1,1,2,2,2,2,1,1,2,2,2,2,2,2,-1,-1,2,2,-1,-1,2,2,2,2,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq28, vector_braid);
  int seq29[] = {-2,-2,-2,-2,-2,-2,-1,-1,2,2,1,1,2,2,-1,-1,-2,-2,1,1,1,1,1,1,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq29, vector_braid);
  int seq30[] = {-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,-2,-2,1,1,-2,-2,-2,-2,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq30, vector_braid);
  int seq31[] = {-2,-2,1,1,-2,-2,1,1,-2,-2,-2,-2,1,1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,1,1,-2,-2,1,1,1,1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq31, vector_braid);
  int seq32[] = {2,2,2,2,1,1,-2,-2,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,-1,-1,2,2,2,2,1,1,2,2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq32, vector_braid);
  int seq33[] = {1,1,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,-2,-2,-1,-1,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq33, vector_braid);
  int seq34[] = {1,1,1,1,2,2,-1,-1,2,2,1,1,1,1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,1,1,2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq34, vector_braid);
  int seq35[] = {-1,-1,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq35, vector_braid);
  int seq36[] = {1,1,1,1,2,2,1,1,-2,-2,1,1,-2,-2,1,1,2,2,-1,-1,-2,-2,1,1,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq36, vector_braid);
  int seq37[] = {-1,-1,2,2,1,1,2,2,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,2,2,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq37, vector_braid);
  int seq38[] = {1,1,1,1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,1,1,1,1,-2,-2,1,1,2,2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq38, vector_braid);
  int seq39[] = {-1,-1,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-2,-2,1,1,2,2,1,1,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq39, vector_braid);
  int seq40[] = {1,1,-2,-2,-1,-1,-2,-2,-2,-2,1,1,2,2,2,2,1,1,1,1,1,1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,1,1,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq40, vector_braid);
  int seq41[] = {-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2,1,1,1,1,1,1};
  push_back_braid_from_sequence(length, seq41, vector_braid);
  int seq42[] = {-2,-2,-2,-2,1,1,2,2,1,1,1,1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq42, vector_braid);
  int seq43[] = {-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq43, vector_braid);
  int seq44[] = {-2,-2,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,2,2,1,1,2,2,-1,-1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq44, vector_braid);
  int seq45[] = {1,1,-2,-2,-2,-2,-1,-1,2,2,2,2,1,1,1,1,2,2,2,2,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq45, vector_braid);
  int seq46[] = {-1,-1,2,2,2,2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,-2,-2,1,1,1,1,-2,-2,-2,-2,1,1,2,2};
  push_back_braid_from_sequence(length, seq46, vector_braid);
  int seq47[] = {-1,-1,2,2,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,-1,-1,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq47, vector_braid);
  int seq48[] = {-2,-2,1,1,2,2,-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,1,1,-2,-2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq48, vector_braid);
  int seq49[] = {-1,-1,2,2,-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2,2,2,1,1,2,2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq49, vector_braid);
  int seq50[] = {1,1,-2,-2,-1,-1,2,2,1,1,2,2,2,2,-1,-1,2,2,1,1,-2,-2,-2,-2,-1,-1,2,2,1,1,1,1,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq50, vector_braid);
  int seq51[] = {1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq51, vector_braid);
  int seq52[] = {2,2,2,2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq52, vector_braid);
  int seq53[] = {2,2,1,1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1,2,2,-1,-1,2,2,-1,-1,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq53, vector_braid);
  int seq54[] = {2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq54, vector_braid);
  int seq55[] = {2,2,1,1,-2,-2,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq55, vector_braid);
  int seq56[] = {-2,-2,-1,-1,2,2,2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,1,1,2,2,2,2,1,1,1,1,2,2,2,2,-1,-1,-2,-2,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq56, vector_braid);
  int seq57[] = {2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,1,1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq57, vector_braid);
  int seq58[] = {1,1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq58, vector_braid);
  int seq59[] = {-1,-1,2,2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,2,2,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq59, vector_braid);
  int seq60[] = {1,1,-2,-2,1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq60, vector_braid);

  return;
}


void braid_representation_44(vector<Braid>& vector_braid) {
  vector_braid.clear();
  int length = 44;

  int seq1[] = {2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1,2,2,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq1, vector_braid);
  int seq2[] = {1,1,-2,-2,1,1,-2,-2,-2,-2,1,1,1,1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq2, vector_braid);
  int seq3[] = {-1,-1,2,2,-1,-1,2,2,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq3, vector_braid);
  int seq4[] = {2,2,-1,-1,-1,-1,2,2,1,1,-2,-2,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,1,1,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq4, vector_braid);
  int seq5[] = {-2,-2,1,1,1,1,-2,-2,-1,-1,2,2,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,-1,-1,2,2,2,2,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq5, vector_braid);
  int seq6[] = {-1,-1,-2,-2,-2,-2,1,1,1,1,2,2,-1,-1,2,2,2,2,1,1,2,2,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,2,2,2,2,-1,-1,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq6, vector_braid);
  int seq7[] = {-2,-2,1,1,1,1,1,1,1,1,2,2,2,2,1,1,1,1,1,1,2,2,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq7, vector_braid);
  int seq8[] = {2,2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,2,2,-1,-1,2,2,2,2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq8, vector_braid);
  int seq9[] = {1,1,2,2,2,2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,1,1,2,2,1,1,1,1,-2,-2,-2,-2,1,1,2,2,1,1,2,2};
  push_back_braid_from_sequence(length, seq9, vector_braid);
  int seq10[] = {-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-2,-2,1,1,1,1,1,1,1,1,2,2,1,1,1,1,1,1,2,2,-1,-1,-1,-1,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq10, vector_braid);
  int seq11[] = {-1,-1,-1,-1,-1,-1,-1,-1,2,2,1,1,1,1,1,1,2,2,1,1,1,1,1,1,1,1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq11, vector_braid);
  int seq12[] = {1,1,2,2,1,1,1,1,1,1,2,2,-1,-1,2,2,2,2,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,1,1,1,1,1,1,1,1};
  push_back_braid_from_sequence(length, seq12, vector_braid);
  int seq13[] = {1,1,1,1,1,1,1,1,-2,-2,-1,-1,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-1,-1,-1,-1,2,2,2,2,-1,-1,2,2,1,1,1,1,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq13, vector_braid);
  int seq14[] = {-1,-1,-1,-1,-2,-2,1,1,2,2,1,1,1,1,-2,-2,1,1,1,1,2,2,2,2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-2,-2,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq14, vector_braid);
  int seq15[] = {1,1,1,1,2,2,-1,-1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,2,2,1,1,1,1,-2,-2,1,1,2,2,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq15, vector_braid);
  int seq16[] = {-2,-2,1,1,-2,-2,-2,-2,-1,-1,2,2,-1,-1,-1,-1,-2,-2,-1,-1,2,2,2,2,1,1,1,1,-2,-2,1,1,1,1,2,2,1,1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq16, vector_braid);
  int seq17[] = {2,2,-1,-1,2,2,2,2,1,1,-2,-2,1,1,1,1,2,2,1,1,-2,-2,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-2,-2,-1,-1,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq17, vector_braid);
  int seq18[] = {2,2,1,1,1,1,-2,-2,-2,-2,1,1,2,2,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,-2,-2};
  push_back_braid_from_sequence(length, seq18, vector_braid);
  int seq19[] = {1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,1,1,-2,-2,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq19, vector_braid);
  int seq20[] = {-1,-1,2,2,-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,2,2,2,2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq20, vector_braid);
  int seq21[] = {-2,-2,-1,-1,-1,-1,2,2,2,2,-1,-1,-2,-2,-2,-2,1,1,2,2,2,2,1,1,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,1,1,2,2,2,2};
  push_back_braid_from_sequence(length, seq21, vector_braid);
  int seq22[] = {1,1,2,2,2,2,1,1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,-2,-2,1,1,2,2,1,1,-2,-2,-1,-1,2,2,1,1,2,2,1,1,1,1};
  push_back_braid_from_sequence(length, seq22, vector_braid);
  int seq23[] = {1,1,1,1,2,2,1,1,2,2,-1,-1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,1,1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq23, vector_braid);
  int seq24[] = {-1,-1,-2,-2,-2,-2,-1,-1,2,2,1,1,-2,-2,1,1,2,2,1,1,2,2,2,2,-1,-1,-2,-2,-1,-1,2,2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq24, vector_braid);
  int seq25[] = {-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,1,1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,1,1,2,2,1,1,-2,-2,1,1,2,2,-1,-1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq25, vector_braid);
  int seq26[] = {-2,-2,1,1,1,1,1,1,1,1,2,2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,-2,-2,1,1,2,2,1,1,2,2,-1,-1,2,2,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq26, vector_braid);
  int seq27[] = {-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,2,2,1,1,1,1,-2,-2,-2,-2,1,1,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq27, vector_braid);
  int seq28[] = {-2,-2,-1,-1,-1,-1,-2,-2,1,1,1,1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,2,2,-1,-1,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq28, vector_braid);
  int seq29[] = {-2,-2,1,1,1,1,-2,-2,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq29, vector_braid);
  int seq30[] = {2,2,1,1,1,1,2,2,-1,-1,-1,-1,2,2,2,2,1,1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,2,2,1,1,-2,-2,1,1,1,1,2,2};
  push_back_braid_from_sequence(length, seq30, vector_braid);
  int seq31[] = {-2,-2,-2,-2,1,1,1,1,-2,-2,1,1,2,2,-1,-1,-2,-2,-2,-2,-2,-2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,-2,-2,-2,-2,1,1,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq31, vector_braid);
  int seq32[] = {1,1,-2,-2,-1,-1,2,2,-1,-1,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,1,1,1,1,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq32, vector_braid);
  int seq33[] = {-1,-1,2,2,1,1,-2,-2,1,1,-2,-2,1,1,2,2,2,2,1,1,1,1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,-1,-1,-1,-1,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq33, vector_braid);
  int seq34[] = {2,2,-1,-1,-2,-2,1,1,2,2,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,1,1,-2,-2,1,1,1,1,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2};
  push_back_braid_from_sequence(length, seq34, vector_braid);
  int seq35[] = {-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,1,1,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,2,2,1,1,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq35, vector_braid);
  int seq36[] = {2,2,2,2,1,1,1,1,2,2,1,1,-2,-2,1,1,2,2,1,1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq36, vector_braid);
  int seq37[] = {-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,1,1,1,1,-2,-2,1,1,-2,-2,-1,-1,2,2,2,2,1,1,2,2,2,2,-1,-1,-1,-1,-1,-1,2,2,1,1};
  push_back_braid_from_sequence(length, seq37, vector_braid);
  int seq38[] = {-2,-2,-2,-2,1,1,-2,-2,-2,-2,-1,-1,2,2,-1,-1,-2,-2,1,1,1,1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq38, vector_braid);
  int seq39[] = {2,2,2,2,-1,-1,2,2,2,2,1,1,-2,-2,1,1,2,2,-1,-1,-1,-1,2,2,2,2,1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq39, vector_braid);
  int seq40[] = {2,2,1,1,1,1,-2,-2,-1,-1,2,2,-1,-1,-1,-1,2,2,-1,-1,2,2,1,1,-2,-2,-2,-2,-1,-1,-2,-2,-2,-2,1,1,1,1,1,1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq40, vector_braid);
  int seq41[] = {-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,1,1,2,2,2,2,-1,-1,2,2,-1,-1,-2,-2,-2,-2,-1,-1,2,2,2,2,-1,-1,-2,-2,1,1,1,1};
  push_back_braid_from_sequence(length, seq41, vector_braid);
  int seq42[] = {-1,-1,2,2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-1,-1,2,2,-1,-1,2,2,1,1,2,2,-1,-1,2,2,2,2,1,1};
  push_back_braid_from_sequence(length, seq42, vector_braid);
  int seq43[] = {1,1,-2,-2,1,1,2,2,2,2,1,1,2,2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,1,1,2,2,1,1,-2,-2,-2,-2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq43, vector_braid);
  int seq44[] = {2,2,-1,-1,-2,-2,-2,-2,1,1,2,2,1,1,-2,-2,1,1,-2,-2,-1,-1,-1,-1,2,2,-1,-1,-1,-1,2,2,1,1,2,2,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq44, vector_braid);
  int seq45[] = {1,1,2,2,1,1,-2,-2,1,1,2,2,1,1,2,2,-1,-1,2,2,2,2,-1,-1,-2,-2,-2,-2,1,1,1,1,1,1,-2,-2,1,1,-2,-2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq45, vector_braid);
  int seq46[] = {-1,-1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,1,1,2,2,2,2,-1,-1,-1,-1,-1,-1,2,2,-1,-1,2,2,-1,-1,2,2};
  push_back_braid_from_sequence(length, seq46, vector_braid);
  int seq47[] = {-1,-1,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,-1,-1,-1,-1,-2,-2,1,1,2,2,1,1,2,2,-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq47, vector_braid);
  int seq48[] = {1,1,-2,-2,-1,-1,2,2,1,1,1,1,-2,-2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,-1,-1,-1,-1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq48, vector_braid);
  int seq49[] = {-1,-1,2,2,-1,-1,-2,-2,1,1,1,1,2,2,-1,-1,2,2,2,2,2,2,1,1,2,2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,2,2,-1,-1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq49, vector_braid);
  int seq50[] = {1,1,-2,-2,-1,-1,2,2,1,1,-2,-2,-2,-2,-2,-2,-1,-1,2,2,1,1,2,2,2,2,2,2,-1,-1,2,2,1,1,1,1,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq50, vector_braid);
  int seq51[] = {2,2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,1,1,-2,-2,1,1,2,2,2,2,1,1,-2,-2,-2,-2,1,1,2,2,-1,-1,-1,-1};
  push_back_braid_from_sequence(length, seq51, vector_braid);
  int seq52[] = {1,1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,1,1,-2,-2,1,1,-2,-2,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq52, vector_braid);
  int seq53[] = {-2,-2,1,1,2,2,2,2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,2,2,1,1,1,1,-2,-2,1,1,1,1,-2,-2,-1,-1,-2,-2,-2,-2,-1,-1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq53, vector_braid);
  int seq54[] = {-1,-1,2,2,-1,-1,-2,-2,-2,-2,-1,-1,-2,-2,1,1,1,1,-2,-2,1,1,1,1,2,2,-1,-1,2,2,-1,-1,-2,-2,-1,-1,2,2,2,2,1,1,-2,-2};
  push_back_braid_from_sequence(length, seq54, vector_braid);
  int seq55[] = {2,2,-1,-1,2,2,-1,-1,2,2,-1,-1,-1,-1,-1,-1,2,2,2,2,1,1,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,-1,-1,-2,-2,-1,-1};
  push_back_braid_from_sequence(length, seq55, vector_braid);
  int seq56[] = {-2,-2,1,1,-2,-2,1,1,-2,-2,1,1,1,1,1,1,-2,-2,-2,-2,-1,-1,2,2,2,2,-1,-1,2,2,1,1,2,2,1,1,-2,-2,1,1,2,2,1,1};
  push_back_braid_from_sequence(length, seq56, vector_braid);
  int seq57[] = {-1,-1,2,2,1,1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,-1,-1,-2,-2,-1,-1,2,2,1,1,1,1,2,2,1,1,2,2,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq57, vector_braid);
  int seq58[] = {1,1,-2,-2,1,1,2,2,2,2,1,1,2,2,1,1,1,1,2,2,-1,-1,-2,-2,-1,-1,-2,-2,1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq58, vector_braid);
  int seq59[] = {-1,-1,2,2,1,1,-2,-2,-1,-1,2,2,2,2,2,2,1,1,-2,-2,-1,-1,-2,-2,-2,-2,-2,-2,1,1,-2,-2,-1,-1,-1,-1,2,2,1,1,-2,-2,1,1};
  push_back_braid_from_sequence(length, seq59, vector_braid);
  int seq60[] = {1,1,-2,-2,1,1,2,2,-1,-1,-1,-1,-2,-2,1,1,-2,-2,-2,-2,-2,-2,-1,-1,-2,-2,1,1,2,2,2,2,2,2,-1,-1,-2,-2,1,1,2,2,-1,-1};
  push_back_braid_from_sequence(length, seq60, vector_braid);

  return;
}
