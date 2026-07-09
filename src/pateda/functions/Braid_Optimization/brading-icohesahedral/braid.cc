/*****************************************************************************
*
* TQC Project: Topological Quantum Compiling/Computation
*
* braid.cc: source file for the braiding library
*
* Copyright (C) 2009 by Xin Wan <xinwan@zimp.zju.edu.cn>
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

#include <iostream>     // std::cout
#include <algorithm>    // std::reverse_copy
#include <vector>       // std::vector

#include "braid.h"

UMatrix& UMatrix::set_axis_angle(const double x, const double y, const double z, const double phi) {
  double norm = sqrt(x * x + y* y + z* z);
  assert(norm != 0.0);
  a = complex<double>(cos(phi / 2.0), -z / norm * sin(phi / 2.0));
  b = complex<double>(-y / norm * sin(phi / 2.0), -x / norm * sin(phi / 2.0));
  theta = 0.0;
  return *this;
}

void UMatrix::matrixElements(complex<double>* m) const {
  complex<double> phase(cos(theta), sin(theta));
  m[0] = a * phase;
  m[1] = b * phase;
  m[2] = -conj(b) * phase;
  m[3] = conj(a) * phase;
  return;
}

double UMatrix::quasidistance(const UMatrix& u){
  double d = abs((a - u.a).real()) + abs((a - u.a).imag()) + abs((b - u.b).real()) + abs((b - u.b).imag());
  double dn = abs((a + u.a).real()) + abs((a + u.a).imag()) + abs((b + u.b).real()) + abs((b + u.b).imag());
  if (d > dn) d = dn;
  return d;
}

double UMatrix::distance(const UMatrix& u){
  double d = sqrt(norm(a - u.a) + norm(b - u.b));
  double dn = sqrt(norm(a + u.a) + norm(b + u.b));
  if (d > dn) d = dn;
  return d;
}

ostream& operator<<(ostream& ostr, const UMatrix& u) { 
  complex<double> phase(cos(u.theta), sin(u.theta));
  return (ostr << "\n          " << u.a * phase << "  " << u.b * phase << "\n          " << -conj(u.b) * phase << "  " << conj(u.a) * phase << endl); 
}

double TAU = 0.5 * (sqrt(5.0) - 1.0);
complex<double> sigma1_a(cos(0.7 * M_PI), -sin(0.7 * M_PI));
complex<double> sigma1_b(0, 0);
complex<double> sigma2_a(-TAU * cos(0.1 * M_PI), TAU * sin(0.1 * M_PI));
complex<double> sigma2_b(0, -sqrt(TAU));

const UMatrix Braid::sigma[4] = { UMatrix(sigma1_a, sigma1_b, -0.1 * M_PI), UMatrix(sigma2_a, sigma2_b, -0.1 * M_PI), UMatrix(conj(sigma2_a), -sigma2_b, 0.1 * M_PI) , UMatrix(conj(sigma1_a), sigma1_b, 0.1 * M_PI)};

// Initialize with a sequence of fundamental braids 
// labeled by {1,2,3,4} or {1,2,-2,-1} (stored as 0,1,2,3)
Braid::Braid(const int bl, const int* bs) {
  for (int i = 0; i < bl; i++) {
    if (bs[i] < 0)
      s.push_back(4 + bs[i]); // {-2, -1} -> {2, 3}
    else
      s.push_back(bs[i] - 1);
    u = u * sigma[s.back()]; // {1, 2} -> {0, 1}
  }
}

Braid& Braid::operator+=(const Braid& b) { // needs improvement to take care of the cancellation
  vector<int>::const_iterator it = b.s.begin();
  s.insert(s.end(), it, b.s.end());
  u = u * b.u;
  return *this;
}

Braid Braid::operator~() {
  vector<int> rs(s.size());
  reverse_copy(s.begin(), s.end(), rs.begin());
  for (int i = 0; i < rs.size(); i++) {
    rs[i] = 3 - rs[i]; // braid (3-i) is the inverse of braid i
  }
  return Braid(rs, ~u);
}

Braid& Braid::simplify() { // taking care of cancellation & repetition
  if (s.size() < 2) return *this;
  int info = 1;
  vector<int>::iterator it;
  while (info > 0) {
    info = 0;
    for (it = s.begin(); it < s.end() - 1; ++it) 
      if (*it == 3 - (*(it+1))) {
	it = s.erase(it);
	it = s.erase(it);
	it--;
	info++;
      }
    for (it = s.begin(); it < s.end() - 5; ++it) {
      int e = *it;
      if (e == *(it+1)) 
	if (e == *(it+2)) 
	  if (e == *(it+3)) 
	    if (e == *(it+4)) 
	      if (e == *(it+5)) {
		it = s.erase(it);
		it = s.erase(it);
		it = s.erase(it);
		it = s.erase(it);
		it = s.erase(it);
		it = s.erase(it);
		info++;
		s.insert(it, 4, 3 - e);
	      }
    }
  }
  return *this;
}

ostream& operator<<(ostream& ostr, const Braid& b){
  cout << "Braid sequence (L = " << b.s.size() << "):\n{";
  for (int i = 0; i < b.s.size(); i++) {
    if (b.s[i] < 2) 
      cout << b.s[i] + 1; // {0, 1} -> {1, 2}
    else 
      cout << b.s[i] - 4; // {2, 3} -> {-2, -1}
    if (i < b.s.size() - 1) 
      cout << ",";
    else 
      cout << "}";
  }
  return ostr << "\nMatrix representation:" << b.u;
}

