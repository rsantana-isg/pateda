#/bin/env python3

import argparse
import random
import aifc
import csv
import scipy
import struct
import sys
import pylab as pl
import numpy as np
from scipy import interp
from scipy.io import loadmat
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=range(25),
         nargs='+', help='an integer in the range 0..25')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum,
        default=max, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    tfeatures=args.integers[0]



aux_data = loadmat('results_MNK_toplot.mat', squeeze_me=True) 
all_negcorrs  = aux_data['all_negcorrs']  
all_poscorrs  = aux_data['all_poscorrs']
all_posPF  = aux_data['all_posPF']

aux_data = loadmat('PFFirst_Last.mat', squeeze_me=True) 
fvals  = aux_data['fvals']  
Index  = aux_data['Index']
org_fvals = aux_data['org_fvals']
org_Index  = aux_data['org_Index']
  

aux_data = loadmat('ApproxMinCorrn50.mat', squeeze_me=True) 
all_approx_negcorrs  = aux_data['all_approx_negcorrs']  


aux_data = loadmat('ApproxResultsNPF-PF.mat', squeeze_me=True) 
all_PF_obj2  = aux_data['all_PF_obj2']  
all_PF_obj3  = aux_data['all_PF_obj3']  
all_NPF_obj2  = aux_data['all_NPF_obj2']  
all_NPF_obj3  = aux_data['all_NPF_obj3']  

markers = [] 
styles = markers + [
    'o',
    's',
    'p',
    'd',
    'h',
    'v',
    'x',
    '<'     
]

colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k','b')
fa =18

a = 0
for i in range(2):
 for j in range(4):
  pl.plot(np.mean(all_negcorrs[i,j],0),marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Correlation',fontsize=fa)
pl.legend(shadow=True, fancybox=True)
pl.tick_params(axis='both', labelsize=14)
pl.savefig("ResMNKExactMinCorr.eps")
pl.show()
   
a = 0
for i in range(2):
 for j in range(4):
  pl.plot(np.mean(all_poscorrs[i,j],0),marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Correlation',fontsize=fa)
pl.legend(shadow=True, fancybox=True,loc=4)
pl.tick_params(axis='both', labelsize=14)
pl.savefig("ResMNKExactMaxCorr.eps")
pl.show()
 

a = 0
for i in range(2):
 for j in range(4):
  pl.plot(np.mean(all_posPF[i,j],0),marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Size of the Pareto front',fontsize=fa)
pl.legend(shadow=True, fancybox=True,loc=4)
pl.tick_params(axis='both', labelsize=14)
pl.savefig("ResMNKExactMaxPF.eps")
pl.show()


#pl.scatter(fvals[:,0],fvals[:,1],s=40,marker='*', color='b')
#pl.scatter(fvals[Index-1,0],fvals[Index-1,1],s=42,marker='o', color='r')
#pl.xlabel('$f_1$',fontsize=fa)
#pl.ylabel('$f_2$',fontsize=fa)
#pl.tick_params(axis='both', labelsize=14)
#pl.savefig("Last_n12_k_1pos_corr.eps")
#pl.show()
 

#pl.scatter(org_fvals[:,0],org_fvals[:,1],s=40,marker='*', color='b')
#pl.scatter(org_fvals[org_Index-1,0],org_fvals[org_Index-1,1],s=42,marker='o', color='r')
#pl.xlabel('$f_1$',fontsize=fa)
#pl.ylabel('$f_2$',fontsize=fa)
#pl.tick_params(axis='both', labelsize=14)
#pl.savefig("First_n12_k_1pos_corr.eps")
#pl.show()


#a = 0
#for j in range(4):
#  pl.plot(np.mean(all_approx_negcorrs[j],0),marker=styles[a], color=colors[a], markersize=8,label="k={}".format(j+1), lw=2)
#  a = a + 1
#pl.xlabel('Iterations',fontsize=fa)
#pl.ylabel('Correlation',fontsize=fa)
#pl.legend(shadow=True, fancybox=True)
#pl.tick_params(axis='both', labelsize=14)
#pl.savefig("ResMNKApproxMinCorr.eps")
#pl.show()
   

a = 0
for i in range(2):
 for j in range(4):
  pl.plot(range(0, 1500, 25),np.mean(all_PF_obj2[i,j],0)[0:1500:25],marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Size of the Pareto front',fontsize=fa)
pl.legend(shadow=True, fancybox=True,loc=4)
pl.tick_params(axis='both', labelsize=14)
#pl.xticks(range(0,1500,75))
ticks = range(0, 1501,150)
labels = range(0,1501,150)
pl.xticks(ticks, labels)
pl.savefig("ResMNKObj2ApproxMaxPF.eps")
pl.show()


a = 0
for i in range(2):
 for j in range(4):
  pl.plot(range(0, 1500, 25),np.mean(all_PF_obj3[i,j],0)[0:1500:25],marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Size of the Pareto front',fontsize=fa)
pl.legend(shadow=True, fancybox=True,loc=4)
pl.tick_params(axis='both', labelsize=14)
ticks = range(0, 1501,150)
labels = range(0,1501,150)
pl.savefig("ResMNKObj3ApproxMaxPF.eps")
pl.show()


a = 0
for i in range(2):
 for j in range(4):
  pl.plot(range(0, 1500, 25),np.mean(all_NPF_obj2[i,j],0)[0:1500:25],marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Number of fronts',fontsize=fa)
pl.legend(shadow=True, fancybox=True,loc=1)
pl.tick_params(axis='both', labelsize=14)
ticks = range(0, 1501,150)
labels = range(0,1501,150)
pl.savefig("ResMNKObj2ApproxMinNPF.eps")
pl.show()


a = 0
for i in range(2):
 for j in range(4):
  pl.plot(range(0, 1500, 25),np.mean(all_NPF_obj3[i,j],0)[0:1500:25],marker=styles[a], color=colors[a], markersize=8,label="n={},k={}".format(i*2+10,j+1), lw=2)
  a = a + 1
  pl.xlabel('Iterations',fontsize=fa)
  pl.ylabel('Number of fronts',fontsize=fa)
pl.legend(shadow=True, fancybox=True,loc=1)
pl.tick_params(axis='both', labelsize=14)
ticks = range(0, 1501,150)
labels = range(0,1501,150)
pl.savefig("ResMNKObj3ApproxMinNPF.eps")
pl.show()
