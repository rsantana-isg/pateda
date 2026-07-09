#!/usr/bin/env python

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

from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model

#from sklearn.ensemble import AdaBoostClassifier

def ROCAnalysis(myclf,X,y):
# Run myclf with crossvalidation and plot ROC curves
    cv = StratifiedKFold(y, k=5)
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    res = y*1.0

    for i, (train, test) in enumerate(cv):
        probas_ = myclf.fit(X[train], y[train]).predict_proba(X[test])
        res[test] = probas_[:,1]
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        print i,  roc_auc 
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    print 'final ',  mean_auc
    #pl.plot(mean_fpr, mean_tpr, 'k--',
    #        label='Mean ROC (area = %0.2f)' % mean_auc, lw=2)

    #pl.xlim([-0.05, 1.05])
    #pl.ylim([-0.05, 1.05])
    #pl.xlabel('False Positive Rate')
    #pl.ylabel('True Positive Rate')
    #pl.title('Receiver operating characteristic example')
    #pl.legend(loc="lower right")
    #pl.show()
    return mean_auc, res




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'integers', metavar='int', type=int, choices=xrange(25),
         nargs='+', help='an integer in the range 0..25')
    parser.add_argument(
        '--sum', dest='accumulate', action='store_const', const=sum,
        default=max, help='sum the integers (default: find the max)')

    args = parser.parse_args()
    tfeatures=args.integers[0]

mylist = []
with open('data/train.csv', 'rb') as csvfile:
     spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
     a = 0
     for row in spamreader:
         if a>0: 
           mylist.append(int(row[1]))
         a = a + 1
y = np.array(mylist)   

a = 1
X = [] 
selvarindices = np.loadtxt('ThreeSelVars.csv',delimiter=',',unpack=True, dtype=int)
selvarindices = selvarindices.transpose()

for f in range(1,11):
      print f
      traindatavector = np.loadtxt('trainBestfftfeatures%i.csv' % (f),delimiter=',',unpack=True)
      traindatavector1 = np.loadtxt('NtrainTMfeatures%i.csv' % (f),delimiter=' ',unpack=True)
      traindatavector =  np.vstack((traindatavector,traindatavector1))
      traindatavector1 = np.loadtxt('trainSUMfeatures%i.csv' % (f),delimiter=',',unpack=True)
      traindatavector =  np.vstack((traindatavector,traindatavector1))
      traindatavector1 = np.loadtxt('trainPredfeatures%i.csv' % (f),delimiter=',',unpack=True)
      traindatavector =  np.vstack((traindatavector,traindatavector1))

     
      if f==1:
         X =  traindatavector 
      else:
         X =  np.hstack((X,traindatavector)) 
X = X.transpose()
X = X[:,selvarindices]

Y = []    
for t in range(1,12):
      testdatavector = np.loadtxt('testBestfftfeatures%i.csv' % (t),delimiter=',',unpack=True)     
      testdatavector1 = np.loadtxt('NtestTMfeatures%i.csv' % (t),delimiter=' ',unpack=True)
      testdatavector =  np.vstack((testdatavector,testdatavector1))
      testdatavector1 = np.loadtxt('testSUMfeatures%i.csv' % (t),delimiter=',',unpack=True)
      testdatavector =  np.vstack((testdatavector,testdatavector1))
      testdatavector1 = np.loadtxt('testPredfeatures%i.csv' % (t),delimiter=',',unpack=True)
      testdatavector =  np.vstack((testdatavector,testdatavector1))

      if t==1:
         Y = testdatavector
      else:     
         Y =  np.hstack((Y,testdatavector)) 
Y = Y.transpose() 
Y = Y[:,selvarindices]     
print Y.shape
#cfile = csv.writer(open(str(outputfilename), "wb"), delimiter=' ', lineterminator='\n') 
#cfile.writerow(FinalAnswer)  


n_features = X.shape[1]
C = 1.0
importance = range(1,10)

n_classifiers = 8
index = 0
#for index in [1,2,3,4,6]:
#for index in [11,12,13,14]:
for index in args.integers[1:]:
   if index==1:      
     clf=LogisticRegression(C=C, penalty='l1')
   if index==2:  
     clf=LogisticRegression(C=C, penalty='l2')
   if index==3:  
     clf=QDA()
   if index==4:
     clf=LDA()             
   if index==5:  
     clf=KNeighborsClassifier(warn_on_equidistant=False)
   if index==6:  
     clf=SVC(kernel='linear', C=C, probability=True, tol=1e-3, verbose=False)        
   if index==7:
     clf = svm.SVC(kernel='poly', degree=3, C=C, probability=True, tol=1e-3, verbose=False)
   if index==8:
     clf = svm.SVC(kernel='rbf', C=C, probability=True, tol=1e-3, verbose=False)
   if index==9:
     clf = GaussianNB() 
   if index==10:
     clf = GradientBoostingClassifier(n_estimators=100, max_depth=11, subsample=1.0)    
   if index==11:
     clf = RandomForestClassifier(max_depth=11, n_estimators=100,compute_importances=True)    
   if index==12:
     clf = DecisionTreeClassifier(max_depth=None, min_samples_split=1,random_state=0, compute_importances=True)     
   if index==13:
     clf = ExtraTreesClassifier(n_estimators=100,random_state=0,compute_importances=True)     
   if index==14:
     clf = DecisionTreeClassifier(max_depth=5, compute_importances=True)   
   if index==15:
     clf = RandomForestClassifier(max_depth=5, n_estimators=500, max_features=1, compute_importances=True)     
   if index==16:
     clf = GradientBoostingClassifier(n_estimators=500, max_depth=11)     
   if index==17:
     clf = linear_model.SGDClassifier(loss='log')
   if index==18:  
     clf=KNeighborsClassifier(n_neighbors=19,warn_on_equidistant=False)
   if index==19:
     clf = GradientBoostingClassifier(n_estimators=500, max_depth=3)   
   if index==20:
     clf = GradientBoostingClassifier(n_estimators=500, max_depth=5)   
   if index==21:
     clf = RandomForestClassifier(max_depth=11, n_estimators=500,compute_importances=True)    

   print tfeatures, index, clf

   if( (index>5) & (index<9) ): 
     conXY = np.vstack([X,Y]);
     print conXY.shape
     normX = (X-np.min(conXY))/(np.max(conXY)-np.min(conXY))
     normY = (Y-np.min(conXY))/(np.max(conXY)-np.min(conXY))
     X = normX
     Y = normY
   meanauc, train_probas = ROCAnalysis(clf,X,y) 
   l = ((train_probas>0.5)==y)
   auxY = (1.0*sum(l))/30000
   print 'Classification rate ', auxY

   clf.fit(X,y)
   probas = clf.predict_proba(Y)
   if( (index>10) & (index<17) ):
      print index
      importance=clf.feature_importances_
  
   outputfilename = 'SelThreePredTrainFeat%i_%i.csv' % (tfeatures,index) 
   cfile = csv.writer(open(str(outputfilename), "wb"), delimiter=' ', lineterminator='\n') 
   cfile.writerow(train_probas.transpose())
  
   outputfilename = 'SelThreePredTestFeat%i_%i.csv' % (tfeatures,index) 
   cfile = csv.writer(open(str(outputfilename), "wb"), delimiter=' ', lineterminator='\n') 
   cfile.writerow(probas[:,1].transpose())
  
   outputfilename = 'SelThreeImpTestFeat%i_%i.csv' % (tfeatures,index) 
   cfile = csv.writer(open(str(outputfilename), "wb"), delimiter=' ', lineterminator='\n') 
   cfile.writerow(importance)
