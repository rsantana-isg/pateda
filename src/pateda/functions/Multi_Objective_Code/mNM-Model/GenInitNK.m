

% Global variables required for the computation of the MNK landscape
global Tables
global PE
global AllVectors

% Parameters of the model

M = 2;  % Number of objectives
n = 12; % Number of variables
k = 3;  % Number of neighbors
instance = 1; % instance  % Instance used to read Table parameters from files
nsteps = 30;


% The initial landscape is created
[AllVectors,ListOfLists,fvals] = CreateLandscape(M,n,k,instance);



% Evolution of the landscape to maximize size of the Pareto set
[ListFactors,TheMax] = EvolvePF_Size(M,ListOfLists,nsteps)

% Evolution of the landscape to maximize correlations
[ListFactors,TheMax] = EvolveCorrelations(M,ListOfLists,nsteps)


for l=1:M,                               
  LF = ListOfLists{l};
  org_fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
end

for l=1:M,                               
  LF = ListFactors{l};
  fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
end

% The Pareto front is computed
[Index]=FindParetoSet(fvals,fvals);
fvals(Index,:)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
plot(fvals(:,1),fvals(:,2),'.b')
hold on
plot(fvals(Index,1),fvals(Index,2),'r*')

figure
plot(fvals(:,2),org_fvals(:,2),'.b')
hold on
plot(fvals(Index,2),org_fvals(Index,2),'r*')




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear ThePairs
clear TheMax
clear TheVals

ListFactors = ListFactors2;


for t=1:2^n;       
  fvals(t,2) = EvaluateNK(AllVectors(t,:));
end  
[Index]=FindParetoSet(fvals,fvals);
fvals(Index,:)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
plot(fvals(:,1),fvals(:,2),'.b')
hold on
plot(fvals(Index,1),fvals(Index,2),'r*')



