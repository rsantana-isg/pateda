

% Global variables required for the computation of the MNK landscape
global Tables
global PE
global AllVectors

% Parameters of the model

M = 2;  % Number of objectives
n = 14; % Number of variables
k = 3;  % Number of neighbors
instance = 1; % instance  % Instance used to read Table parameters from files
nsteps = 25;
nruns = 30;
% AllNKInstances(12,4,5,'cfiles')



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maximization of the correlations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for k=1:4,
    % The initial landscape is created
    [AllVectors,ListOfLists,fvals] = CreateLandscape(M,n,k,instance);         
    for r=1:nruns,        
      [n,k,r]    
      for l=1:M,                               
        LF =  CreateListFactorsNK(n,k);  
        ListOfLists{l} = LF;  
        org_fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end
      
      % Evolution of the landscape to maximize size of the Pareto set
      %[ListFactors,TheMax] = EvolvePF_Size(M,ListOfLists,nsteps,1);
      [ListFactors,TheMax] = EvolveCorrelations(M,ListOfLists,nsteps,1);
      
      for l=1:M,                               
        LF = ListFactors{l};
        fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end              
      eval(['save ', 'pos_corr_results_',num2str(n),'_',num2str(k),'_',num2str(r),'.mat ListOfLists ListFactors TheMax org_fvals fvals']);
   end 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Minimization of the correlations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for k=1:4,
    % The initial landscape is created
    [AllVectors,ListOfLists,fvals] = CreateLandscape(M,n,k,instance);         
    for r=1:nruns,        
      [n,k,r]    
      for l=1:M,                               
        LF =  CreateListFactorsNK(n,k);  
        ListOfLists{l} = LF;  
        org_fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end
      
      % Evolution of the landscape to maximize size of the Pareto set
      %[ListFactors,TheMax] = EvolvePF_Size(M,ListOfLists,nsteps,1);
      [ListFactors,TheMax] = EvolveCorrelations(M,ListOfLists,nsteps,-1);
      
      for l=1:M,                               
        LF = ListFactors{l};
        fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end              
      corrcoef(fvals)
      eval(['save ', 'neg_corr_results_',num2str(n),'_',num2str(k),'_',num2str(r),'.mat ListOfLists ListFactors TheMax org_fvals fvals']);
   end 
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Maximization of solutions in the Pareto front
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



for k=1:4,
    % The initial landscape is created
    [AllVectors,ListOfLists,fvals] = CreateLandscape(M,n,k,instance);         
    for r=1:nruns,        
      [n,k,r]    
      for l=1:M,                               
        LF =  CreateListFactorsNK(n,k);  
        ListOfLists{l} = LF;  
        org_fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end
      
      % Evolution of the landscape to maximize size of the Pareto set
      [ListFactors,TheMax] = EvolvePF_Size(M,ListOfLists,nsteps,1);
      %[ListFactors,TheMax] = EvolveCorrelations(M,ListOfLists,nsteps,-1);
      
      for l=1:M,                               
        LF = ListFactors{l};
        fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end              
      eval(['save ', 'pos_PF_results_',num2str(n),'_',num2str(k),'_',num2str(r),'.mat ListOfLists ListFactors TheMax org_fvals fvals']);
   end 
end



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Approximate case. Minimization of the correlations n=50
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n= 50;
ntrials = n;
nsteps = 25;
nruns = 10;
sample_size = 1024;
for k=1:4,
    % The initial landscape is created
    [AllVectors,ListOfLists,fvals] = ApproxCreateLandscape(M,n,k,instance,sample_size);         
    for r=1:nruns,        
      [n,k,r]    
      for l=1:M,                               
        LF =  CreateListFactorsNK(n,k);  
        ListOfLists{l} = LF;  
        org_fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end
      
      % Evolution of the landscape to maximize size of the Pareto set
      %[ListFactors,TheMax] = EvolvePF_Size(M,ListOfLists,nsteps,1);
      [ListFactors,TheMax] = ApproxEvolveCorrelations(M,ListOfLists,nsteps,ntrials,-1);
      
      for l=1:M,                               
        LF = ListFactors{l};
        fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
      end              
      corrcoef(fvals)
      eval(['save ', 'approx_neg_corr_results_',num2str(n),'_',num2str(k),'_',num2str(r),'.mat ListOfLists ListFactors TheMax org_fvals fvals']);
   end 
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Evolution of the landscape to maximize correlations
[ListFactors,TheMax] = EvolveCorrelations(M,ListOfLists,nsteps)

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



