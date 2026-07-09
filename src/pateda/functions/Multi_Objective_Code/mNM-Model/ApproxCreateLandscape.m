function [AllVectors,ListOfLists,fvals] = ApproxCreateLandscape(M,n,k,instance,sample_size)

global Tables
global PE
global AllVectors



% Input parameters of the model
% M   % Number of objectives
% n   % Number of variables
% k   % Number of neighbors
% instance  % Instance used to read Table parameters from files


% Auxiliary variables
PE = 2.^[k:-1:0];
AccCard = 2.^[n-1:-1:0];
Card = 2*ones(1,n);

% INITIALIZATION OF THE MODEL

% The full population is created 


AllVectors = fix(2*rand(sample_size,n));    


% File with the Tables for each variable
filename = ['cfiles_fnt_N',num2str(n),'_k',num2str(k),'Inst_',num2str(instance),'.txt']; 

% The neighborhoods are randomly selected for each objective
for l=1:M,                           
  LF =  CreateListFactorsNK(n,k);  
  ListOfLists{l} = LF;  
end

% The Tables are read (the same set of tables is used for all objectives)
[Tables] = ReadFunctionsFromData(filename,ListOfLists{1},Card);

% Initial evaluation of the model

for l=1:M,                               
  LF = ListOfLists{l};
  fvals(:,l) =  EvaluatePopNK(AllVectors,LF);
end
