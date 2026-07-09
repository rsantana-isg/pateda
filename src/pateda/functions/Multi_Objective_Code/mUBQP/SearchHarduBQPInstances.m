n = 5;
Card = 2*ones(1,n); 
AccCard = cumprod(Card)/2;
AccCard = AccCard([n:-1:1]);
N = 2^n;
Pop = zeros(N,n);

% Generation of the population
for j=1:N,
 num = IndexconvertCard(j-1,n,AccCard);
 Pop(j,:) = num;
 IndPop{j} = find(Pop(j,:)==1);
end

% Generation of the space of parameters
nparams =10;   % Number of parameters that define an ubqp of order 5
Card = 2*ones(1,nparams); 
AccCard = cumprod(Card)/2;  
AccCard = AccCard([nparams:-1:1]);

np_confs = 2^nparams; % Number of possible configurations of 10 parameters
for j=1:np_confs,
 num = IndexconvertCard(j-1,nparams,AccCard)';
 Params(:,j) = 2*num-1;
end

% Creation of the uBQP function structure
clear all_obj
a = 0;
for i=1:n-1,
 for  j=i+1:n,
   a = a + 1;  
   all_obj(a,:) = [i,j,0];  
 end
end 

% Creation of all possible values of the functions
FunEvals = zeros(N,np_confs);
 for l=1:np_confs,
  for i=1:N,   
   for j=1:nparams,
     FunEvals(i,l) = FunEvals(i,l) + Pop(i,all_obj(j,1))*Pop(i,all_obj(j,2))*Params(j,l);
   end
  end
 end
 
% Computation of the Boltzmann probabilities
t = 1.0; % temperature
for l=1:np_confs,
   Fit = FunEvals(:,l);  
   BoltzMannProb = 2.^(Fit/t);
   BoltzMannProb = BoltzMannProb/sum(BoltzMannProb);
   for i=1:n,   
     ind = find(Pop(:,i)==1);
     Univ(l,i) = sum(BoltzMannProb(ind));  
   end
   for i=1:N,   
       prop =1;
       for j=1:n,  
         if(Pop(i,j)==0) prop = prop*(1-Univ(l,j));
         else   prop = prop*(Univ(l,j));  
         end    
       end
       AllProp(i,l) = prop;
   end    
end 
     
% Determination of deceptive value
for l=1:np_confs,
  a = max(FunEvals(:,l));
  b = find(FunEvals(:,l)==a);
  [c,d] = sort(AllProp(:,l));  
  decp(l) = mean(d(b));
end 

fs = 16;
H=figure
plot(33-decp,'*')
xlabel('Functions','fontsize',fs)
ylabel('Degree of deception','fontsize',fs)
axis([0 1050 0 33])
saveas(H, 'newDegreeDeception.eps', 'eps2')


fs = 16;
H=figure
hist((ParData(:,5)+ParData(:,6))/2,100)
ylabel('Number of functions','fontsize',fs)
xlabel('Degree of deception','fontsize',fs)
saveas(H, 'newHistDegreeDeception.eps', 'eps2')

 
% Determination of univariate ranking of solutions
for l=1:np_confs,
  [val,pos] = sort(AllProp(:,l),'descend');
  AllRankings(:,l) = pos;
end 



% Creation of Pareto sets
a = 0;
 for i=1:np_confs-1,    
   for  j=i+1:np_confs,
    a = a + 1;   
    [Index]=FindParetoSet(Pop,FunEvals(:,[i,j]));
    ParData(a,:) = [i,j,size(Index,2)];
   end   
 end  
    
 
 %*************************************************************************
 %**********************
 
 % Creation of Pareto sets saving the average rank of Pareto set solutions
 % for  the two objectives
ncands = np_confs*(np_confs-1)/2;
ParData = zeros(ncands,6);
a = 0;
 for i=1:np_confs-1,   
   i  
   for  j=i+1:np_confs,
    a = a + 1;   
    [Index]=FindParetoSet(Pop,FunEvals(:,[i,j]));
    ll =0;
    allsols = [];
    for k=1:size(Index,2),
        thesols = find(FunEvals(:,i)==FunEvals(Index(k),i) & FunEvals(:,j)==FunEvals(Index(k),j));
        ll = ll + size(thesols,1);
        allsols = [allsols;thesols]; 
    end    
    meani = mean(AllRankings(allsols,i));
    meanj = mean(AllRankings(allsols,j));
    ParData(a,:) = [i,j,size(Index,2),ll,meani,meanj];
   end   
 end  
 
 
 %*************************************************************************
 %**********************
 
dd = find(ParData(:,3)>=7 & decp(ParData(:,1))'<8 & decp(ParData(:,2))'<8)
 for i=1:size(dd,1),    
    [Index]=FindParetoSet(Pop,FunEvals(:,[ParData(dd(i),1),ParData(dd(i),2)]));
    AllIndex{i} =  FunEvals(Index,[ParData(dd(i),1),ParData(dd(i),2)]);
 end  

 
% Generation of instances that have at least 7 non-dominated solutions and 
% deceptive values below 8 
dd = find(ParData(:,3)>=7 & decp(ParData(:,1))'<8 & decp(ParData(:,2))'<8);
    
% Generate the desired instances
mat1 = all_obj;
mat2 = all_obj;
newn = 100;
nobj = 2;
seed = 10;
for i=1:size(dd,1),   
  fname = ['HardInst',num2str(i),'_'];  
  mat1(:,3) = Params(:,ParData(dd(i),1));
  mat2(:,3) = Params(:,ParData(dd(i),2));
  CreateUBQInstanceFromChunk(n,newn,nobj,mat1,mat2,fname,seed);
end


% Generation of instances that have at least 4 non-dominated solutions and 
% deceptive values below 3
dd = find(ParData(:,3)>=4 & decp(ParData(:,1))'<3 & decp(ParData(:,2))'<3)
% Generate the desired instances
mat1 = all_obj;
mat2 = all_obj;
newn = 500;
nobj = 2;
seed = 10;
for i=1:size(dd,1),   
  fname = ['LessPopInst',num2str(i),'_'];  
  mat1(:,3) = Params(:,ParData(dd(i),1));
  mat2(:,3) = Params(:,ParData(dd(i),2));
  CreateUBQInstanceFromChunk(n,newn,nobj,mat1,mat2,fname,seed);
end


 
 for i=1:size(dd,1),
  hold on; plot(AllIndex{i}(:,1),AllIndex{i}(:,2),'ro')
 end  

 
 %%%%%%%%%%%%%%
 
  load ParData
  nmemb = 10;
  totake = 100;
  PopulatedPareto_10 = find(ParData(:,4)>nmemb);
  [valp,posp] = sort(ParData(PopulatedPareto_10,5)+ParData(PopulatedPareto_10,6),'descend');
  
  
  for i=1:totake,
     ind =  PopulatedPareto_10(posp(i));
     SelectedParams{i,1} = all_obj; 
     SelectedParams{i,2} = all_obj;
     SelectedParams{i,1}(:,3) = Params(:,ParData(ind,1));
     SelectedParams{i,2}(:,3) = Params(:,ParData(ind,2));
  end
  
 n = 5; newn = 100; nobj=2; 
for i=1:10,   
  fname = ['ComposeInstHard10P',num2str(i),'_'];  
   perm = randperm(totake);
   ComposePerm100(i,:) = perm(1:newn/5);
   CreateUBQInstanceFromListChunks(n,newn,nobj,SelectedParams,perm(1:newn),fname,seed);
end
  

%%%%%%%%%%%%%%%%%%%% Saturated instances


  load ParData
  nmemb = 10;
  totake = 100;
  PopulatedPareto_10 = find(ParData(:,4)>nmemb);
  [valp,posp] = sort(ParData(PopulatedPareto_10,5)+ParData(PopulatedPareto_10,6),'descend');
  
  
  for i=1:totake,
     ind =  PopulatedPareto_10(posp(i));
     SelectedParams{i,1} = all_obj; 
     SelectedParams{i,2} = all_obj;
     SelectedParams{i,1}(:,3) = Params(:,ParData(ind,1));
     SelectedParams{i,2}(:,3) = Params(:,ParData(ind,2));
  end
  
 n = 5; newn = 100; nobj=2; seed=101; seeding_size = 100;
 clear SelectedParams
for desp=[500], 
 for j=0:10,   
  totake = 10 + j*desp;    
   for i=1:seeding_size,
     ind =  PopulatedPareto_10(posp(i));
     SelectedParams{i,1} = all_obj; 
     SelectedParams{i,2} = all_obj;
     SelectedParams{i,1}(:,3) = Params(:,ParData(ind,1));
     SelectedParams{i,2}(:,3) = Params(:,ParData(ind,2));
  end
  fname = ['HeavyInstHard',num2str(totake),'_'];  
   perm = fix(rand(1,totake)*seeding_size)+1;
   CreateHeavyUBQInstanceFromListChunks(n,newn,nobj,SelectedParams,perm,fname,seed);
 end
end
  
% *************************************************************

GenoDist = pdist(Pop);

for i=1:size(FunEvals,2)
  PhenoDist = pdist(FunEvals(:,i));
  ccorr = corrcoef(PhenoDist,GenoDist);
  DistanceCorr(i) = ccorr(1,2);
end
  
neg_fitcorr = find(DistanceCorr<0);
theind = [];
for i=1:length(neg_fitcorr)-1,
 for j=i+1:length(neg_fitcorr),   
  dd = find(ParData(:,1)==neg_fitcorr(i) & ParData(:,2)==neg_fitcorr(j));
  theind = [theind,dd];
 end 
end

feasible = find(ParData(theind,3)>1);



% *************************************************************

GenoDist = pdist(Pop);

for i=1:size(FunEvals,2)
  PhenoDist = pdist(FunEvals(:,i));
  ccorr = corrcoef(PhenoDist,GenoDist);
  DistanceCorr(i) = ccorr(1,2);
end
  
neg_fitcorr = find(DistanceCorr<0);
for i=1:length(neg_fitcorr), 
  theind = [];  
  dd = find(ParData(:,1)==neg_fitcorr(i) & ParData(:,3)>1 & ParData(:,3)<=2 & ParData(:,4)<=2);
  theind = [theind,dd'];
  dd = find(ParData(:,2)==neg_fitcorr(i) & ParData(:,3)>1 & ParData(:,3)<=2 & ParData(:,4)<=2);
  theind = [theind,dd'];
  TheFitCorrInd{i} = theind;
end

 n = 5; newn = 500; nobj=2; seed=101; 
 clear SelectedParams

 for j=1:length(neg_fitcorr),   
  totake = 500;    
  seeding_size = size(TheFitCorrInd{j},1);
   for i=1:seeding_size,
     ind =TheFitCorrInd{j}(i);    
     SelectedParams{i,1} = all_obj; 
     SelectedParams{i,2} = all_obj;
     SelectedParams{i,1}(:,3) = Params(:,ParData(ind,1));
     SelectedParams{i,2}(:,3) = Params(:,ParData(ind,2));
  end
  fname = ['FitCorrInstHard',num2str(j),'_'];  
   perm = fix(rand(1,totake)*seeding_size)+1;
   CreateHeavyUBQInstanceFromListChunks(n,newn,nobj,SelectedParams,perm,fname,seed);
 end



