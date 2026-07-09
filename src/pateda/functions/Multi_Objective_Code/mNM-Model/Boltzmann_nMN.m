
N = 10;
M = 9;
AccCard = 2.^[N-1:-1:0];
Card = 2*ones(1,N);
sigma = 1;

% The complete population is created
clear AllVectors;
for i=1:2^N,
  AllVectors(i,:) = IndexconvertCard(i-1,N,AccCard);    
end  

count=1;
for sigma=21:2:20,
for runs=1:10,
   [count,runs] 

% Bi-objective models are computed for different numbers of 
% interactions

 % The mNM model is created
 [mNM] =  Create_mNM_model(sigma,M,N);
 
for k=1:9,   
 %[sigma,k]   
 other_mNM1 =  Save_mNM_model(mNM,k);
  for m=1:2,     
   if m==1,
    [fv] =  EvaluatePop_nNM(2*AllVectors-1,other_mNM1);
   elseif m==2
    [fv] =  EvaluatePop_nNM(-2*AllVectors+1,other_mNM1);  
   end
   Multi_NM(m,:) = fv;
  end
  AllMulti_NM{k} = Multi_NM; 
end  


% The Boltzmann distribution, Univ. prob and MI are computed


%index = find(triu(MI)~=0);


for kk=1:9,
  Fit = AllMulti_NM{kk}(2,:);
  Pop = AllVectors;
  % Computation of the Boltzmann probabilities
  t = 1.0; % temperature
  BoltzMannProb = exp(Fit/t);
  BoltzMannProb = BoltzMannProb/sum(BoltzMannProb);

  % Computation of the univariate probabilities
  n = N;
  clear Univ
  for i=1:n,
    for j=1:2;
     ind = find(Pop(:,i)==(j-1));
     Univ(j,i) = sum(BoltzMannProb(ind));
    end
  end

  AllUniv{runs}(kk,:) =Univ(j,:);

% Computation of the bivariate probabilities
% for every pair of variables
if (n>1)
    clear Biv
    for i=1:n,
        for ii=1:n;
            for j=1:2;
                for jj=1:2;
                    ind = find(Pop(:,i)==(j-1) & Pop(:,ii)==(jj-1));
                    Biv{i,ii}(j,jj) = sum(BoltzMannProb(ind));
                end
            end
        end
    end
   
    MI = zeros(n);    
    BivProb = Biv;
    UnivProb = Univ;
    for i=1:n-1,
        for j=i+1:n,
            for k=1:Card(i),
                for l=1:Card(j),
                    if(BivProb{i,j}(k,l)>0)
                        MI(i,j) = MI(i,j) + BivProb{i,j}(k,l)* log(BivProb{i,j}(k,l)/(UnivProb(k,i)*UnivProb(l,j)));
                    end
                end,
            end,
            MI(i,j) = MI(i,j)/(Card(i)*Card(j)); %Normalization of the mutual information
            MI(j,i) = MI(i,j);
        end,
    end,    
    %eval(['save ',filename, ' Fit Pop Univ BoltzMannProb Biv N MI ;']);    
else 
    %eval(['save ',filename, ' Fit Pop Univ BoltzMannProb N ;']);    
end    
 MeanMI(runs,kk) = mean(MI(index));
 
end
 Probs(:,runs) = BoltzMannProb;
 Models{runs} = AllMulti_NM;
end
AllSig{count} = {AllUniv,MeanMI,Models,Probs};
count = count+1;
end

thecolors = 'rcbgkyrcbgky'
theforms =  '*oxpdvxpdv'

close all
%%%% Showing the statistics 
sumAllUniv = zeros(10);
figure
for count=1:8,
 AllUniv = AllSig{count}{1}
 MeanMI = AllSig{count}{2}(:,1:5)
 plot(mean(MeanMI),['-',theforms(count),thecolors(count)])
 hold on
 %sumAllUniv = sumAllUniv + AllUniv;
end 

% FIGURES
%load BoltzmannModels_1-19.mat
%Models = AllSig{1}{3};

t = 1;
themod = [Models{3}{2}(1,:);Models{3}{2}(2,:)];
thebol =  exp(themod/t)';
thebol =  [thebol./repmat(sum(thebol),1024,1)]';

 clear Univ
  for i=1:n,
    for j=1:2;
     ind = find(Pop(:,i)==(j-1));
     aUniv(j,i) = sum(thebol(1,ind));
     bUniv(j,i) = sum(thebol(2,ind));
    end
  end
  for i=1:2^N,
    theproba =1;
    theprobb =1;
    for j=1:n,
      if  AllVectors(i,j) ==0
       theproba = theproba * aUniv(1,j);
       theprobb = theprobb * bUniv(1,j);
      else
       theproba = theproba * aUniv(2,j);
       theprobb = theprobb * bUniv(2,j);   
      end 
    end
    newbol(:,i) = [theproba;theprobb];
  end 

figure
plot(thebol(1,:),thebol(2,:),'r.')

 fa = 16;
 h = figure
 plot(themod(1,:),themod(2,:),'r.')
 set(gca,'fontsize',fa)
 fname = ['OrigF_n10_s1.eps']; 
 xlabel('f_1','FontSize',fa);
 ylabel('f_2','FontSize',fa); 
 saveas(gcf,fname,'psc2');

 
 fa = 16;
 h = figure
 plot(thebol(1,:),thebol(2,:),'b.')
 set(gca,'fontsize',fa)
 fname = ['Bolt_n10_s1.eps']; 
 xlabel('p^1_{B_i}','FontSize',fa);
 ylabel('p^2_{B_i}','FontSize',fa); 
 saveas(gcf,fname,'psc2');
 
 fa = 16;
 h = figure
 plot(newbol(1,:),newbol(2,:),'g.')
 set(gca,'fontsize',fa)
 fname = ['UMDA_n10_s1.eps']; 
 xlabel('q^1_{B_i}','FontSize',fa);
 ylabel('q^2_{B_i}','FontSize',fa); 
 saveas(gcf,fname,'psc2');
 
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
 close all
 thecolors = 'rcbgkyrcbgky'
theforms =  '*oxpdvxpdv'
%%%% Showing the statistics 
 


fa = 14;
 h = figure
 fname = ['MutInf_s1.eps']; 
for count=1:10, 
 MeanMI = AllSig{count}{2}(:,1:9)
 plot(mean(MeanMI),['-',theforms(count),thecolors(count)],'Markersize',8)
 hold on 
end
xlabel('Maximum order of interactions','FontSize',fa);
ylabel('Mutual information','FontSize',fa);
legend(['\sigma=1 ' ;'\sigma=3 ';'\sigma=5 ';'\sigma=7 ';'\sigma=9 ';'\sigma=11';'\sigma=13';'\sigma=15';'\sigma=17';'\sigma=19']);
set(gca,'fontsize',fa)
saveas(gcf,fname,'psc2');

%%% Kullback

for count=1:10,
     AllUniv = AllSig{count}{1};
     Models= AllSig{count}{3};
     Probs = AllSig{count}{4};
  for runs=1:10,
   
     for kk=1:9
        aUniv = AllUniv{runs}(kk,:);  
        t = 1;
        themod = Models{runs}{kk}(2,:);
        thebol =  exp(themod/t)';
         thebol =  [thebol./repmat(sum(thebol),1024,1)]';
      clear Univ
      for i=1:n,
        for j=1:2;
         ind = find(Pop(:,i)==(j-1));
         aUniv(j,i) = sum(thebol(1,ind));  
        end
      end 

        for i=1:2^N,
         theproba =1;
         for j=1:n,
          if  AllVectors(i,j) ==0
           theproba = theproba * aUniv(1,j);  
          else
           theproba = theproba * aUniv(2,j);    
          end 
         end
        newbol(:,i) = [theproba];
        end 
  
       KL{runs}(count,kk) =sum(thebol.*log(abs(thebol./newbol)))
       
     end
     
  end
end

AllKL = zeros(10,9);
for i=1:10,
    AllKL = AllKL +  KL{i};
end    
 AllKL = AllKL/10; 
 
 fa = 14;
 h = figure
 fname = ['KL_s1.eps']; 
for count=1:10, 
 
 plot(AllKL(count,:),['-',theforms(count),thecolors(count)],'Markersize',8)
 hold on 
end
xlabel('Maximum order of interactions','FontSize',fa);
ylabel('Kullback–Leibler divergence','FontSize',fa);
legend(['\sigma=1 ' ;'\sigma=3 ';'\sigma=5 ';'\sigma=7 ';'\sigma=9 ';'\sigma=11';'\sigma=13';'\sigma=15';'\sigma=17';'\sigma=19']);
set(gca,'fontsize',fa)
saveas(gcf,fname,'psc2');

 