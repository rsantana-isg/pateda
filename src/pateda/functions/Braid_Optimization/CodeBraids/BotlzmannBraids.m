n = 10;
Card = 4*ones(1,n); 
AccCard = cumprod(Card)/4;
AccCard = AccCard([n:-1:1]);
N = 4^n;
Pop = zeros(N,n);

for j=1:N,
 num = IndexconvertCard(j-1,n,AccCard);
 Pop(j,:) = num;
 Fit(j,1) = fitness_braid(num);
end

% Computation of the Boltzmann probabilities
t = 1.0; % temperature
BoltzMannProb = exp(Fit/t);
BoltzMannProb = BoltzMannProb/sum(BoltzMannProb);

% Computation of the univariate probabilities

clear Univ
for i=1:n,
 for j=1:4;
   ind = find(Pop(:,i)==(j-1));
   Univ(j,i) = sum(BoltzMannProb(ind));
 end
end


% Computation of the bivariate probabilities
% for every pair of variables

clear Biv
for i=1:n,
 for ii=1:n;   
  for j=1:4;
   for jj=1:4;         
     ind = find(Pop(:,i)==(j-1) & Pop(:,ii)==(jj-1));
     Biv{i,ii}(j,jj) = sum(BoltzMannProb(ind));
   end
  end
 end
end


thecolors = 'rbgkmcyrbg';
figure
for i=1:n,
  plot(Univ(:,i),[thecolors(i),'-.']);    
  hold on
end
close all
figure
for i=1:4,
  plot(Univ(i,:),[thecolors(i),'-.']);    
  hold on
end

close all
figure
for i=2:n,
  plot(Biv{1,i}(1,:),[thecolors(i),'-.']);    
  hold on
end


%%%%%%%%%%%%%%%%%%

% Computation of the Boltzmann distribution for when 
% effective length is used in the computation of the fitness

for i=1:N,
 ELFit(i,1) = fitness_braid_cond(Pop(i,:));
end

% Computation of the Boltzmann probabilities
t = 1.0; % temperature
ELBoltzMannProb = exp(ELFit/t);
ELBoltzMannProb = ELBoltzMannProb/sum(ELBoltzMannProb);

% Computation of the univariate probabilities

clear ELUniv
for i=1:n,
 for j=1:4;
   ind = find(Pop(:,i)==(j-1));
   ELUniv(j,i) = sum(ELBoltzMannProb(ind));
 end
end


% Computation of the bivariate probabilities
% for every pair of variables

clear ELBiv
for i=1:n,
 for ii=1:n;   
  for j=1:4;
   for jj=1:4;         
     ind = find(Pop(:,i)==(j-1) & Pop(:,ii)==(jj-1));
     ELBiv{i,ii}(j,jj) = sum(ELBoltzMannProb(ind));
   end
  end
 end
end


thecolors = 'rbgkmcyrbgkmcyr';

figure
for i=1:n,
  plot(ELUniv(:,i),[thecolors(i),'-.']);    
  hold on
end
close all
figure
for i=1:4,
  plot(ELUniv(i,:),[thecolors(i),'-.']);    
  hold on
end

close all
figure
for i=2:n,
  plot(ELBiv{1,i}(1,1),[thecolors(i),'-.']);    
  hold on
end

a = 1;
for i=1:4,
  for j=1:4,
    for k=1:9,
      bp(a,k) = ELBiv{k,k+1}(i,j)  
    end
    a = a + 1;
  end
end

close all
figure
for i=1:16,
  plot(bp(i,:),[thecolors(i),'-.']);    
  hold on
end


%%%%%%%%%%%%%%%%%%

% Computation of the Boltzmann distribution for when 
% we find the best possible product of matrices within a braid (starting
% from the first)


for i=1:N,
 BESTFit(i,1) = fitness_braid_best(Pop(i,:));
end

% Computation of the Boltzmann probabilities
t = 1.0; % temperature
BESTBoltzMannProb = exp(BESTFit/t);
BESTBoltzMannProb = BESTBoltzMannProb/sum(BESTBoltzMannProb);

% Computation of the univariate probabilities

clear BESTUniv
for i=1:n,
 for j=1:4;
   ind = find(Pop(:,i)==(j-1));
   BESTUniv(j,i) = sum(BESTBoltzMannProb(ind));
 end
end


% Computation of the bivariate probabilities
% for every pair of variables

clear BESTBiv
for i=1:n,
 for ii=1:n;   
  for j=1:4;
   for jj=1:4;         
     ind = find(Pop(:,i)==(j-1) & Pop(:,ii)==(jj-1));
     BESTBiv{i,ii}(j,jj) = sum(BESTBoltzMannProb(ind));
   end
  end
 end
end


thecolors = 'rbgkmcyrbgkmcyr';

figure
for i=1:n,
  plot(BESTUniv(:,i),[thecolors(i),'-.']);    
  hold on
end

close all
figure
for i=1:4,
  plot(BESTUniv(i,:),[thecolors(i),'-.']);    
  hold on
end

close all
figure
for i=2:n,
  plot(BESTBiv{1,i}(1,1),[thecolors(i),'-.']);    
  hold on
end


a = 1;
for i=1:4,
  for j=1:4,
    for k=1:9,
      bp(a,k) = BESTBiv{k,k+1}(i,j)  
    end
    a = a + 1;
  end
end

close all
figure
for i=1:16,
  plot(bp(i,1:5),[thecolors(i),'-.']);    
  hold on
end


