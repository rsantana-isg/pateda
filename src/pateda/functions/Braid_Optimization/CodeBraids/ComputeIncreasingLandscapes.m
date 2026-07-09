clear all
InitBraid(0.0);

for n=1:9,
Card = 4*ones(1,n); 
AccCard = cumprod(Card)/4;
AccCard = AccCard([n:-1:1]);
N = 4^n;
Pop = zeros(N,n);

for i=1:N,
 num = IndexconvertCard(i-1,n,AccCard);
 Pop(i,:) = num;
 Fit(i,1) = fitness_braid(num);
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
filename = ['IncBraidLandScape_',num2str(n),'.mat'];
if (n>1)
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
    eval(['save ',filename, ' Fit Pop Univ BoltzMannProb Biv N MI ;']);    
else 
    eval(['save ',filename, ' Fit Pop Univ BoltzMannProb N ;']);    
end    
AllUniv1(:,n) = Univ(:,1);
if (n>2)
  AllUniv2(:,n) = Univ(:,2);
end
   
n
end


