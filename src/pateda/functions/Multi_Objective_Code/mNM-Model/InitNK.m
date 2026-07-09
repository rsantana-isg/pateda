
global Tables
global ListFactors
global PE

clear fvals
n = 12;
k = 2;
PE = 2.^[k:-1:0];
filename = 'cfiles_fnt_N12_k2Inst_1.txt';

AccCard = 2.^[n-1:-1:0];

Card = 2*ones(1,n);

[ListFactors1] =  CreateListFactorsNK(n,k);
[ListFactors2] =  CreateListFactorsNK(n,k);

ListFactors = ListFactors1;
[Tables] = ReadFunctionsFromData(filename,ListFactors,Card);

for i=1:2^n;
  AllVectors(i,:) = IndexconvertCard(i-1,n,AccCard);  
  fvals(i,1) = EvaluateNK(AllVectors(i,:));
end  


ListFactors = ListFactors2;
[Tables] = ReadFunctionsFromData(filename,ListFactors,Card);

for i=1:2^n;
  %[vector] = IndexconvertCard(i-1,n,AccCard);  
  fvals(i,2) = EvaluateNK(AllVectors(i,:));
end  


[Index]=FindParetoSet(fvals,fvals);
fvals(Index,:)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
plot(fvals(:,1),fvals(:,2),'.b')
hold on
plot(fvals(Index,1),fvals(Index,2),'r*')

clear ThePairs
a = 0;
for i=1:n,
 i   
 set1 = ListFactors{i}(:)';
 set2 = setdiff([1:n],set1);
 for j=2:size(set1,2),
   for l=1:size(set2,2),  
     ListFactors = ListFactors2;
     ListFactors{i}(j) = set2(l);
     for t=1:2^n;       
       fvals(t,2) = EvaluateNK(AllVectors(t,:));
     end  
     a = a + 1;
     [Index]=FindParetoSet(fvals,fvals);
     TheVals{a} = fvals(Index,:);   
     aux_ind = 
     ThePairs(a,:) = [i,j,ListFactors2{i}(j),set2(l),size(Index,2)];
   end
 end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear ThePairs
clear TheMax
clear TheVals

ListFactors = ListFactors2;

for ll=1:10,    
a = 0;
for i=1:n,
 [ll,i]   
 set1 = ListFactors{i}(:)';
 set2 = setdiff([1:n],set1);
 for j=2:size(set1,2),
   for l=1:size(set2,2),  
     ListFactors = ListFactors2;
     ListFactors{i}(j) = set2(l);
     for t=1:2^n;       
       fvals(t,2) = EvaluateNK(AllVectors(t,:));
     end  
     a = a + 1;
     [Index]=FindParetoSet(fvals,fvals);
     TheVals{a} = fvals(Index,:);     
     ThePairs(a,:) = [i,j,ListFactors2{i}(j),set2(l),size(Index,2)];
   end
 end
end
 figure
 hist(ThePairs(:,5),30)
 [b1,c1] = max(ThePairs(:,5));
 TheMax(ll) = b1
 [ac1] = find(ThePairs(:,5)==b1);
  s_ac1 = size(ac1,1);
  if s_ac1==1
    c1 = ac1;
  else
    pos = randint(1,1,s_ac1)+1;
    c1 = ac1(pos)
  end
 
 ListFactors2{ThePairs(c1,1)}(ThePairs(c1,2)) = ThePairs(c1,4);
 ListFactors = ListFactors2;
end



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



