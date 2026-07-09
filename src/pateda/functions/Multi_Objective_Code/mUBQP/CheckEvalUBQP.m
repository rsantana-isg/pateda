cd /home/rsantana/Dropbox/WorkBrazil/Murilo/murilo-roberto-alex/Test' for mUBQP'/VAR/

load LessPop2.dat.txt
X = LessPop2_dat;

for i=1:size(X,1),
 fVals(i,:) = Eval_uBQP(X(i,:));
end

imagesc(LessPop2_dat)
gg = unique(LessPop2_dat,'rows');