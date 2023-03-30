close all; clear;
%generate a series of random patterns with size 32 by 32 
%the number of random patterns is M
r=32;
M=r^2;

pattern=zeros(M,r,r);

for ii=1:M 
    %randomly generate one pattern
    pattern(ii,:,:)=rand(r,r);
end

save('pattern.mat','pattern');