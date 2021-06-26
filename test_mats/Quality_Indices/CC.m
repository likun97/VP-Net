function cc=CC(F,MS) 
d=size(F, 3);
F=double(F);
MS=double(MS);
for i=1:d
    c(i)=corr2(F(:,:,i),MS(:,:,i));
end
cc=(1/3)*(c(1)+c(2)+c(3));