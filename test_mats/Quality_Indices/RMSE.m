% function sol=RMSE(MS,F)
% MS=double(MS*255);
% F=double(F*255);
% [n,m,d]=size(F);
% D=(MS(:,:,1:d)-F).^2;
% sol=sqrt(sum(sum(sum(D)))/(n*m*d));




% 原来版本是下面这个  但是TAGRS实验 自己从头到尾都是归一化之后的数据很小

function sol=RMSE(MS,F)
MS=double(MS);
F=double(F);
[n,m,d]=size(F);
D=(MS(:,:,1:d)-F).^2;
sol=sqrt(sum(sum(sum(D)))/(n*m*d));
