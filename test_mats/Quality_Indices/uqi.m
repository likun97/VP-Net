function Q = uqi(x,y)
x = double(x(:));
y = double(y(:));
mx = mean(x);
my = mean(y);
C = cov(x,y);
Q = (4*C(1,2)*(mx)*(my))/((C(1,1)+C(2,2))*((mx)^2+(my)^2));
end