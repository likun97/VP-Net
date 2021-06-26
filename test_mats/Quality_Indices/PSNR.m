function out = PSNR(ref,tar)

[~,~,bands] = size(ref);
ref = reshape(ref,[],bands);
tar = reshape(tar,[],bands);
msr = mean((ref-tar).^2,1);
max2 = max(ref,[],1).^2;
    
psnrall = 10*log10(double(max2)./msr);
out.all = psnrall;
out.ave = mean(psnrall);
end