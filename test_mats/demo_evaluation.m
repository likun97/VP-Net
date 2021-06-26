

%%  You can also use matlab version code to test the fusion results of the model output

clear;clc;close all;
addpath('Quality_indices');  
matPath_ref = 'test_1\';                  % this folder includes .mats with fusion result and reference 
imgDir_ref  = dir([matPath_ref '*.mat']); 
 
% for i=1
for i=1:length(imgDir_ref)                % To traverse all fusion results in the folder   
load([matPath_ref imgDir_ref(i).name]);

I_GT = double(ref);
I_Proposed = double(fusion);

ratio=4;L=11;Qblocks_size=25;flag_cut_bounds=0;dim_cut=11;thvalues=0;

[Q_avg_proposed, SAM_proposed, ERGAS_proposed,RASE_proposed,UIQI_proposed,PNSR_proposed,SSIM_proposed,RMSE_proposed] = indexes_evaluation(I_Proposed,I_GT,ratio,L,Qblocks_size,flag_cut_bounds,dim_cut,thvalues);
Matrix(i,:) =[ERGAS_proposed,SAM_proposed,Q_avg_proposed,PNSR_proposed,SSIM_proposed,RMSE_proposed,RASE_proposed,UIQI_proposed];

end



