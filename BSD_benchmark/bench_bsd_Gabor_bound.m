addpath benchmarks

clear all;close all;clc;

num_imgs = 25;
imgDir = '../../data/images/'+ string(num_imgs) + 'images/test/';
gtDir = '../../data/groundTruth/';
inDir = '../outdir/'+ string(num_imgs) + 'images/image_contours/';
%inDir = '../outdir/'+ string(num_imgs) + 'images/image_segmentations/';
nthresh = 20;

%%
% openfig('isoF.fig')
nslic_list_dir = dir(fullfile(inDir));
nslic_list_dir = nslic_list_dir([nslic_list_dir.isdir]);
nslic_list_dir = nslic_list_dir(~ismember({nslic_list_dir.name},{'.','..'}));

%disp(nslic_list_dir)
%%

for i = 1:numel(nslic_list_dir)
    % models_list_dir = dir(fullfile(inDir));
    models_list_dir = dir(fullfile(strcat(nslic_list_dir(i).folder), strcat(nslic_list_dir(i).name)));
    models_list_dir = models_list_dir([models_list_dir.isdir]);
    models_list_dir = models_list_dir(~ismember({models_list_dir.name},{'.','..'}));
    % disp(models_list_dir)
    
    for j = 1:numel(models_list_dir)
        gabor_combination_dir = dir(fullfile(strcat(models_list_dir(j).folder), strcat(models_list_dir(j).name)));
        gabor_combination_dir = gabor_combination_dir([gabor_combination_dir.isdir]);
        gabor_combination_dir = gabor_combination_dir(~ismember({gabor_combination_dir.name},{'.','..'}));
    %     disp(gabor_combination_dir);
    
        for k = 1 :length(gabor_combination_dir)
            list_dir_pngs = dir(fullfile(strcat(gabor_combination_dir(k).folder), strcat(gabor_combination_dir(k).name), '*.png'));
    %         disp(gabor_combination_dir);
            outDir = fullfile(strcat(gabor_combination_dir(k).folder), strcat(gabor_combination_dir(k).name));
            disp(outDir);
            %% clean up
            system(sprintf('rm -f %s/eval_bdry.txt',outDir));
            system(sprintf('rm -f %s/eval_bdry_img.txt',outDir));
            system(sprintf('rm -f %s/eval_bdry_thr.txt',outDir));
            system(sprintf('rm -f %s/eval_cover.txt',outDir));
            system(sprintf('rm -f %s/eval_cover_img.txt',outDir));
            system(sprintf('rm -f %s/eval_cover_th.txt',outDir));
            system(sprintf('rm -f %s/eval_RI_VOI.txt',outDir));
            system(sprintf('rm -f %s/eval_RI_VOI_thr.txt',outDir));
            
            %%
            tic;
            boundaryBench(imgDir, gtDir, outDir, outDir, nthresh);
            %allBench(imgDir, gtDir, outDir, outDir, nthresh);
            toc;
            plot_eval(outDir);      
            close all;
        end
    end
end


