%make_lr_datasets.m
%
%creates low resolution versions of the datasets using the matlab bicubic resizing scheme

clear all;close all;clc;
datasets = {'./div2k/urban100/test/'};
for d = 1:length(datasets)
    dset = datasets{d};
    files = dir(dset);
    files = files(3:end);
    output_dir = [dset(1:end-1) '_lr/'];
    mkdir(output_dir);
    for f = 1:length(files)
        fname = files(f).name;
        im = imread([dset fname]);
        sz = size(im);
        im = im(1:end-mod(sz(1),4),1:end-mod(sz(2),4),:);
        im_lr = imresize(im,0.25,'bicubic');
        disp(size(im_lr));
        imwrite(im_lr,[output_dir fname]);
    end
end
