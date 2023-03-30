%% this file is to generate the dataset with the label (clean image) and the input data (recovered image by using DGI)
clear;close all;

%load dataset and random patterns
load('mnist.mat','test');
load('pattern.mat','pattern');

savepath = 'test_mnist.h5';

size_input = 32;
data = zeros(size_input, size_input, 1, 1);
label = zeros(size_input, size_input, 1, 1);

count = 0;

% define how many testing images in MNIST will be used 
length=100;

for i = 1 : length
    
    image = test.images(:,:,i);

    image = imresize(image,[size_input,size_input],'nearest');
    
   %% using DGI to recover the image
    %measurement
    Mea_B=zeros(size_input*size_input,1);
    mean_B = 0;
    mean_I = 0;
    mean_R = 0;
    O = 0;
    
    for ii=1:size_input*size_input
        I = pattern(ii);
        %correlation
        Mea_B(ii)=sum(sum(I.*image));
        %recover
        mean_R = (mean_R*(ii-1)+sum(sum(I)))/ii;    % <R>
        mean_B = (mean_B*(ii-1)+Mea_B(ii))/ii;     % <B>
        BB = Mea_B(ii)-mean_B/mean_R*sum(sum(I));  % <B>/<R>*R
        mean_I=(mean_I*(ii-1)+I)/ii;                % <I>
        II=I-mean_I;                                % I-<I>
        O=(O*(ii-1)+BB*II)/ii; 
    end
    %O is the recovered image with DGI 
    O=(O-min(min(O)))/(max(max(O))-min(min(O)));

    count=count+1;
    
    data(:, :, 1, count) = O;
    label(:, :, 1, count) = image; 
end

%% shuffle the dataset
order = randperm(count);
data = data(:, :, 1, order);
label = label(:, :, 1, order); 

%% writing to HDF5
chunksz = 100;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    %batchno
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,1,last_read+1:last_read+chunksz); 
    batchlabs = label(:,:,1,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz); 
    created_flag = true;
    totalct = curr_dat_sz(end);
end

h5disp(savepath);
