% Calculate the gist descriptors for all of the images within a folder
% Created by Joe Ellis -- Columbia University 2013
% This function will read in all of the image pictures in a file and then
% extract gist descriptors for them.

function [gistandlab_Mat, paramsforimg] = Getgistdescriptorsandlab(filepath);

% Read in all of the files in a path and then extrat gist for them
pic_files = dir(filepath);
lengthofdir = length(pic_files);
% Get rid of the first 3 file names because they are '.' and '..'
pic_files = pic_files(3:lengthofdir);
lengthofdir = length(pic_files);

% Set up the Matrix Structure
gistandlab_Mat = [];

% GIST Parameters
param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
param.numberBlocks = 4;
param.fc_prefilt = 4;

% For loop to process all of the files
for j = 1:lengthofdir
    
    % Read in the image name
    imagename = strcat(filepath,pic_files(j).name);
    img = imread(imagename);
    
    %compute the gist features
    [gist, param] = LMgist(img,'',param);
    gistandlab_Mat(j,1:512) = gist;
    paramsforimg(j) = param;
    
    % Now get the lab color image space
    cform = makecform('srgb2lab');
    
    %now make the lab_img
    lab_img = applycform(img,cform);
    
    %histogram of the lab space is 
    hist_lab = histc(lab_img(:),30);
    
    % add this to the hist and lab mat 
    gistandlab_Mat(j,513:542) = hist_lab;
    
end
    


