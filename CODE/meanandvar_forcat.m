% Generate mean and covariance matrix for each categories relative scores.
% Created by Joe Ellis for the Reproduction Code Class
% Reproducing Relative Attributes

function [means, Covariances] = meanandvar_forcat(Training_Samples)

% variables
% means = 2-d matrix each row is a mean of the labels should be 8x6 rows
% Covariances = 3-d matrix.  Should be 6x6x8 to finish this work.

% means of the set ups
means = mean(Training_Samples);

% Set up the covariance
Covariances = zeros(6,6,8);

% for loop to iterate over the sections of the training_samples
for j = 1:size(Training_Samples,3)
    Covariances(:,:,j) = cov(Training_Samples(:,:,j));
end




