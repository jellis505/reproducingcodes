% This function checks the differences between the feature implementation
% of gist from the authors and my implementation of the feature extraction

clear all;

% load the models that we need to run the models
load('osr_gist.mat');
load('category_order_osr.mat');
load('../../DATA/osr/data.mat');

% The two features are compared to each other and distances are found
dist = 0;
for j = 1:length(feat)
    dist = dist + pdist([feat(j,:);osr_gist_Mat(j,:)],'euclidean');
end

norm_dist = 0;
for j = 1:length(feat)
    norm_dist = norm_dist + (norm(feat(j,:)) + norm(osr_gist_Mat(j,:)))/2;
end

% Thus the difference of these features should be the dist/norm_dist*100
percent_diff = (dist/norm_dist)*100;
disp('The percent differences in the learned features are')
disp(percent_diff);

