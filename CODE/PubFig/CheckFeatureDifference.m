% This function checks the differences between the feature implementation
% of gist from the authors and my implementation of the feature extraction

clear all;

% load the models that we need to run the models
load('category_order_pubfig.mat');
load('../../DATA/pubfig/data.mat');
load('gistandlab.mat');

for j = 1:size(gistandlab_Mat,1)
    labpart = gistandlab_Mat(j,513:542);
    value = sum(labpart);
    labpart = labpart/value;
    gistandlab_Mat(j,513:542) = labpart;
end

% The two features are compared to each other and distances are found
dist = 0;
for j = 1:length(feat)
    dist = dist + pdist([feat(j,1:512);gistandlab_Mat(j,1:512)],'euclidean');
end

norm_dist = 0;
for j = 1:length(feat)
    norm_dist = norm_dist + norm(feat(j,1:512));
end

% Thus the difference of these features should be the dist/norm_dist*100
percent_diff = (dist/norm_dist)*100;
disp('The percent differences in the learned features are')
disp(percent_diff);

