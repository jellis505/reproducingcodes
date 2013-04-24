% This function checks the differences between the feature implementation
% learned weights using the author's code and implementation of RankSVM,
% and my implementation.

clear all;

% load the models that we need to run the models
load('osr_gist.mat');
load('category_order_osr.mat');
load('../../DATA/osr/data.mat');

% These are the num of unseen classes and training images per class
num_unseen = 0;
trainpics = 30;
num_iter = 10;
held_out_attributes = 0;
labeled_pairs = 10;
looseness_constraint = 1;
% This is the number of iterations we want to do
accuracy = zeros(1,num_iter);

% load the models that we need to run the models
load('osr_gist.mat');
load('category_order_osr.mat');
load('../../DATA/osr/data.mat');
category_order = relative_ordering;


    
    % Create a random list of unseen images
    unseen = randperm(8,num_unseen);
    
    % The three possible ways to train the matrices
    %[O,S] = Create_O_and_S_Mats(category_order,used_for_training,class_labels,8);
    %[O,S] = Create_O_and_S_Mats2(relative_ordering,used_for_training,class_labels,8,unseen,trainpics);
    [O,S] = Create_O_and_S_Mats3(relative_ordering,used_for_training,class_labels,8,unseen,trainpics,labeled_pairs);
    
    % Now we need to train the ranking function, but we have some values in the
    % matrices that will not correspond to the anything becuase some attributes
    % will have more nodes with similarity.
    weights = zeros(512,6);
    for l = 1:6
        
        % Find where each O and S matrix stops having values for each category
        % matrix section
        
        % Find when the O matrix for this dimension no longer has real values
        
        for j = 1:size(O,1)
            O_length = j;
            if ismember(1,O(j,:,l)) == 0;
                break;
            end
        end
        
        % Find when the S matrix for this dimension no longer has real values.
        for j = 1:size(S,1)
            S_length = j;
            if ismember(1,S(j,:,l)) == 0;
                break;
            end
        end
        
        % Now set up the cost matrices both are initialized to 0.1 in the
        % Relative Attributes paper from 2011;
        Costs_for_O = .1*ones(O_length,1);
        Costs_for_S = .1*ones(S_length,1);
        
        if O_length > 1
            w_mine(:,l) = ranksvm_with_sim(osr_gist_Mat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
            w_theirs(:,l) = testrank(psr_gist_Mat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
        else
            l = l-1;
        end
    end
    
    % The two features are compared to each other and distances are found
dist = 0;
for j = 1:length(w_mine)
    dist = dist + pdist([w_mine(j,:);w_theirs(j,:)],'euclidean');
end

norm_dist = 0;
for j = 1:length(w_mine)
    norm_dist = norm_dist + (norm(w_mine(j,:)) + norm(w_theirs(j,:)))/2;
end

% Thus the difference of these features should be the dist/norm_dist*100
percent_diff = (dist/norm_dist)*100;
disp('The percent differences in the learned features are')
disp(percent_diff);
