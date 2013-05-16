% This function checks the difference between the learned ranks from the
% entire pipeline using the author's values, compared to my values.

% This test is done with no unseen classes, simply because we assume the
% author's learned weights were learned with no unseen classes.

clear all;

% These are the num of unseen classes and training images per class
num_unseen = 2;
trainpics = 30;
num_iter = 10;
held_out_attributes = 0;
labeled_pairs = 15;
looseness_constraint = 1;

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
%[O,S] = Create_O_and_S_Mats3(relative_ordering,used_for_training,class_labels,8,unseen,trainpics,labeled_pairs);
[O,S] = Create_O_and_S_Mats(relative_ordering,used_for_training,class_labels,8,unseen,trainpics,labeled_pairs);

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
        weights(:,l) = ranksvm_with_sim(feat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
        %w_theirs(:,l) = testrank(osr_gist_Mat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
    else
        % Re-Do the ranking and start over, because we chose category pairs
        % that did not have the O matrix for a given attribute.
        
        % This function creates the O and S matrix used in the ranking algorithm
        [O,S] = Create_O_and_S_Mats(category_order,used_for_training,class_labels,8,unseen,trainpics,labeled_pairs);
        
        % initialize the weights matrix that will be learned for ranking
        weights = zeros(512,6);
        
        % re-do the creation of the O and S matrix
        l = 1;
        disp('We had to redo the O and S matrix ranking, Pairs chosen were all similar for an attribute');
    end
end

% Get the learned ranks and the ranks provided
% Use there trained data
relative_att_predictions_theirs = feat*relative_att_predictor;
% Use my trained data
relative_att_predictions_mine = feat*weights;

% Seperate the training samples from the other training samples
Train_samples_theirs = GetTrainingSample_per_category(relative_att_predictions_theirs,class_labels,used_for_training);
Train_samples_mine = GetTrainingSample_per_category(relative_att_predictions_mine,class_labels,used_for_training);
 
% This is for debug to find the problem with the unseen scategories
means_mine = meanandvar_forcat(Train_samples_mine,unseen,category_order,8,looseness_constraint);
means_theirs = meanandvar_forcat(Train_samples_theirs,unseen,category_order,8,looseness_constraint);
 
% Now we want to make some plots of the ranking algorithms output
x = 1:1:8;
hold on;
for j = 1:6
    subplot(3,2,j);
    Y_mine = reshape(means_mine(:,j,:),1,8,1);
    Y_theirs = reshape(means_theirs(:,j,:),1,8,1);
    plot(x,Y_theirs,'-b',x,Y_mine,'-r');
    xlabel('Category Number');
    ylabel('Ranking Scores');
    axis tight
end

suptitle('Comparison of Found Ranking Scores -- 4 Pairs');

 
 
