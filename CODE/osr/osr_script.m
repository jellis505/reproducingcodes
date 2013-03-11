% Script created to create the graphs that we want to create for the osr
% dataset with the Relative attributes method

% Train the ranking function should be right here
% This portion of the code needs to have some ground truth data labeled and
% the relative similarities finished

% Clear all the data before running the script
clear all;


% These are the num of unseen classes and training images per class
num_unseen = 2;
trainpics = 30;
num_iter = 10;
held_out_attributes = 1;
% This is the number of iterations we want to do
accuracy = zeros(1,num_iter);

% load the models that we need to run the models
load('osr_gist.mat');
load('category_order_osr.mat');
load('..\..\DATA\osr\data.mat');

for iter = 1:num_iter
    
    % Create a random list of unseen images
    unseen = randperm(8,num_unseen);
    
    [O,S] = Create_O_and_S_Mats2(category_order,used_for_training,class_labels,8,unseen,trainpics);
    
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
        
        w = ranksvm_with_sim(osr_gist_Mat,O(1:O_length,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
        weights(:,l) = w;
    end
    
    % here we want to choose to take out some of the weights for each
    % attribute and also the category order
    rand_atts = randperm(6,6-held_out_attributes);
    for j = 1:length(rand_atts);
        new_weights(:,j) = weights(:,rand_atts(j));
        new_cat_order(j,:) = category_order(rand_atts(j),:);
    end
    
    
    % Get the predictions based on the outputs from rank svm
    relative_att_predictions = osr_gist_Mat*new_weights;
    
    % Seperate the training samples from the other training samples
    Train_samples = GetTrainingSample_per_category(relative_att_predictions,class_labels,used_for_training);
    
    % This section of the code we will choose lines from the attributes to
    % leave out
    
    % Calculate the means and covariances from the samples
    [means, Covariances] = meanandvar_forcat(Train_samples,unseen,new_cat_order,8);
    
    % Classify the predicted features from the system
    accuracy(iter) = BayesClass_RelAtt(relative_att_predictions,class_labels,means,Covariances,used_for_training);
end

    total_acc = mean(accuracy);

disp('The accuracy of this calculation: ');
disp(total_acc);