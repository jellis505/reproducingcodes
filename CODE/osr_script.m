% Script created to create the graphs that we want to create for the osr
% dataset with the Relative attributes method

% Train the ranking function should be right here
% This portion of the code needs to have some ground truth data labeled and
% the relative similarities finished

% These are the unseen classes

% This is the number of iteratins we want to do



accuracy 
for iter = 1:20



unseen_list = 
unseen = [];

[O,S] = Create_O_and_S_Mats2(category_order,used_for_training,class_labels,8,unseen);

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

% Get the predictions based on the outputs from rank svm
relative_att_predictions = osr_gist_Mat*weights;

% Seperate the training samples from the other training samples
Train_samples = GetTrainingSample_per_category(relative_att_predictions,class_labels,used_for_training);

% Calculate the means and covariances from the samples
[means, Covariances] = meanandvar_forcat(Train_samples,unseen,category_order,8);

% Classify the predicted features from the system
accuracy = BayesClass_RelAtt(relative_att_predictions,class_labels,means,Covariances,used_for_training);

disp('The accuracy of this calculation: ');
disp(accuracy);