% Relative Attributes %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This program performs the tests that are described in the paper that
% corresponds to the citaiton below.
% D. Parikh and K. Grauman. "Relative Attributes". International Conference
% on Computer Vision (ICCV), 2011
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%
% Created by: Joseph Ellis
% PhD Candidate -- Columbia University
% jge2105@columbia.edu
%%%%%%%%%%%%%%%%%%

% Clear all the data before running the script
clear all;

%% Parameters 
num_unseen = 5; % number of unseen classes
trainpics = 30; % number of pictures used in training the O matrix for each class
held_out_attributes = 0; % The number of attributes held out 
labeled_pairs = 4; % The number of category pairs that are used for creating the O matrix
looseness_constraint = 1;  % Looseness Constraint as described in the paper for for Relative Attributes


%% Load Data
% Load the data to perform the relative attribute experiments.  This data
% can be downloaded at http://filebox.ece.vt.edu/~parikh/relative.html
load('osr_gist.mat');
load('data.mat');
category_order = relative_ordering;

%% Execute Code
% Create a random list of unseen images
for k = 1:10
    
    unseen = randperm(8,num_unseen);
    %{
    % This function creates the O and S matrix used in the ranking algorithm
    [O,S] = Create_O_and_S_Mats(category_order,used_for_training,class_labels,8,unseen,trainpics,labeled_pairs);
    
    % initialize the weights matrix that will be learned for ranking
    weights = zeros(512,6);
    for l = 1:6
        
        % Find when the O matrix for this dimension no longer has real values
        % We do this because the O and S matrix are generated with randomly
        % chosen category pairs, so we do not know how large this should be for
        % each attribute.
        for j = 1:size(O,1)
            O_length = j;
            if ismember(1,O(j,:,l)) == 0;
                break;
            end
        end
        
        % Find when the S matrix for this dimension no longer has real values.
        % We do this because the O and S matrix are generated with randomly
        % chosen category pairs, so we do not know how large this should be for
        % each attribute.
        for j = 1:size(S,1)
            S_length = j;
            if ismember(1,S(j,:,l)) == 0;
                break;
            end
        end
        
        % Now set up the cost matrices both are initialized to 0.1 in the
        % Relative Attributes paper from 2011;
        Costs_for_O = .1*ones(O_length-1,1);
        Costs_for_S = .1*ones(S_length,1);
        
        % We want to make sure that the random pairings that we have chosen have
        % atleast one pairing for each attribute in the O matrix, we can not
        % rank if we ONLY have similarities for an attribute.
        if O_length > 1
            w = ranksvm_with_sim(osr_gist_Mat,O(1:O_length-1,:,l),S(1:S_length,:,l),Costs_for_O,Costs_for_S);
            weights(:,l) = w;
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
    %}
    % If we do not want to use the Full
    % here we want to choose to take out some of the weights for each
    % attribute and also the category order
    
    if held_out_attributes ~= 0
        rand_atts = randperm(6,6-held_out_attributes);
        for j = 1:length(rand_atts);
            %new_weights(:,j) = weights(:,rand_atts(j));
            new_cat_order(j,:) = category_order(rand_atts(j),:);
            new_relative_att_predictor(:,j) = relative_att_predictor(:,rand_atts(j));
        end
    else
        new_cat_order = category_order;
        %new_weights = weights;
        new_relative_att_predictor = relative_att_predictor;
    end
    
    %%%%%%%%%%%%%%%
    % Choose to use the weights given with the data, that are loaded with
    % data.mat, or the learned weights from above
    % Comment out LINE 1 to use the given ranks with data.mat
    % Comment out LINE 2 to use the ranks learned above by the ranking function
    % LINE 1
    % relative_att_predictions = feat*new_weights;
    % LINE 2
    relative_att_predictions = feat*new_relative_att_predictor;
    %%%%%%%%%%%%%%%
    
    
    % Seperate the training samples from the testing samples to learn the
    % distributions or the classes over the attribute ranks.
    Train_samples = GetTrainingSample_per_category(relative_att_predictions,class_labels,used_for_training);
    
    % Calculate the true means and covariances for all of the samples with the training data
    % whether they are seen or unseen.  This is done by sending this function
    % the empty matrix for the unseen classes.  This is only done for debug
    % purposes so that we can compare the actual training sample means to the
    % means and covariances found using the algorithm specified in section 3.2
    % in the paper.
    [means, Covariances] = meanandvar_forcat2(Train_samples,[],new_cat_order,8,looseness_constraint);
    
    % Calculate the means and covariances according to the framework specificed
    % in the paper for the all seen and unseen classes.  Section 3.2
    [means_unseen, Covariances_unseen] = meanandvar_forcat2(Train_samples,unseen,new_cat_order,8,looseness_constraint);
    
    % This function classifies each sample that is not used for training as a
    % class based on the distrubtions that were created by the above function
    % meanandvar_forcat.  The result of the function is the accuracy of the
    % classification scheme for this iteration.
    
    % This is the accuracy of only the unseen classes
    unseen_accuracy(k) = BayesClass_RelAtt_unseen_only(relative_att_predictions,class_labels,means_unseen,Covariances_unseen,used_for_training,unseen);
    % This is the accuracy of all samples
    other_acc(k) = BayesClass_RelAtt(relative_att_predictions,class_labels,means_unseen,Covariances_unseen,used_for_training,unseen);
    
    
    %% Print Statements for Debug Purposes
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Debug %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    disp('The relative ordering of the attributes for each image');
    disp('   -(Transpose to display in the same oreintation as the means below)')
    category_order'
    disp('These are the Actual Means that exist for this test. They should be similar to the ordering above');
    means
    disp('The unseen classes are')
    unseen
    disp('Difference between the actual means and the predicted means for our unseen classes');
    disp(means_unseen - means);
    disp('Accuracy on Unseen Classes');
    disp(unseen_accuracy(k))
    disp('Accuracy on all Classes');
    disp(other_acc(k));
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

% The calculation accuracy 
unseen_average = mean(unseen_accuracy)
total_average = mean(other_acc)