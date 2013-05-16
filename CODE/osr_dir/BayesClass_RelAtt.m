% Bayesian Classification of the Relative Attributes
% Created by Joe Ellis -- PhD Candidate Columbia University
% This function takes in the means and Covariances Matrices of each class
% and then classifies the variables based on their values

function accuracy = BayesClass_RelAtt(predicts,ground_truth,means,Covariances,used_for_training,unseen)

% INPUTS
% predicts = the values that need to be predicted and classified these are
%   the relative predictions
% ground_truth = the real class_labels they are a 2668 vector;
% means = 1x6x8 matrix of the covariances and the means
% Covariances = 6x6x8 matrix fo the covariances

% OUTPUTS
% accuracy = Outputs the accuray of this classification

% This is for tracking the accuracy of the set up
correct = 0;
total = 0;

% Now do a for loop for each of the predicts variables, which are the
% predicted ranking scores for each sample.

for j = 1:length(predicts)
    % We don't want to use the variables that are used for training so
    % let's skip those in test, but include all of the unseen variables.
    if used_for_training(j) == 0 || ismember(ground_truth(j),unseen) 
        
        % For each of the categories find the guassian probability of the
        % each variable and each point
        best_prob = 0;
        for k = 1:size(means,3)
            
            % Add a bit of value to the Covariances to ensure they are
            % positive semi-definite, for this calculation.
            Cov_ex = Covariances(:,:,k) + eye(size(Covariances,1)).*.00001;
            prob = mvnpdf(predicts(j,:),means(:,:,k),Cov_ex);
            
            if prob > best_prob
                best_prob = prob;
                label_guess = k;
            end
        end
        
        % Now see if the label is the same as the ground truth label;
        if ground_truth(j) == label_guess;
            correct = correct + 1;
        end
        
        % Add to the total numbers of predicts that are analyzed
        total = total + 1;
    end
end

accuracy = correct/total;
    