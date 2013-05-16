% Generate mean and covariance matrix for each categories relative scores.
% Created by Joe Ellis -- PhD Candidate Columbia University

function [means, Covariances] = meanandvar_forcat(Training_Samples,unseen,category_order,num_classes, looseness_constraint)

% INPUTS
% Training_Samples = the training samples for each category this is a 2-d
%   Matrix
% unseen = A vector containing the unseen categories
% category_order = The ordering of the classes based on attribute, 2-D
%   matrix
% num_classes = The number of classes that are present in this example
% looseness_constraint = The losseness constraint as specified in the
%   paper.

% OUTPUTS
% means = 3-d matrix each depth is a mean of the labels should be 1x8x6 rows
% Covariances = 3-d matrix.  Should be 6x6x8.

% The looseness constraint should be the looseness-1 in the way that I am
% applying it in the algorithm below, however the looseness constraint in 
% the paper is described this way.
looseness_constraint = looseness_constraint - 1;

% means of the set ups
% Create the list of seen categories
seen = [];
seen_index = 1;
for z = 1:num_classes
    if ismember(z,unseen) == 0
        seen(seen_index) = z;
        seen_index = seen_index + 1;
    end
end

% now we have the seen categories, and we want to find the mean and
% covariance of each of these values.

% set up the means and covariance matrices that we want to find in this
% function
means = zeros(1,size(Training_Samples,2),size(Training_Samples,3));
Covariances = zeros(size(Training_Samples,2),size(Training_Samples,2),size(Training_Samples,3));

% Get the means and covariance of the values 
for k = 1:length(seen)
    
    % Get the seen variable index
    class = seen(k);
    
    % Find the means of the seen  classes
    means(:,:,class) = mean(Training_Samples(:,:,class));
    
    % Find the covariance of the samples
    Covariances(:,:,class) = cov(Training_Samples(:,:,class));
end

% Find the average covariance matrix across all of the seen classes
AVG_COV = sum(Covariances,3)/length(seen);

% Now we have to find the average distance between the means 
dm = zeros(1,size(category_order,1));

for j = 1:size(category_order,1)
    % This section finds the means and sorts the average distance between
    % the neightbors in a sorted list of these means
    sorted_means = sort(nonzeros(means(1,j,:)));
    diff = 0;
    for z = 1:length(sorted_means)-1
        diff = diff + abs(sorted_means(z)-sorted_means(z+1));
    end
    dm(j) = diff/(length(seen)-1);
end

% We need to create a category ordering of only the categories available
% not the unseen categories
for j = 1:length(seen)
    there = seen(j);
    new_category_order(:,j) = category_order(:,there);
end
        
% Now loop through the number of unseen variables that we have and solve
% for their distributions given the seen categories 
for k = 1:length(unseen)
    % This is the unseen class
    class = unseen(k);
   
    % now we have to go through every attribute for this unseen
    % class and then derive it's given distribution using the rules
    % described in the Relative Attribute Paper.
    for j = 1:size(new_category_order,1)
        
        % Get the attribute rank of the curren unseen class
        attr_rank = category_order(j,class);
        
        % Now get the max and min values and indices for that particular 
        % attribute
        [max_rank max_idx] = max(new_category_order(j,:));
        [min_rank min_idx] = min(new_category_order(j,:));
            
        % Check to see if this attribute rank is larger than any other seen
        % attribute level
        if attr_rank > max_rank
            % Here we find the max_mean for the given attribute, and then
            % add the average distance between the seen means to this
            % value.
            max_mean = means(1,j,seen(max_idx(1)));
            means(1,j,class) = max_mean + dm(j);
            
        % Check to see if the attribute rank is the same as the max_rank
        % for this attribute.
        elseif attr_rank == max_rank
            % If the attribute rank is the same as the max_rank find the
            % closest seen class to this value, we do this by subtracting 
            new_rank = attr_rank;
            while 1
                new_rank = new_rank - 1;
                idx = find(new_category_order(j,:) == new_rank);
                if isempty(idx) == 0
                    one_less_mean = means(1,j,seen(idx(1)));
                    means(1,j,class) = one_less_mean + dm(j);
                    break;
                % If for some reason there are NO seen classes with the
                % attribute ranks less than our unseen class, this could
                % happen if very few classes are seen, then simply apply
                % the mean of the max_rank to this mean.
                elseif new_rank < 0
                   max_mean = means(1,j,seen(max_idx(1)));
                   means(1,j,class) = max_mean;
                   break;
                end
            end
        
        % Check to see if this attribute rank is smaller than any other seen
        % attribute level
        elseif attr_rank < min_rank
            % Now get the smallest attribute mean for this attribute and
            % subtract the average distance between sorted means for this
            % attribure
             min_mean = means(1,j,seen(min_idx(1)));
             means(1,j,class) = min_mean - dm(j);
             
        elseif attr_rank == min_rank
            % If the attribute rank is the same as the min_rank find the
            % closest seen class to this value, we do this by adding one 
            new_rank = attr_rank;
            while 1
                new_rank = new_rank + 1;
                idx = find(new_category_order(j,:) == new_rank);
                if isempty(idx) == 0
                    one_more_mean = means(1,j,seen(idx(1)));
                    means(1,j,class) = one_more_mean - dm(j);
                    break;
                % If for some reason there are NO seen classes with the
                % attribute ranks more than our unseen class (this could
                % happen if very few classes are seen) then simply apply
                % the mean of the min_rank to this mean.
                elseif new_rank > 10
                   min_mean = means(1,j,seen(min_idx(1)));
                   means(1,j,class) = min_mean;
                   break;
                end
            end
            
        else
            
            % If the given attribute is not on the boundaries of the seen
            % ranking then we can compare it to multiple elements, and we
            % want to create the mean by choosing the closest seen class
            % above and below the unseen class in attribute rank and
            % average the two values for this attribute rank.
            
            % This section gets the closest lower mean to the unseen class
            % for this attribute.
            
            % Get the seen classes for an attribute in a vector
            row_vec = new_category_order(j,:);
            % This finds all of the values that have a lower rank than the
            % given attribute rank, the looseness constraint can be changed
            % to find candidates that are farther away from the attribute
            % rank.
            min_cand = row_vec < attr_rank - looseness_constraint;
            value = 0;
            min_use_index = 1;
            
            % This for loop returns the index of the seen class attribute
            % that is closest to, but below the unseen attribute rank
            for a = 1:length(min_cand)
                if min_cand(a) == 1
                    if row_vec(a) > value;
                        min_use_index = a;
                        value = row_vec(a);
                    end
                end
            end
            lower_u = means(1,j,seen(min_use_index));
            
            % This section gets the closest higher mean to the unseen class
            % for this attribute

            
            % This finds all of the values that have a higher rank than the
            % given attribute rank, the looseness constraint can be changed
            % to find candidates that are farther away from the attribute
            % rank.
            max_cand = row_vec > attr_rank + looseness_constraint;
            value = 9; % This is larger than any possible attribute rank
            max_use_index = 1;
            
            % This for loop returns the index of the seen class attribute
            % that is closest to, but below the unseen attribute rank
            for a = 1:length(max_cand)
                if max_cand(a) == 1
                    if row_vec(a) < value;
                        max_use_index = a;
                        value = row_vec(a);
                    end
                end
            end
            higher_u = means(1,j,seen(max_use_index));
            
            % This solves for the mean for this unseen class given the 
            % closest seen class attribute scores
            means(1,j,class) = (higher_u + lower_u)/2;
        end
        
        % Give the unseen class the average covariance.
        Covariances(:,:,class) = AVG_COV;
    end
end

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



