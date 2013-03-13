% This function takes in a matrix with attribute relations in numeric
% categories, and the images that will be used for trainings indices, and
% class labels of all of the images, and outputs the O and S matrix used
% with rank_with_sim rank svm implementation for training
% Created by Joe Ellis for Reproducible Codes class

function [O,S] = Create_O_and_S_Mats(category_order,used_for_training,class_labels,num_classes)

% variables
% category_order = the order of the relative attributes of each category.
%   The matrix is saved and can be loaded by loading category_order_osr.mat
% used_for_training = A vector the length of all of the features, and this
%   is used to solve for the features that will be used to create the 
%   ranking function
% class_labels = the class_labels of each image and corresponds to
%   extracted features
num_categories = 6;

train_by_class = zeros(num_classes,30);

% Create the train_by_class matrix to create the o and s matrix for ranking
% functions
index = ones(1,num_classes);
for j = 1:length(used_for_training)
    
    % pick out the images that are going to e used_for_training
    if used_for_training(j) == 1;
        switch class_labels(j)
            case 1
                train_by_class(1,index(1)) = j;
                index(1) = index(1) + 1;
            case 2
                train_by_class(2,index(2)) = j;
                index(2) = index(2) + 1;
            case 3
                train_by_class(3,index(3)) = j;
                index(3) = index(3) + 1;
            case 4
                train_by_class(4,index(4)) = j;
                index(4) = index(4) + 1;
            case 5
                train_by_class(5,index(5)) = j;
                index(5) = index(5) + 1;
            case 6
                train_by_class(6,index(6)) = j;
                index(6) = index(6) + 1;
            case 7
                train_by_class(7,index(7)) = j;
                index(7) = index(7) + 1;
            case 8
                train_by_class(8,index(8)) = j;
                index(8) = index(8) + 1;
        end
    end
end

% Now we have the train_by_class matrix which have the training images for
% each seperate variable.  Now we are going to write the code as to how we
% are going to create the o matrix.

% Each category should be compared to itself and should have relatively
% similar attributes as well.  Therefore, the first four elements in each
% row will be compared to themselves and used as the first four elements of
% the Similarity matrix

num_images = 2688;
s_index = ones(1,num_categories);
o_index = ones(1,num_categories);
for l = 1:num_categories;
    for j = 1:num_classes
            S_row = zeros(1,num_images);
            S_row(train_by_class(j,1)) = 1;
            S_row(train_by_class(j,2)) = -1;
            S(s_index(l),:,l) = S_row;
            s_index(l) = s_index(l) + 1;
    end
end

% Now comes the really complicated part... Start at 4 for all of these, and
% then iterate through the matrix and based on how the class labels look
% compared to each other we will add rows to either the O matrix or the S
% matrix for training of Rank SVM

% Do this for every visible category that we have
for l = 1:6
    % Every 4 variables we will need to move forward in the matrix here so
    %   we then have to move forward by 4 on each iteration of the category
    on_class = 1;
    for j = 3:4:27 % This section uses different image pairs for training
        for k = on_class+1:8
            % If the two relative comparisons are equal
            if category_order(l,on_class) == category_order(l,k)
                for i = 0:3
                    S_row = zeros(1,num_images);
                    S_row(train_by_class(on_class,j+i)) = 1;
                    S_row(train_by_class(k,j+i)) = -1;
                    S(s_index(l),:,l) = S_row;
                    s_index(l) = s_index(l) + 1;
                end
            
            % If the relative comparison of the new class is greater than
            % that of the compared class
            elseif category_order(l,on_class) > category_order(l,k)
                for i = 0:3
                    O_row = zeros(1,num_images);
                    O_row(train_by_class(on_class,j+i)) = 1;
                    O_row(train_by_class(k,j+i)) = -1;
                    O(o_index(l),:,l) = O_row;
                    o_index(l) = o_index(l) + 1;
                end
                
           % If the relative comparison of the new class is greater than
           % that of the compared class
            elseif category_order(l,on_class) < category_order(l,k)
                for i = 0:3
                    O_row = zeros(1,num_images);
                    O_row(train_by_class(on_class,j+i)) = -1;
                    O_row(train_by_class(k,j+i)) = 1;
                    O(o_index(l),:,l) = O_row;
                    o_index(l) = o_index(l) + 1;
                end
            end
        end
        on_class = on_class + 1;
    end
end
    

