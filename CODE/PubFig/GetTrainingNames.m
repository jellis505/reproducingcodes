% Get the images that will be used for training
length_of_names = length(im_names);   
names_index = 1;
for j = 1:length_of_names
    if used_for_training(j) == 1;
        names_train(names_index,1) = im_names(j);
        names_train(names_index,2) = j;
        names_index = names_index + 1;
    end
end
disp(names_train);