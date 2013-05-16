load('data.mat')

x = 1:1:8;
hold on;
for j = 1:6
    subplot(3,2,j);
    r = relative_ordering(j,:);
    plot(x,r);
    xlabel('Category Number');
    ylabel('Ranking');
    axis tight
end

suptitle('Ground Truth Relative Ranking');