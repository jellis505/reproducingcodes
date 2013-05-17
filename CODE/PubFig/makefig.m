% This script makes the desired Figures
figure()
theirs = [70 65 65 60 57 40];
mine = [72 66 62 60 51 43];
x = [0 1 2 3 4 5];
plot(x,theirs,'r-',x,mine,'b');
xlabel('# of Unseen Classes')
ylabel('Classification Accuracy (%)')
title('Generative Framework Test');
axis([0 5 0 80]);
set(gca,'XTick',[1:5]);
legend('Paper Results','Their Learned Ranks Results');