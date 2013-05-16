% This script makes the desired Figures
figure()
theirs = [57 57 55 56 55 55];
mine = [55 54 52 40 40 30];
x = [6 5 4 3 2 1];
plot(x,theirs,'r-',x,mine,'b');
xlabel('# of Attributes Used')
ylabel('Classification Accuracy (%)')
title('OSR -- # of Attributes Used ');
axis([0 11 0 80]);
set(gca,'XTick',[1:11]);
set(gca,'XDir','Reverse');
legend('Paper Results','My Results');