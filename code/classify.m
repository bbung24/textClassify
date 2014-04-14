function [c] = classify(XTrain_fName, yTrain_fName, XTest_fName)
% This program implements the naive bayes classifier
% based on the tf-idf ranks of the XTest and XTrain 
% text files.

% Read the CSV files
XTrain = csvread(XTrain_fName);
XTest = csvread(XTest_fName);
yTrain = csvread(yTrain_fName);

% ------ REPLACE WITH YOUR CODE ------
[row col] = size(XTrain);
u = unique(yTrain);
nClasses = numel(u);

% Calculate the mean and the standard deviation
for c = 1:nClasses
    mu(c,:) = mean(XTrain(yTrain == u(c),:));        
    sigma(c,:) = std(XTrain(yTrain == u(c),:),1); 
end

% Calculate the normpdf for all the rows
for i=1:row
    one = (log(normpdf(XTest(i,:),mu(1,:),sigma(1,:))));
    two = (log(normpdf(XTest(i,:),mu(2,:),sigma(2,:))));
    three = (log(normpdf(XTest(i,:),mu(3,:),sigma(3,:))));
    four = (log(normpdf(XTest(i,:),mu(4,:),sigma(4,:))));
    
    z(i,1) = sum(one);
    z(i,2) = sum(two);
    z(i,3) = sum(three);
    z(i,4) = sum(four);
end

% Find the max probability of the 4 possible classes
[t,classPred] = max(z,[],2);
c = classPred -1;

end
