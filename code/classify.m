function [c] = classify(XTrain_fName, yTrain_fName, XTest_fName)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
fprintf('Classify start\n');
fprintf('csvread start');
clock
XTrain = csvread(XTrain_fName);
fprintf('csvread done', clock);
clock
XTest = csvread(XTest_fName);
yTrain = csvread(yTrain_fName);

% ------ REPLACE WITH YOUR CODE ------
k = 4;
nTest = size(XTest,1);
b = zeros(nTest, 1);
D = knn(XTrain, XTest, k);

for i = 1:nTest
	b(i) = mode(yTrain(D(i, :)));
end
c = b;
end


