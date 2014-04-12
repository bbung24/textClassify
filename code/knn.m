function [ D ] = knn( XTrain, XTest,k )
% Filler code - replace with your code
fprintf('Knn start\n');
nTest = size(XTest,1);
nTrain = size(XTrain, 1);
D = zeros(nTest,k);
distance = zeros(nTrain,2);
for i = 1:nTest
    fprintf('Knn-i: %d\n',i);
	for j = 1:nTrain
        fprintf('Knn-j: %d\n',j);
		distance(j,1) = norm(XTest(i,:) - XTrain(j,:));
		distance(j,2) = j;
	end
	temp = sortrows(distance);
	for l = 1:k
        fprintf('Knn-k: %d\n', k);
		D(i,l) = temp(l,2);
	end
end
end

