function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

X = [ones(m, 1), X];

a1 = X;

z2 = a1*Theta1';
% add a bias node to each training example
a2 = [ones(m, 1), sigmoid(z2)];

z3 = a2*Theta2';
% a3 is a vector that contains the probability for each label.
a3 = sigmoid(z3);

% returns the maximum value (highest prediction) from the output layer a3
[maxval, indices] = max(a3, [], 2);
p = indices;

end
