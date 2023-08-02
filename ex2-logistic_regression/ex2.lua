package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method2"
local plot = require"gnuplot"
local optim = require"optim"

misc.clear_screen()

-- Load Data
-- The first two columns contains the exam scores and the third column
-- contains the label.

local data = loader.load_from_txt('ex2data1.txt');
local X = data[{{}, {1, 2}}];   -- n x 2
local y = data[{{}, {3, 3}}];   -- n x 1

-- ==================== Part 1: Plotting ====================
-- We start the exercise by first plotting the data to understand the 
-- the problem we are working with.

misc.printf('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n');

method.plot_data(X, y);

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

-- ============ Part 2: Compute Cost and Gradient ============
--  In this part of the exercise, you will implement the cost and gradient
--  for logistic regression. You neeed to complete the code in 
--  costFunction.m

--  Setup the data matrix appropriately, and add ones for the intercept term
local m = X:size(1)
local n = X:size(2)

-- Add intercept term to x and X_test
X = torch.cat(torch.ones(m), X, 2)

-- Initialize fitting parameters
local initial_theta = torch.zeros(n + 1)

-- Compute and display initial cost and gradient
local cost, grad = method.cost_function(initial_theta, X, y)

misc.printf('Cost at initial theta (zeros): %f\n', cost);
misc.printf('Expected cost (approx): 0.693\n');
misc.printf('Gradient at initial theta (zeros): \n');
misc.printf(tostring(grad))
misc.printf('Expected gradients (approx):\n -0.1000\n -12.0092\n -11.2628\n');

-- Compute and display cost and gradient with non-zero theta
local test_theta = torch.Tensor({-24; 0.2; 0.2})
cost, grad = method.cost_function(test_theta, X, y)

misc.printf('\nCost at test theta: %f\n', cost);
misc.printf('Expected cost (approx): 0.218\n');
misc.printf('Gradient at test theta: \n');
misc.printf(tostring(grad))
misc.printf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

-- ============= Part 3: Optimizing using fminunc  =============
--  In this exercise, you will use a built-in function (fminunc) to find the
--  optimal parameters theta.

local config = {
    maxIter = 400,
    --learningRate = 0.01,
}
local feval = function(theta)
    local cost, grad = method.cost_function(theta, X, y)
    return cost, grad
end
--  obtain the optimal theta
local theta, all_fs = optim.cg(feval, initial_theta, config)
--local theta, all_fs = optim.lbfgs(feval, initial_theta, config)
local cost = all_fs[#all_fs]

-- Print theta to screen
misc.printf('Cost at theta found by fminunc: %f\n', cost);
misc.printf('Expected cost (approx): 0.203\n');
misc.printf('theta: \n');
misc.printf(tostring(theta))
misc.printf('Expected theta (approx):\n');
misc.printf(' -25.161\n 0.206\n 0.201\n');

-- Plot Boundary
method.plot_decision_boundary(theta, X, y);

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

-- ============== Part 4: Predict and Accuracies ==============
--  After learning the parameters, you'll like to use it to predict the outcomes
--  on unseen data. In this part, you will use the logistic regression model
--  to predict the probability that a student with score 45 on exam 1 and 
--  score 85 on exam 2 will be admitted.

--  Furthermore, you will compute the training and test set accuracies of 
--  our model.

--  Your task is to complete the code in predict.m

--  Predict probability for a student with score 45 on exam 1 
--  and score 85 on exam 2 

local prob = torch.sigmoid(torch.dot(torch.Tensor({1, 45, 85}), theta))
misc.printf('For a student with scores 45 and 85, we predict an admission probability of %f\n', prob);
misc.printf('Expected value: 0.775 +/- 0.002\n\n');

-- Compute accuracy on our training set
local p = method.predict(theta, X)

misc.printf('Train Accuracy: %f\n', 
    torch.eq(p, y):double():mean() * 100);
misc.printf('Expected accuracy (approx): 89.0\n');
misc.printf('\n');


