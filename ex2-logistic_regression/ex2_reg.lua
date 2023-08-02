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

local data = loader.load_from_txt('ex2data2.txt');
local X = data[{{}, {1, 2}}];   -- n x 2
local y = data[{{}, {3, 3}}];   -- n x 1

method.plot_data(X, y, false, {"y = 1", "y = 0"}, 
    {"Microchip Test 1", "Microchip Test 2"}, "chips.png");

-- =========== Part 1: Regularized Logistic Regression ============
--  In this part, you are given a dataset with data points that are not
--  linearly separable. However, you would still like to use logistic
--  regression to classify the data points.

--  To do so, you introduce more features to use -- in particular, you add
--  polynomial features to our data matrix (similar to polynomial
--  regression).

-- Add Polynomial Features

-- Note that mapFeature also adds a column of ones for us, so the intercept
-- term is handled
X = method.map_feature(X, 6)
print(X[1])

-- Initialize fitting parameters
local initial_theta = torch.zeros(X:size(2), 1)

-- Set regularization parameter lambda to 1
local lambda = 1;

-- Compute and display initial cost and gradient for regularized logistic
-- regression
local cost, grad = method.cost_function_reg(initial_theta, X, y, lambda)

misc.printf('Cost at initial theta (zeros): %f\n', cost);
misc.printf('Expected cost (approx): 0.693\n');
misc.printf('Gradient at initial theta (zeros) - first five values only:\n');
misc.printf(tostring(grad[{{1, 5}, {}}]))
misc.printf('Expected gradients (approx) - first five values only:\n');
misc.printf(' 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115\n');

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

-- Compute and display cost and gradient
-- with all-ones theta and lambda = 10
test_theta = torch.ones(X:size(2), 1);
cost, grad = method.cost_function_reg(test_theta, X, y, 10);

misc.printf('\nCost at test theta (with lambda = 10): %f\n', cost);
misc.printf('Expected cost (approx): 3.16\n');
misc.printf('Gradient at test theta - first five values only:\n');
misc.printf(tostring(grad[{{1, 5}, {}}]))
misc.printf('Expected gradients (approx) - first five values only:\n');
misc.printf(' 0.3460\n 0.1614\n 0.1948\n 0.2269\n 0.0922\n');

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

---- ============= Part 2: Regularization and Accuracies =============
--  Optional Exercise:
--  In this part, you will get to try different values of lambda and
--  see how regularization affects the decision coundart
--
--  Try the following values of lambda (0, 1, 10, 100).
--
--  How does the decision boundary change when you vary lambda? How does
--  the training set accuracy vary?
--

for _, lambda in ipairs({0, 1, 10, 100}) do
--for _, lambda in ipairs({1}) do
    misc.printf('Running with lambda = %f\n', lambda);

    -- Initialize fitting parameters
    initial_theta = torch.rand(X:size(2), 1)
    
    -- Set Options
    local config = {
        maxIter = 400,
    }
    --local config = {
    --    learningRate = 0.1,
    --    momentum = 0.9,
    --}
    local feval = function(t)
        local cost, grad = method.cost_function_reg(t, X, y, lambda)
        return cost, grad
    end
    
    -- Optimize
    local theta, loss = optim.cg(feval, initial_theta, config)
    --print(loss)
    --local theta = initial_theta
    --local loss
    --for epoch = 1, 400 do
    --    theta, loss  = optim.sgd(feval, theta, config)
    --end
    
    -- Plot Boundary
    method.plot_decision_boundary(theta:view(theta:size(1)), X, y, false, {"y = 1", "y = 0"}, 
        {"Microchip Test 1", "Microchip Test 2"}, 
        string.format("chips_reg_lambda_%s.png", lambda));
    
    -- Compute accuracy on our training set
    local p = method.predict(theta, X)
    
    misc.printf('Train Accuracy: %f\n', 
        torch.eq(p, y):double():mean() * 100);
    misc.printf('Expected accuracy (with lambda = 1): 83.1 (approx)\n');
end


