package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method"
local plot = require"gnuplot"

local table_insert = table.insert

-- ================ Part 1: Feature Normalization ================
-- Clear and Close Figures
misc.clear_screen()

misc.printf('Loading data ...\n');

-- Load Data
local data = loader.load_from_txt('ex1data2.txt')
local X = data[{{}, {1,2}}]
local y = data[{{}, 3}]
local m = y:size(1)

-- Print out some data points
misc.printf('First 10 examples from the dataset: \n');
misc.printf(' x = %s, y = %s \n', 
    tostring(X[{{1, 10}, {}}]), 
    tostring(y[{{1, 10}}]))

misc.printf('Program paused. Press enter to continue.\n')
misc.pause()

-- Scale features and set them to zero mean
misc.printf('Normalizing Features ...\n');

local mu, sigma
X, mu, sigma = method.feature_normalize(X)

-- Add intercept term to X
X = torch.cat(torch.ones(m), X, 2)

-- ================ Part 2: Gradient Descent ================

misc.printf('Running gradient descent ...\n');

-- Choose some alpha value
local al = {0.01,0.03,0.1,0.3,1}
local num_iters = 400;
local plotstyle ={'r','g','b','y','k'};

local lines = {}
local theta
for _, alpha in ipairs(al) do
    theta = torch.zeros(3, 1)
    local J_history
    theta, J_history = method.gradient_descent(X, y, theta, alpha, num_iters, true)
    table_insert(lines, {"alpha = ".. alpha, torch.range(1, #J_history), torch.Tensor(J_history)})
end

plot.pngfigure("multi_alpha.png")
plot.plot(lines)
plot.xlabel("Number of iterations")
plot.ylabel("Cost J")
plot.plotflush()

-- Display gradient descent's result
misc.printf('Theta computed from gradient descent: \n')
misc.printf(tostring(theta))

-- Estimate the price of a 1650 sq-ft, 3 br house
local tmp = torch.cat(torch.ones(1), torch.cdiv(torch.Tensor({1650, 3}) - mu, sigma))
local price = torch.dot(tmp, theta)
misc.printf('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent):\n $%f\n', price);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

-- ================ Part 3: Normal Equations ================

misc.printf('Solving with normal equations...\n');

-- Load Data
data = loader.load_from_txt('ex1data2.txt')
X = data[{{}, {1,2}}]
y = data[{{}, 3}]
m = y:size(1)

-- Add intercept term to X
X = torch.cat(torch.ones(m), X, 2)

-- Calculate the parameters from the normal equation
theta = method.normal_eqn(X, y);

-- Display normal equation's result
misc.printf('Theta computed from the normal equations: \n');
misc.printf(tostring(theta))

-- Estimate the price of a 1650 sq-ft, 3 br house
price = torch.dot(torch.Tensor({1, 1650, 3}), theta)

misc.printf('Predicted price of a 1650 sq-ft, 3 br house (using normal equations):\n $%f\n', price);