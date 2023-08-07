-- Machine Learning Online Class - Exercise 1: Linear Regression
--
-- Instructions
-- ------------
--
-- This file contains code that helps you get started on the
-- linear exercise. 

package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method"
local plot = require"gnuplot"

misc.clear_screen()

-- ==================== Part 1: Basic Function ====================
misc.printf("Running warmUpExercise ... ")
misc.printf("5x5 Identity Matrix: ")

method.warm_up_exercise()

misc.printf("Program paused. Press enter to continue.")
misc.pause()

-- ======================= Part 2: Plotting =======================
misc.printf('Plotting Data ...')

local data = loader.load_from_txt("ex1data1.txt")
local X = data[{{}, 1}]
local y = data[{{}, 2}]
local m = y:size(1)

method.plot_data(X, y)

misc.printf('Program paused. Press enter to continue.');
misc.pause()

-- =================== Part 3: Cost and Gradient descent ===================

X = torch.cat(torch.ones(m), X, 2) -- Add a column of ones to x
local theta = torch.zeros(2, 1) --initialize fitting parameters

-- Some gradient descent settings
local iterations = 1500
local alpha = 0.01

misc.printf("\nTesting the cost function ...")
-- compute and display initial cost
local J = method.compute_cost(X, y, theta)
misc.printf('With theta = [0 ; 0]\nCost computed = %f\n', J)
misc.printf('Expected cost value (approx) 32.07\n')

-- further testing of the cost function
J = method.compute_cost(X, y, torch.Tensor({-1, 2}))
misc.printf('\nWith theta = [-1 ; 2]\nCost computed = %f\n', J)
misc.printf('Expected cost value (approx) 54.24\n')

misc.printf('Program paused. Press enter to continue.\n')
misc.pause()

misc.printf('\nRunning Gradient Descent ...\n')
-- run gradient descent
theta = method.gradient_descent(X, y, theta, alpha, iterations)

-- print theta to screen
misc.printf('Theta found by gradient descent:\n');
misc.printf(tostring(theta));
misc.printf('Expected theta values (approx)\n');
misc.printf(' -3.6303\n  1.1664\n\n');

-- Plot the linear fit
plot.plot({"Training data", X[{{}, 2}], y, "+"},
    {"Linear regression", X[{{}, 2}], X * theta, "-"})
plot.plotflush()

-- Predict values for population sizes of 35,000 and 70,000
local predict1 = torch.dot(torch.Tensor({1, 3.5}), theta)
misc.printf('For population = 35,000, we predict a profit of %f\n', predict1*10000)
local predict2 = torch.dot(torch.Tensor({1, 7}), theta)
misc.printf('For population = 70,000, we predict a profit of %f\n', predict2*10000)

misc.printf('Program paused. Press enter to continue.\n')
misc.pause()

-- ============= Part 4: Visualizing J(theta_0, theta_1) =============
misc.printf('Visualizing J(theta_0, theta_1) ...\n')

-- Grid over which we will calculate J
local theta0_vals = torch.linspace(-10, 10, 100);
local theta1_vals = torch.linspace(-1, 4, 100);
local val_size = theta0_vals:size(1) * theta1_vals:size(1)
local x_vals = torch.zeros(val_size)
local y_vals = torch.zeros(val_size)

-- initialize J_vals to a matrix of 0's
local J_vals = torch.Tensor(theta0_vals:size(1), theta1_vals:size(1))
local n = 0
for i = 1, theta0_vals:size(1), 1 do
    for j = 1, theta1_vals:size(1), 1 do
        n = n + 1
        x_vals[n] = theta0_vals[i]
        y_vals[n] = theta1_vals[j]
        J_vals[i][j] = method.compute_cost(X, y, torch.Tensor({theta0_vals[i], theta1_vals[j]}))
    end
end
J_vals = J_vals:t():reshape(1, val_size)

-- Surface plot
plot.close()

plot.pngfigure("surface.png")
plot.splot(x_vals:view(1, val_size), y_vals:view(1, val_size), J_vals)
plot.xlabel("theta_0")
plot.ylabel("theta_1")
plot.plotflush()

plot.close()