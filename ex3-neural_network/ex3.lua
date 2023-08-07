-- Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
--
-- Instructions
-- ------------
--
-- This file contains code that helps you get started on the
-- linear exercise. 

package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method3"
local plot = require"gnuplot"
local optim = require"optim"

misc.clear_screen()

---- Setup the parameters you will use for this part of the exercise
local input_layer_size  = 400;  -- 20x20 Input Images of Digits
local num_labels = 10;          -- 10 labels, from 1 to 10
                          -- (note that we have mapped "0" to label 10)

---- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  You will be working with a dataset that contains handwritten digits.
--

-- Load Training Data
misc.printf('Loading and Visualizing Data ...\n')

local rlt = loader.load_from_mat('ex3data1.mat'); -- training data stored in arrays X, y
local X = rlt.X
local y = rlt.y
local m = X:size(1)

-- Randomly select 100 data points to display
local rand_indices = torch.randperm(m):long()
local sel = X:index(1, rand_indices[{{1, 100}}])

method.display_data(sel)

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- ============ Part 2a: Vectorize Logistic Regression ============
--  In this part of the exercise, you will reuse your logistic regression
--  code from the last exercise. You task here is to make sure that your
--  regularized logistic regression implementation is vectorized. After
--  that, you will implement one-vs-all classification for the handwritten
--  digit dataset.
--

-- Test case for lrCostFunction
misc.printf('\nTesting lrCostFunction() with regularization');

local theta_t = torch.Tensor({-2, -1, 1, 2})
local X_t = torch.cat(torch.ones(5, 1), torch.linspace(1, 15, 15):view(3, 5):t() / 10)
local y_t = torch.Tensor({1, 0, 1, 0, 1})
local lambda_t = 3
local J, grad = method.lr_cost_function(theta_t, X_t, y_t, lambda_t)

misc.printf('\nCost: %f\n', J);
misc.printf('Expected cost: 2.534819\n');
misc.printf('Gradients:\n');
print(grad);
misc.printf('Expected gradients:\n');
misc.printf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- ============ Part 2b: One-vs-All Training ============
misc.printf('\nTraining One-vs-All Logistic Regression...\n')

local lambda = 0.1
local all_theta = method.one_vs_all(X, y, num_labels, lambda)

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


---- ================ Part 3: Predict for One-Vs-All ================

local pred = method.predict_one_vs_all(all_theta, X)

misc.printf('\nTraining Set Accuracy: %f\n', 
    torch.eq(pred, y):double():mean() * 100)
