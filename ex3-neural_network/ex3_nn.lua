---- Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

--  Instructions
--  ------------
-- 
--  This file contains code that helps you get started on the
--  linear exercise. You will need to complete the following functions 
--  in this exericse:
--
--     lrCostFunction.m (logistic regression cost function)
--     oneVsAll.m
--     predictOneVsAll.m
--     predict.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--
package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method3"
local plot = require"gnuplot"
local optim = require"optim"

---- Initialization
misc.clear_screen()

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

---- ================ Part 2: Loading Pameters ================
-- In this part of the exercise, we load some pre-initialized 
-- neural network parameters.

misc.printf('\nLoading Saved Neural Network Parameters ...\n')

-- Load the weights into variables Theta1 and Theta2
rlt = loader.load_from_mat('ex3weights.mat');
local Theta1 = rlt.Theta1
local Theta2 = rlt.Theta2

---- ================= Part 3: Implement Predict =================
--  After training the neural network, we would like to use it to predict
--  the labels. You will now implement the "predict" function to use the
--  neural network to predict the labels of the training set. This lets
--  you compute the training set accuracy.

local pred = method.predict(Theta1, Theta2, X)

misc.printf('\nTraining Set Accuracy: %f\n', 
    torch.eq(pred, y):double():mean() * 100)

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--  To give you an idea of the network's output, you can also run
--  through the examples one at the a time to see what it is predicting.

--  Randomly permute examples
local rp = torch.randperm(m)
for i = 1, m, 1 do
    local idx = rp[i]
    local this_row = X[{{idx}, {}}]
    -- Display 
    misc.printf('\nDisplaying Example Image\n');
    method.display_data(this_row, nil, false, "example_img.png")

    pred = method.predict(Theta1, Theta2, this_row)
    local num = pred[1][1]
    misc.printf('\nNeural Network Prediction: %d (digit %d)\n', num, num % 10);


    local s = misc.input('Paused - press enter to continue, q to exit:');
    if s == 'q' then
        break
    end
end
