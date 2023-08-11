---- Machine Learning Online Class - Exercise 4 Neural Network Learning

--  Instructions
--  ------------
-- 
--  This file contains code that helps you get started on the
--  linear exercise. You will need to complete the following functions 
--  in this exericse:
--
--     sigmoidGradient.m
--     randInitializeWeights.m
--     nnCostFunction.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--
package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method4"
local plot = require"gnuplot"
local optim = require"optim"
local nn = require"nn"

---- Initialization
misc.clear_screen()

---- Setup the parameters you will use for this exercise
local input_layer_size  = 400;  -- 20x20 Input Images of Digits
local hidden_layer_size = 25;   -- 25 hidden units
local num_labels = 10;          -- 10 labels, from 1 to 10   
                          -- (note that we have mapped "0" to label 10)

---- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset. 
--  You will be working with a dataset that contains handwritten digits.
--

-- Load Training Data
misc.printf('Loading and Visualizing Data ...\n')

local rlt = loader.load_from_mat('ex4data1.mat'); -- training data stored in arrays X, y
local X = rlt.X
local y = rlt.y
local m = X:size(1)

-- Randomly select 100 data points to display
local rand_indices = torch.randperm(m):long()
local sel = X:index(1, rand_indices[{{1, 100}}])

method.display_data(sel)

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- ================ Part 2: Loading Parameters ================
-- In this part of the exercise, we load some pre-initialized 
-- neural network parameters.

misc.printf('\nLoading Saved Neural Network Parameters ...\n')

-- Load the weights into variables Theta1 and Theta2
local rlt = loader.load_from_mat('ex4weights.mat')
local Theta1 = rlt.Theta1:clone()   -- Clone it in case of rlt.Theta being not Contiguous
local Theta2 = rlt.Theta2:clone()

-- Unroll parameters 
local nn_params = torch.cat(torch.Tensor(Theta1:storage()),
    torch.Tensor(Theta2:storage()))

---- ================ Part 3: Compute Cost (Feedforward) ================
--  To the neural network, you should first start by implementing the
--  feedforward part of the neural network that returns the cost only. You
--  should complete the code in nnCostFunction.m to return cost. After
--  implementing the feedforward to compute the cost, you can verify that
--  your implementation is correct by verifying that you get the same cost
--  as us for the fixed debugging parameters.
--
--  We suggest implementing the feedforward cost *without* regularization
--  first so that it will be easier for you to debug. Later, in part 4, you
--  will get to implement the regularized cost.
--
misc.printf('\nFeedforward Using Neural Network ...\n')

-- Weight regularization parameter (we set this to 0 here).
local lambda = 0;

local J = method.nn_cost_function(nn_params, input_layer_size, hidden_layer_size,
                   num_labels, X, y, lambda)

misc.printf('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n', J);

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

---- =============== Part 4: Implement Regularization ===============
--  Once your cost function implementation is correct, you should now
--  continue to implement the regularization with the cost.
--

misc.printf('\nChecking Cost Function (w/ Regularization) ... \n')

-- Weight regularization parameter (we set this to 1 here).
lambda = 1;

J = method.nn_cost_function(nn_params, input_layer_size, hidden_layer_size, 
                   num_labels, X, y, lambda);

misc.printf('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)\n', J);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


---- ================ Part 5: Sigmoid Gradient  ================
--  Before you start implementing the neural network, you will first
--  implement the gradient for the sigmoid function. You should complete the
--  code in the sigmoidGradient.m file.
--

misc.printf('\nEvaluating sigmoid gradient...\n')

local g = method.sigmoid_gradient(torch.Tensor{-1, -0.5, 0, 0.5, 1})
misc.printf('Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:\n  ');
print(g)
misc.printf('\n\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


---- ================ Part 6: Initializing Pameters ================
--  In this part of the exercise, you will be starting to implment a two
--  layer neural network that classifies digits. You will start by
--  implementing a function to initialize the weights of the neural network
--  (randInitializeWeights.m)

misc.printf('\nInitializing Neural Network Parameters ...\n')

local initial_Theta1 = method.rand_initialize_weight(input_layer_size, hidden_layer_size);
local initial_Theta2 = method.rand_initialize_weight(hidden_layer_size, num_labels);

-- Unroll parameters
local initial_nn_params = torch.cat(torch.Tensor(initial_Theta1:storage()),
    torch.Tensor(initial_Theta2:storage()))


---- =============== Part 7: Implement Backpropagation ===============
--  Once your cost matches up with ours, you should proceed to implement the
--  backpropagation algorithm for the neural network. You should add to the
--  code you've written in nnCostFunction.m to return the partial
--  derivatives of the parameters.
--
misc.printf('\nChecking Backpropagation... \n');

--  Check gradients by running checkNNGradients
method.check_nn_gradients()

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()


---- =============== Part 8: Implement Regularization ===============
--  Once your backpropagation implementation is correct, you should now
--  continue to implement the regularization with the cost and gradient.
--

misc.printf('\nChecking Backpropagation (w/ Regularization) ... \n')

--  Check gradients by running checkNNGradients
lambda = 3;
method.check_nn_gradients(lambda)

-- Also output the costFunction debugging values
local debug_J  = method.nn_cost_function(nn_params, input_layer_size, 
    hidden_layer_size, num_labels, X, y, lambda);

misc.printf('\n\nCost at (fixed) debugging parameters (w/ lambda = %f): %f \n(for lambda = 3, this value should be about 0.576051)\n\n', lambda, debug_J);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


---- =================== Part 8: Training NN ===================
--  You have now implemented all the code necessary to train a neural 
--  network. To train your neural network, we will now use "fmincg", which
--  is a function which works similarly to "fminunc". Recall that these
--  advanced optimizers are able to train our cost functions efficiently as
--  long as we provide them with the gradient computations.
--
misc.printf('\nTraining Neural Network... \n')

--  After you have completed the assignment, change the MaxIter to a larger
--  value to see how more training helps.
local options = {
    maxIter = 100,
}

--  You should also try different values of lambda
lambda = 1;

-- Create "short hand" for the cost function to be minimized
local cost_function = function(param)
    local cost, grad = method.nn_cost_function(param, input_layer_size,
        hidden_layer_size, num_labels, X, y, lambda)
    return cost, grad
end

-- Now, costFunction is a function that takes in only one argument (the
-- neural network parameters)
nn_params = method.fmincg(cost_function, initial_nn_params, options, 10)

-- Obtain Theta1 and Theta2 back from nn_params
Theta1 = nn_params[{{1, hidden_layer_size * (input_layer_size + 1)}}]
    :view(hidden_layer_size, input_layer_size + 1) -- 25 401
Theta2 = nn_params[{{hidden_layer_size * (input_layer_size + 1) + 1, -1}}]
    :view(num_labels, hidden_layer_size + 1) -- 10 26

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


---- ================= Part 9: Visualize Weights =================
--  You can now "visualize" what the neural network is learning by 
--  displaying the hidden units to see what features they are capturing in 
--  the data.

misc.printf('\nVisualizing Neural Network... \n')

method.display_data(Theta1[{{}, {2, -1}}], nil, nil, "visualizing_hidden_layer.png")

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()

---- ================= Part 10: Implement Predict =================
--  After training the neural network, we would like to use it to predict
--  the labels. You will now implement the "predict" function to use the
--  neural network to predict the labels of the training set. This lets
--  you compute the training set accuracy.

pred = method.predict(Theta1, Theta2, X);

misc.printf('\nTraining Set Accuracy: %f\n', 
    torch.mean(torch.eq(pred, y):double()) * 100);

-----------------
-- extra: train nn using nn lib

local model = nn.Sequential()
model:add(nn.Linear(input_layer_size, hidden_layer_size))
model:add(nn.ReLU())
model:add(nn.Linear(hidden_layer_size, num_labels))
model:add(nn.LogSoftMax())

local y2_tbl = {}
for i = 1, num_labels, 1 do
    y2_tbl[i] = torch.eq(y, i):double()
end
local y2 = torch.cat(y2_tbl, 2) -- 5000 10

local ceriterion = nn.ClassNLLCriterion()
local m_params, m_grads = model:getParameters()
local f_eval = function(params)
    m_grads:zero() 
    local outputs = model:forward(X)
    local loss = ceriterion:forward(outputs, y:view(y:numel()))

    local dloss_doutputs = ceriterion:backward(outputs, y:view(y:numel()))
    model:backward(X, dloss_doutputs)

    local l2 = 0
    for _, m in pairs(model:listModules()) do
        if m.weight then
            l2 = l2 + m.weight:norm() ^ 2
            m.gradWeight:add(lambda / X:size(1), m.weight)
        end
    end
    loss = loss + lambda * l2 / (2 * m)

    return loss, m_grads
end
optim.cg(f_eval, m_params, {maxIter = 100})

local pred2 = model:forward(X)
_, pred2 = torch.max(pred2, 2)

misc.printf('\nTraining Set Accuracy: %f\n', 
    torch.mean(torch.eq(pred2:double(), y):double()) * 100);
--
---------------




