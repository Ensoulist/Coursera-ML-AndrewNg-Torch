--% Machine Learning Online Class
--  Exercise 8 | Anomaly Detection and Collaborative Filtering
--
--  Instructions
--  ------------
--
--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:
--
--     estimateGaussian.m
--     selectThreshold.m
--     cofiCostFunc.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--
package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method8"
local plot = require"gnuplot"
local optim = require"optim"
local nn = require"nn"
local image = require"image"

local string_format = string.format

--% Initialization
misc.clear_screen()

--% =============== Part 1: Loading movie ratings dataset ================
--  You will start by loading the movie ratings dataset to understand the
--  structure of the data.
--  
misc.printf('Loading movie ratings dataset.\n\n');

--  Load data
local load_rlt = loader.load_from_mat('ex8_movies.mat');
local Y = load_rlt.Y
local R = load_rlt.R

--  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
--  943 users
--
--  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
--  rating to movie i

--  From the matrix, we can compute statistics like average rating.
misc.printf('Average rating for movie 1 (Toy Story): %f / 5\n\n', 
        Y[1][R[1]:byte()]:mean())

--  We can "visualize" the ratings matrix by plotting it with imagesc
method.plot(Y, "visualize_ratings.png", {"Users", "Movies"}, nil, "imagesc", "color");

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();

--% ============ Part 2: Collaborative Filtering Cost Function ===========
--  You will now implement the cost function for collaborative filtering.
--  To help you debug your cost function, we have included set of weights
--  that we trained on that. Specifically, you should complete the code in 
--  cofiCostFunc.m to return J.

--  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
load_rlt = loader.load_from_mat('ex8_movieParams.mat');
X = load_rlt.X;
Theta = load_rlt.Theta;

--  Reduce the data set size so that this runs faster
local num_users = 4;
local num_movies = 5;
local num_features = 3;
X = X[{{1, num_movies}, {1, num_features}}];
Theta = Theta[{{1, num_users}, {1, num_features}}];
Y = Y[{{1, num_movies}, {1, num_users}}];
R = R[{{1, num_movies}, {1, num_users}}];

--  Evaluate cost function
local param = torch.cat(X, Theta, 1)
local J = method.cofi_cost_func(param:view(param:numel()), 
    Y, R, num_users, num_movies, num_features, 0)
           
misc.printf('Cost at loaded parameters: %f \n(this value should be about 22.22)\n', J);
misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();


--% ============== Part 3: Collaborative Filtering Gradient ==============
--  Once your cost function matches up with ours, you should now implement 
--  the collaborative filtering gradient function. Specifically, you should 
--  complete the code in cofiCostFunc.m to return the grad argument.
--  
misc.printf('\nChecking Gradients (without regularization) ... \n');

--  Check gradients by running checkNNGradients
method.check_cost_function()

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();


--% ========= Part 4: Collaborative Filtering Cost Regularization ========
--  Now, you should implement regularization for the cost function for 
--  collaborative filtering. You can implement it by adding the cost of
--  regularization to the original cost computation.
--  

--  Evaluate cost function
J = method.cofi_cost_func(param:view(param:numel()), 
    Y, R, num_users, num_movies, num_features, 1.5)
           
misc.printf('Cost at loaded parameters (lambda = 1.5): %f (this value should be about 31.34)\n', J);

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();


--% ======= Part 5: Collaborative Filtering Gradient Regularization ======
--  Once your cost matches up with ours, you should proceed to implement 
--  regularization for the gradient. 
--

--  
misc.printf('\nChecking Gradients (with regularization) ... \n');

--  Check gradients by running checkNNGradients
method.check_cost_function(1.5)

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();


--% ============== Part 6: Entering ratings for a new user ===============
--  Before we will train the collaborative filtering model, we will first
--  add ratings that correspond to a new user that we just observed. This
--  part of the code will also allow you to put in your own ratings for the
--  movies in our dataset!
--
local movie_list = method.load_movie_list()

--  Initialize my ratings
local my_ratings = torch.zeros(1682)

-- Check the file movie_idx.txt for id of each movie in our dataset
-- For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[1] = 4;

-- Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[98] = 2;

-- We have selected a few movies we liked / did not like and the ratings we
-- gave are as follows:
my_ratings[7] = 3;
my_ratings[12]= 5;
my_ratings[54] = 4;
my_ratings[64]= 5;
my_ratings[66]= 3;
my_ratings[69] = 5;
my_ratings[183] = 4;
my_ratings[226] = 5;
my_ratings[355] = 5;

misc.printf('\n\nNew user ratings:\n');

for i = 1, my_ratings:numel(), 1 do
    if my_ratings[i] > 0 then
        misc.printf('Rated %d for %s\n', my_ratings[i],
            movie_list[i] or 'unknown');
    end
end

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();


--% ================== Part 7: Learning Movie Ratings ====================
--  Now, you will train the collaborative filtering model on a movie rating 
--  dataset of 1682 movies and 943 users
--

misc.printf('\nTraining collaborative filtering...\n');

--  Load data
load_rlt = loader.load_from_mat('ex8_movies.mat');
Y = load_rlt.Y
R = load_rlt.R

--  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
--  943 users
--
--  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
--  rating to movie i

--  Add our own ratings to the data matrix
Y = torch.cat(my_ratings, Y, 2)
R = torch.cat(torch.ne(my_ratings, 0), R, 2)

--  Normalize Ratings
local Ynorm, Ymean = method.normalize_ratings(Y, R)

--  Useful Values
num_users = Y:size(2)
num_movies = Y:size(1)
num_features = 10

-- Set Initial Parameters (Theta, X)
X = torch.randn(num_movies, num_features)
Theta = torch.randn(num_users, num_features)

local initial_parameters = torch.cat(torch.Tensor(X:storage()), 
    torch.Tensor(Theta:storage()))


-- Set options for fmincg
local options = {
    maxIter = 100
}

-- Set Regularization
local lambda = 0.1
local cost_func = function(param)
    return method.cofi_cost_func(param, Ynorm, R, num_users, num_movies, num_features, lambda)
end

local theta = method.fmincg(cost_func, initial_parameters, options, 20)
--local theta = optim.cg(cost_func, initial_parameters, options)

-- Unfold the returned theta back into U and W
X = theta[{{1, num_movies * num_features}}]:view(num_movies, num_features)
Theta = theta[{{num_movies * num_features + 1, -1}}]:view(num_users, num_features)

misc.printf('Recommender system learning completed.\n');

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause();

--% ================== Part 8: Recommendation for you ====================
--  After training the model, you can now make recommendations by computing
--  the predictions matrix.
--

local p = X * Theta:t()
local my_predictions = p[{{}, 1}] + Ymean

local r, ix = my_predictions:sort(1, true)
misc.printf('\nTop recommendations for you:\n');
for i = 1, 10, 1 do
    local j = ix[i]
    misc.printf('Predicting rating %.1f for movie %s\n', 
        my_predictions[j], movie_list[j]);
end

misc.printf('\n\nOriginal ratings provided:\n');
for i = 1, my_ratings:numel(), 1 do
    if my_ratings[i] > 0 then
        misc.printf('Rated %d for %s\n', my_ratings[i], movie_list[i])
    end
end

