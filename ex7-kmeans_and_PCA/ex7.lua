--% Machine Learning Online Class
--  Exercise 7 | Principle Component Analysis and K-Means Clustering
--
--  Instructions
--  ------------
--
--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:
--
--     pca.m
--     projectData.m
--     recoverData.m
--     computeCentroids.m
--     findClosestCentroids.m
--     kMeansInitCentroids.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--

package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method7"
local plot = require"gnuplot"
local optim = require"optim"
local nn = require"nn"
local image = require"image"

local string_format = string.format

--% Initialization
misc.clear_screen()

--% ================= Part 1: Find Closest Centroids ====================
--  To help you implement K-Means, we have divided the learning algorithm 
--  into two functions -- findClosestCentroids and computeCentroids. In this
--  part, you should complete the code in the findClosestCentroids function. 
--
misc.printf('Finding closest centroids.\n\n');

-- Load an example dataset that we will be using
local load_rlt = loader.load_from_mat('ex7data2.mat');
local X = load_rlt.X

-- Select an initial set of centroids
local K = 3; 
local initial_centroids = {{3, 3}, {6, 2}, {8, 5}}

-- Find the closest centroids for the examples using the
-- initial_centroids
local idx = method.find_closest_centroids(X, initial_centroids)

misc.printf('Closest centroids for the first 3 examples: \n')
misc.printf(tostring(idx[{{1, 3}}]))
misc.printf('\n(the closest centroids should be 1, 3, 2 respectively)\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% ===================== Part 2: Compute Means =========================
--  After implementing the closest centroids function, you should now
--  complete the computeCentroids function.
--
misc.printf('\nComputing centroids means.\n\n');

--  Compute means based on the closest centroids found in the previous part.
local centroids = method.compute_centroids2(X, idx, K)

misc.printf('Centroids computed after initial finding of closest centroids: \n')
misc.printf(tostring(centroids))
misc.printf('\n(the centroids should be\n');
misc.printf('   [ 2.428301 3.157924 ]\n');
misc.printf('   [ 5.813503 2.633656 ]\n');
misc.printf('   [ 7.119387 3.616684 ]\n\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% =================== Part 3: K-Means Clustering ======================
--  After you have completed the two functions computeCentroids and
--  findClosestCentroids, you have all the necessary pieces to run the
--  kMeans algorithm. In this part, you will run the K-Means algorithm on
--  the example dataset we have provided. 
--
misc.printf('\nRunning K-Means clustering on example dataset.\n\n');

-- Settings for running K-Means
K = 3;
local max_iters = 10;

-- For consistency, here we set centroids to specific values
-- but in practice you want to generate them automatically, such as by
-- settings them to be random examples (as can be seen in
-- kMeansInitCentroids).
local initial_centroids = {{3, 3}, {6, 2}, {8, 5}}

-- Run K-Means algorithm. The 'true' at the end tells our function to plot
-- the progress of K-Means
method.run_k_means(X, initial_centroids, max_iters, true);
misc.printf('\nK-Means Done.\n\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% ============= Part 4: K-Means Clustering on Pixels ===============
--  In this exercise, you will use K-Means to compress an image. To do this,
--  you will first run K-Means on the colors of the pixels in the image and
--  then you will map each pixel onto its closest centroid.
--  
--  You should now complete the code in kMeansInitCentroids.m
--

misc.printf('\nRunning K-Means clustering on pixels from an image.\n\n');

--  Load an image of a bird
local A = image.load("bird_small.png", 3, "double");

-- If imread does not work for you, you can try instead
--   load ('bird_small.mat');

A = A / 255; -- Divide by 255 so that all values are in the range 0 - 1

-- Size of the image
A = A:permute(2, 3, 1)
local img_size = A:size()

-- Reshape the image into an Nx3 matrix where N = number of pixels.
-- Each row will contain the Red, Green and Blue pixel values
-- This gives us our dataset matrix X that we will use K-Means on.
X = A:reshape(A:size(1) * A:size(2), A:size(3))

-- Run your K-Means algorithm on this data
-- You should try different values of K and max_iters here
K = 16; 
max_iters = 10;

-- When using K-Means, it is important the initialize the centroids
-- randomly. 
-- You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = method.k_means_init_centroids(X, K);

-- Run K-Means
centroids, idx = method.run_k_means(X, initial_centroids, max_iters);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% ================= Part 5: Image Compression ======================
--  In this part of the exercise, you will use the clusters of K-Means to
--  compress an image. To do this, we first find the closest clusters for
--  each example. After that, we 

misc.printf('\nApplying K-Means to compress an image.\n\n');

-- Find closest cluster members
idx = method.find_closest_centroids(X, centroids)

-- Essentially, now we have represented the image X as in terms of the
-- indices in idx. 

-- We can now recover the image from the indices (idx) by mapping each pixel
-- (specified by its index in idx) to the centroid value
local X_recovered = centroids:index(1, idx:long())

-- Reshape the recovered image into proper dimensions
local X_trans = X_recovered:view(img_size[1], img_size[2], 3):permute(3, 1, 2) * 255

image.save("compressed.png", X_trans)
misc.printf('Compressed image is saved into compressed.png.\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


