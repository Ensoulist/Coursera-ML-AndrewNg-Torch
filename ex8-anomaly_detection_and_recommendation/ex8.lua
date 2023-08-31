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

--% ================== Part 1: Load Example Dataset  ===================
--  We start this exercise by using a small dataset that is easy to
--  visualize.
--
--  Our example case consists of 2 network server statistics across
--  several machines: the latency and throughput of each machine.
--  This exercise will help us find possibly faulty (or very fast) machines.
--

misc.printf('Visualizing example dataset for outlier detection.\n\n');

--  The following command loads the dataset. You should now have the
--  variables X, Xval, yval in your environment
local load_rlt = loader.load_from_mat('ex8data1.mat');
local X = load_rlt.X
local Xval = load_rlt.Xval
local yval = load_rlt.yval

--  Visualize the example dataset
method.plot({{"", X[{{}, 1}], X[{{}, 2}], "+"}}, 
    "the_first_dataset.png", {"Latency (ms)", "Throughput (mb/s)"}, 
    {0, 30, 0, 30});

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% ================== Part 2: Estimate the dataset statistics ===================
--  For this exercise, we assume a Gaussian distribution for the dataset.
--
--  We first estimate the parameters of our assumed Gaussian distribution, 
--  then compute the probabilities for each of the points and then visualize 
--  both the overall distribution and where each of the points falls in 
--  terms of that distribution.
--
misc.printf('Visualizing Gaussian fit.\n\n');

--  Estimate my and sigma2
local mu, sigma2 = method.estimate_gaussian(X)

--  Returns the density of the multivariate normal at each data point (row) 
--  of X
local p = method.multi_variate_gaussian(X, mu, sigma2)

--  Visualize the fit
local plot_tbl = method.visualize_fit(X, mu, sigma2);
method.plot(plot_tbl, "visualize_fit.png", 
    {"Latency (ms)", "Throughput (mb/s)"}, {0, 30, 0, 30});

misc.printf('Program paused. Press enter to continue.\n');
misc.pause();

--% ================== Part 3: Find Outliers ===================
--  Now you will find a good epsilon threshold using a cross-validation set
--  probabilities given the estimated Gaussian distribution
-- 

local pval = method.multi_variate_gaussian(Xval, mu, sigma2)

local epsilon, F1 = method.select_threshold(yval, pval)
misc.printf('Best epsilon found using cross-validation: %e\n', epsilon);
misc.printf('Best F1 on Cross Validation Set:  %f\n', F1);
misc.printf('   (you should see a value epsilon of about 8.99e-05)\n');
misc.printf('   (you should see a Best F1 value of  0.875000)\n\n');

--  Find the outliers in the training set and plot the
p = p:squeeze()
local outliers = torch.lt(p, epsilon)
outliers = outliers:nonzero():squeeze()
local pt = X:index(1, outliers)


--  Draw a red circle around those outliers
table.insert(plot_tbl, 2, {"", pt[{{}, 1}], pt[{{}, 2}], "+"})
method.plot(plot_tbl, "outliers.png", 
    {"Latency (ms)", "Throughput (mb/s)"}, {0, 30, 0, 30});

misc.printf('Program paused. Press enter to continue.\n');
misc.pause();

--% ================== Part 4: Multidimensional Outliers ===================
--  We will now use the code from the previous part and apply it to a 
--  harder problem in which more features describe each datapoint and only 
--  some features indicate whether a point is an outlier.
--

--  Loads the second dataset. You should now have the
--  variables X, Xval, yval in your environment
load_rlt = loader.load_from_mat('ex8data2.mat');
X = load_rlt.X
Xval = load_rlt.Xval
yval = load_rlt.yval

--  Apply the same steps to the larger dataset
mu, sigma2 = method.estimate_gaussian(X)

--  Training set 
p = method.multi_variate_gaussian(X, mu, sigma2)

--  Cross-validation set
pval = method.multi_variate_gaussian(Xval, mu, sigma2)

--  Find the best threshold
epsilon, F1 = method.select_threshold(yval, pval)

misc.printf('Best epsilon found using cross-validation: %e\n', epsilon);
misc.printf('Best F1 on Cross Validation Set:  %f\n', F1);
misc.printf('   (you should see a value epsilon of about 1.38e-18)\n');
misc.printf('   (you should see a Best F1 value of 0.615385)\n');
misc.printf('# Outliers found: %d\n\n', torch.lt(p, epsilon):sum())

