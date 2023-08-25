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

--% ================== Part 1: Load Example Dataset  ===================
--  We start this exercise by using a small dataset that is easily to
--  visualize
--
misc.printf('Visualizing example dataset for PCA.\n\n');

--  The following command loads the dataset. You should now have the 
--  variable X in your environment
local load_rlt = loader.load_from_mat('ex7data1.mat');
local X = load_rlt.X

--  Visualize the example dataset
method.plot({{"", X[{{}, 1}], X[{{}, 2}], "+"}}, 
    "example_dataset.png", nil, {0.5, 6.5, 0, 8})

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% =============== Part 2: Principal Component Analysis ===============
--  You should now implement PCA, a dimension reduction technique. You
--  should complete the code in pca.m
--
misc.printf('\nRunning PCA on example dataset.\n\n');

--  Before running PCA, it is important to first normalize X
local X_norm, mu, sigma = method.feature_normalize(X)

--  Run PCA
local U, S = method.pca(X_norm)

--  Compute mu, the mean of the each feature

--  Draw the eigenvectors centered at mean of data. These lines show the
--  directions of maximum variations in the dataset.
local plot_tbl = {{"", X[{{}, 1}], X[{{}, 2}], "+"}}
plot_tbl = method.draw_line(plot_tbl, mu, mu + 1.5 * S[1] * U[{{}, 1}])
plot_tbl = method.draw_line(plot_tbl, mu, mu + 1.5 * S[2] * U[{{}, 2}])
method.plot(plot_tbl, "eigenvectors.png", nil, {0.5, 6.5, 0, 8})

misc.printf('Top eigenvector: \n');
misc.printf(' U(:,1) = %f %f \n', U[1][1], U[2][1]);
misc.printf('\n(you should expect to see -0.707107 -0.707107)\n');

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% =================== Part 3: Dimension Reduction ===================
--  You should now implement the projection step to map the data onto the 
--  first k eigenvectors. The code will then plot the data in this reduced 
--  dimensional space.  This will show you what the data looks like when 
--  using only the corresponding eigenvectors to reconstruct it.
--
--  You should complete the code in projectData.m
--
misc.printf('\nDimension reduction on example dataset.\n\n');

--  Project the data onto K = 1 dimension
local K = 1
local Z = method.project_data(X_norm, U, K)
misc.printf('Projection of the first example: %s\n', tostring(Z[1]))
misc.printf('\n(this value should be about 1.481274)\n\n');

local X_rec = method.recover_data(Z, U, K)
misc.printf('Approximation of the first example: %f %f\n', X_rec[1][1], X_rec[1][2]);
misc.printf('\n(this value should be about  -1.047419 -1.047419)\n\n');

local plot_tbl = {
    {"", X_norm[{{}, 1}], X_norm[{{}, 2}], "+"},
    {"", X_rec[{{}, 1}], X_rec[{{}, 2}], "+"},
}
for i = 1, X_norm:size(1), 1 do
    plot_tbl = method.draw_line(plot_tbl, X_norm[i], X_rec[i])
end
method.plot(plot_tbl, "data_PCA.png", nil, {-4, 3, -4, 3})

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% =============== Part 4: Loading and Visualizing Face Data =============
--  We start the exercise by first loading and visualizing the dataset.
--  The following code will load the dataset into your environment
--
misc.printf('\nLoading face dataset.\n\n');

--  Load Face dataset
load_rlt = loader.load_from_mat('ex7faces.mat');
local X = load_rlt.X

--  Display the first 100 faces in the dataset
method.display_data(X[{{1, 100}, {}}], nil, nil, "face_dataset.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% =========== Part 5: PCA on Face Data: Eigenfaces  ===================
--  Run PCA and visualize the eigenvectors which are in this case eigenfaces
--  We display the first 36 eigenfaces.
--
misc.printf('\nRunning PCA on face dataset.\n(this might take a minute or two ...)\n\n');

--  Before running PCA, it is important to first normalize X by subtracting 
--  the mean value from each feature
X_norm, mu, sigma = method.feature_normalize(X)

--  Run PCA
U, S = method.pca(X_norm)

--  Visualize the top 36 eigenvectors found
method.display_data(U[{{}, {1, 36}}]:t(), nil, nil, "principal_components_face.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% ============= Part 6: Dimension Reduction for Faces =================
--  Project images to the eigen space using the top k eigenvectors 
--  If you are applying a machine learning algorithm 
misc.printf('\nDimension reduction for face dataset.\n\n');

K = 100;
Z = method.project_data(X_norm, U, K);

misc.printf('The projected data Z has a size of: ')
misc.printf('%s ', tostring(Z:size()));

misc.printf('\n\nProgram paused. Press enter to continue.\n');
misc.pause()

--% ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
--  Project images to the eigen space using the top K eigen vectors and 
--  visualize only using those K dimensions
--  Compare to the original input, which is also displayed

misc.printf('\nVisualizing the projected (reduced dimension) faces.\n\n');

K = 100;
X_rec = method.recover_data(Z, U, K)

-- Display normalized data
method.display_data(X_norm[{{1, 100}, {}}], nil, nil, "faces_original.png");

-- Display reconstructed data from only k eigenfaces
method.display_data(X_rec[{{1, 100}, {}}], nil, nil, "faces_recovered.png");

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


--% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
--  One useful application of PCA is to use it to visualize high-dimensional
--  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
--  pixel colors of an image. We first visualize this output in 3D, and then
--  apply PCA to obtain a visualization in 2D.

misc.clear_screen()

-- Reload the image from the previous exercise and run K-Means on it
-- For this to work, you need to complete the K-Means assignment first
local A = image.load('bird_small.png', 3, "double")

--A = A / 255;
A = A:permute(2, 3, 1)
local img_size = A:size()
X = A:reshape(img_size[1] * img_size[2], img_size[3])

local K = 16; 
local max_iters = 10;
local initial_centroids = method.k_means_init_centroids(X, K);
local centroids, idx = method.run_k_means(X, initial_centroids, max_iters);

--  Sample 1000 random indexes (since working with all the data is
--  too expensive. If you have a fast computer, you may increase this.
local sel = torch.randperm(X:size(1))[{{1, 1000}}]:long()
local sel_idx = idx:index(1, sel)
local sel_X = X:index(1, sel)

local plot_tbl = {}
for i = 1, K, 1 do
    local this_idx = torch.nonzero(torch.eq(sel_idx, i))[{{}, 1}]
    local this_X = sel_X:index(1, this_idx)
    if this_X:numel() > 0 then
        table.insert(plot_tbl, {"", this_X[{{}, 1}], this_X[{{}, 2}], this_X[{{}, 3}]})
    end
end
method.plot(plot_tbl, "data_in_3D.png", nil, nil, "scatter3")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
-- Use PCA to project this cloud to 2D for visualization

-- Subtract the mean to use PCA
X_norm, mu, sigma = method.feature_normalize(sel_X)

-- PCA and project the data to 2D
U, S = method.pca(X_norm)
Z = method.project_data(X_norm, U, 2)

-- Plot in 2D
plot_tbl = {}
for i = 1, K, 1 do
    local this_idx = torch.nonzero(torch.eq(sel_idx, i))[{{}, 1}]
    local this_Z = Z:index(1, this_idx)
    if this_Z:numel() > 0 then
        table.insert(plot_tbl, {"", this_Z[{{}, 1}], this_Z[{{}, 2}], "+"})
    end
end
method.plot(plot_tbl, "data_pca_2D.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

