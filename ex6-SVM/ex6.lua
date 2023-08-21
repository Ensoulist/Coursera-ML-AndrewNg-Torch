--% Machine Learning Online Class
--  Exercise 6 | Support Vector Machines
--
--  Instructions
--  ------------
-- 
--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:
--
--     gaussianKernel.m
--     dataset3Params.m
--     processEmail.m
--     emailFeatures.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--

package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method6"
local plot = require"gnuplot"
local optim = require"optim"
local nn = require"nn"

local string_format = string.format

--% Initialization
misc.clear_screen()

--% =============== Part 1: Loading and Visualizing Data ================
--  We start the exercise by first loading and visualizing the dataset. 
--  The following code will load the dataset into your environment and plot
--  the data.
--

misc.printf('Loading and Visualizing Data ...\n')

-- Load from ex6data1: 
-- You will have X, y in your environment
local load_rlt = loader.load_from_mat('ex6data1.mat');
local X = load_rlt.X
local y = load_rlt.y

-- Plot training data
method.plot_data(X, y, false, {"", ""}, {"Profit in $10,000s", "Population of City in 10,000s"}, "dataset1.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% ==================== Part 2: Training Linear SVM ====================
--  The following code will train a linear SVM on the dataset and plot the
--  decision boundary learned.
--

misc.printf('\nTraining Linear SVM ...\n')

-- You should try to change the C value below and see how the decision
-- boundary varies (e.g., try C = 1000)
local C = 1
local model = method.svm_train(X, y, C, 0, 1e-3, 20);
method.visualize_boundary_linear(X, y, model, string_format("boundary_C_%d.png", C))

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

misc.printf('\nTraining Linear SVM ...\n')

-- You should try to change the C value below and see how the decision
-- boundary varies (e.g., try C = 1000)
C = 10;
local model = method.svm_train(X, y, C, 0, 1e-3, 20);
method.visualize_boundary_linear(X, y, model, string_format("boundary_C_%d.png", C))

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% =============== Part 3: Implementing Gaussian Kernel ===============
--  You will now implement the Gaussian kernel to use
--  with the SVM. You should complete the code in gaussianKernel.m
--
misc.printf('\nEvaluating the Gaussian Kernel ...\n')

local x1 = torch.Tensor({1, 2, 1})
local x2 = torch.Tensor({0, 4, -1})
local sigma = 2
local sim = method.gaussian_kernel(x1, x2, sigma);

misc.printf('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = %f :\n\t%f\n(for sigma = 2, this value should be about 0.324652)\n', sigma, sim);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% =============== Part 4: Visualizing Dataset 2 ================
--  The following code will load the next dataset into your environment and 
--  plot the data. 
--

misc.printf('Loading and Visualizing Data ...\n')

-- Load from ex6data2: 
-- You will have X, y in your environment
local load_rlt = loader.load_from_mat('ex6data2.mat');
X = load_rlt.X
y = load_rlt.y

-- Plot training data
method.plot_data(X, y, false, {"", ""}, {"", ""}, "dataset2.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
--  After you have implemented the kernel, we can now use it to train the 
--  SVM classifier.
-- 
misc.printf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

-- SVM Parameters
C = 1; sigma = 0.1;

-- We set the tolerance and max_passes lower here so that the code will run
-- faster. However, in practice, you will want to run the training to
-- convergence.
model = method.svm_train(X, y, C, 2, 1e-3, 20, sigma);
method.visualize_boundary_linear(X, y, model, "boundary_RBF.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% =============== Part 6: Visualizing Dataset 3 ================
--  The following code will load the next dataset into your environment and 
--  plot the data. 
--

misc.printf('Loading and Visualizing Data ...\n')

-- Load from ex6data3: 
-- You will have X, y in your environment
local load_rlt = loader.load_from_mat('ex6data3.mat');
X = load_rlt.X
y = load_rlt.y
Xval = load_rlt.Xval
yval = load_rlt.yval

print(Xval:size())
print(yval:size())

-- Plot training data
method.plot_data(X, y, false, {"", ""}, {"", ""}, "dataset3.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

--% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

--  This is a different dataset that you can use to experiment with. Try
--  different values of C and sigma here.
-- 

-- Try different SVM Parameters here
C, sigma = method.dataset3_params(X, y, Xval, yval)

-- Train the SVM
model = method.svm_train(X, y, C, 2, 1e-3, 20, sigma);
method.visualize_boundary_linear(X, y, model, "boundary_ds3.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


