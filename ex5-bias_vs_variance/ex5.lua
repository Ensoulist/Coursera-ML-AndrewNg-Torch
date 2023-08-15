---- Machine Learning Online Class
--  Exercise 5 | Regularized Linear Regression and Bias-Variance
--
--  Instructions
--  ------------
-- 
--  This file contains code that helps you get started on the
--  exercise. You will need to complete the following functions:
--
--     linearRegCostFunction.m
--     learningCurve.m
--     validationCurve.m
--
--  For this exercise, you will not need to change any code in this file,
--  or any other files other than those mentioned above.
--

package.path = package.path .. ";../?.lua"
local misc = require"utils.misc"
local loader = require"utils.loader"
local method = require"method5"
local plot = require"gnuplot"
local optim = require"optim"
local nn = require"nn"

local string_format = string.format

---- Initialization
misc.clear_screen()

---- =========== Part 1: Loading and Visualizing Data =============
--  We start the exercise by first loading and visualizing the dataset. 
--  The following code will load the dataset into your environment and plot
--  the data.
--

-- Load Training Data
misc.printf('Loading and Visualizing Data ...\n')

-- Load from ex5data1: 
-- You will have X, y, Xval, yval, Xtest, ytest in your environment
local load_rlt = loader.load_from_mat('ex5data1.mat');
local X = load_rlt.X
local y = load_rlt.y
local Xval = load_rlt.Xval
local yval = load_rlt.yval
local Xtest = load_rlt.Xtest
local ytest = load_rlt.ytest

-- m = Number of examples
local m = X:size(1)

-- Plot training data
plot.pngfigure("training_data.png")
plot.plot("", torch.Tensor(X:storage()), torch.Tensor(y:storage()), "+")
plot.xlabel("Change in water level (x)")
plot.ylabel("Water flowing out of the dam (y)")
plot.plotflush()
misc.printf("plot. see training_data.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 2: Regularized Linear Regression Cost =============
--  You should now implement the cost function for regularized linear 
--  regression. 
--

local theta = torch.Tensor({1, 1})
local J = method.liner_reg_cost_function(torch.cat(torch.ones(m), X), y, theta, 1);

misc.printf('Cost at theta = [1 ; 1]: %f \n(this value should be about 303.993192)\n', J);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 3: Regularized Linear Regression Gradient =============
--  You should now implement the gradient for regularized linear 
--  regression.
--

theta = torch.Tensor({1, 1})
local grad
J, grad = method.liner_reg_cost_function(torch.cat(torch.ones(m), X), y, theta, 1)

misc.printf('Gradient at theta = [1 ; 1]: ')
misc.printf(tostring(grad))
misc.printf('this value should be about [-15.303016; 598.250744])\n')

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 4: Train Linear Regression =============
--  Once you have implemented the cost and gradient correctly, the
--  trainLinearReg function will use your cost function to train 
--  regularized linear regression.
-- 
--  Write Up Note: The data is non-linear, so this will not give a great 
--                 fit.
--

--  Train linear regression with lambda = 0
local lambda = 0;
theta = method.train_linear_reg(torch.cat(torch.ones(m), X), y, lambda);

--  Plot fit over the data
plot.pngfigure("linear_fit.png")
plot.plot({{"", torch.Tensor(X:storage()), torch.Tensor(y:storage()), "+"},
    {"", torch.Tensor(X:storage()), torch.cat(torch.ones(m), X) * theta, "-"}})
plot.xlabel("Change in water level (x)")
plot.ylabel("Water flowing out of the dam (y)")
plot.plotflush()
misc.printf("plot. see linear_fit.png")

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()


---- =========== Part 5: Learning Curve for Linear Regression =============
--  Next, you should implement the learningCurve function. 
--
--  Write Up Note: Since the model is underfitting the data, we expect to
--                 see a graph with "high bias" -- Figure 3 in ex5.pdf 
--

lambda = 0
local error_train, error_val = method.learning_curve(torch.cat(torch.ones(m), X), y, 
    torch.cat(torch.ones(Xval:size(1)), Xval), yval, lambda)

plot.pngfigure("learning_curve.png")
plot.plot({{"Train", torch.range(1, m), error_train, "-"}, 
    {"Cross Validation", torch.range(1, m), error_val, "-"}})
plot.xlabel("Number of training examples")
plot.ylabel("Error")
plot.plotflush()

misc.printf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1, m, 1 do
    misc.printf('  \t%d\t\t%f\t%f\n', i, error_train[i], error_val[i]);
end

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 6: Feature Mapping for Polynomial Regression =============
--  One solution to this is to use polynomial regression. You should now
--  complete polyFeatures to map each example into its powers
--

local p = 8

-- Map X onto Polynomial Features and Normalize
local X_poly = method.poly_features(X, p)
local mu, sigma
X_poly, mu, sigma = method.feature_normalize(X_poly)
X_poly = torch.cat(torch.ones(m), X_poly)

-- Map X_poly_test and normalize (using mu and sigma)
local X_poly_test = method.poly_features(Xtest, p)
X_poly_test = method.feature_normalize(X_poly_test, mu, sigma)
X_poly_test = torch.cat(torch.ones(X_poly_test:size(1)), X_poly_test)

-- Map X_poly_val and normalize (using mu and sigma)
local X_poly_val = method.poly_features(Xval, p)
X_poly_val = method.feature_normalize(X_poly_val, mu, sigma)
X_poly_val = torch.cat(torch.ones(X_poly_val:size(1)), X_poly_val)

misc.printf('Normalized Training Example 1:\n');
misc.printf(tostring(X_poly[1]))

misc.printf('\nProgram paused. Press enter to continue.\n');
misc.pause()


    ---- =========== Part 7: Learning Curve for Polynomial Regression =============
--  Now, you will get to experiment with polynomial regression with multiple
--  values of lambda. The code below runs polynomial regression with 
--  lambda = 0. You should try running the code with different values of
--  lambda to see how the fit and learning curve change.
--
for _, lambda in ipairs({0, 1, 100}) do
    misc.printf("For lambda = %f\n", lambda);

    theta = method.train_linear_reg(X_poly, y, lambda)
    
    local pic_name = string_format("polynomial_fit_lambda_%d.png", lambda)
    plot.pngfigure(pic_name)
    local plots = {{"", torch.Tensor(X:storage()), torch.Tensor(y:storage()), "+"}}
    method.plot_fit(torch.min(X), torch.max(X), mu, sigma, theta, p, plots)
    plot.plot(plots)
    plot.xlabel("Change in water level (x)")
    plot.ylabel("Water flowing out of the dam (y)")
    plot.plotflush()
    misc.printf("plot. see %s\n", pic_name)
    
    pic_name = string_format("polynomial_learning_curve_lambda_%d.png", lambda)
    plot.pngfigure(pic_name)
    error_train, error_val = method.learning_curve(X_poly, y, X_poly_val, yval, lambda)
    plot.plot({"Train", torch.range(1, m), error_train, "-"}, 
        {"Cross Validation", torch.range(1, m), error_val, "-"})
    plot.xlabel("Number of training examples")
    plot.ylabel("Error")
    plot.plotflush()
    misc.printf("plot. see %s\n", pic_name)
    
    misc.printf('Polynomial Regression (lambda = %f)\n\n', lambda);
    misc.printf('# Training Examples\tTrain Error\tCross Validation Error\n');
    for i = 1, m, 1 do
        misc.printf('  \t%d\t\t%f\t%f\n', i, error_train[i], error_val[i]);
    end
end

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 8: Validation for Selecting Lambda =============
--  You will now implement validationCurve to test various values of 
--  lambda on a validation set. You will then use this to select the
--  "best" lambda value.
--

local lambda_vec
lambda_vec, error_train, error_val = method.validation_curve(X_poly, y, X_poly_val, yval)   

plot.pngfigure("selecting_lambda.png")
plot.plot({
    {"Train", lambda_vec, error_train, "-"},
    {"Cross Validation", lambda_vec, error_val, "-"},
})
plot.xlabel("lambda")
plot.ylabel("Error")
plot.plotflush()
misc.printf("plot. see selecting_lambda.png")

misc.printf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1, lambda_vec:numel(), 1 do
	misc.printf(' %f\t%f\t%f\n', 
        lambda_vec[i], error_train[i], error_val[i]);
end

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 9: Computing test set error =============
misc.printf('Computing test set error:\n');
theta = method.train_linear_reg(X_poly, y, 3)
error_test = method.liner_reg_cost_function(X_poly_test, ytest, theta, 0)

misc.printf('test set error = %f \n(this value should be about 3.8599 for lambda =3)\n',error_test);

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()

---- =========== Part 10:Plotting learning curves with randomly selected examples =============
lambda = 0.01;
error_train, error_val = method.random_learning_curve(X_poly, y, 
    X_poly_val, yval, lambda)

plot.pngfigure("learning_curve_randomly_selected.png")
plot.plot({"Train", torch.range(1, m), error_train, "-"}, 
    {"Cross Validation", torch.range(1, m), error_val, "-"})
plot.xlabel("Number of training examples")
plot.ylabel("Error")
plot.plotflush()
misc.printf("plot. see learning_curve_randomly_selected.png")

misc.printf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1, m, 1 do
    misc.printf('  \t%d\t\t%f\t%f\n', i, error_train[i], error_val[i]);
end

misc.printf('Program paused. Press enter to continue.\n');
misc.pause()
