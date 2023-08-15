local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"
local optim = require"optim"
local nn = require"nn"

local table_insert = table.insert
local math_floor = math.floor
local math_ceil = math.ceil
local math_sqrt = math.sqrt
local math_min = math.min

local method = {}

misc.extend_method(method, "ex4-NN_back_propagation.method4")

function method.liner_reg_cost_function(X, y, theta, lambda)
    local y_hat = X * theta
    local reg_theta = theta[{{2, -1}}]
    local cost = torch.cmul(y_hat - y, y_hat - y):sum() / (y:size(1) * 2)
        + (lambda * torch.pow(reg_theta, 2):sum()) / (y:size(1) * 2)
    
    local grad = (X:t() * (y_hat - y) + lambda * torch.cat(torch.zeros(1), reg_theta)) / y:size(1)
    return cost, grad
end

function method.train_linear_reg(X, y, lambda)
    local n = X:size(2)
    local theta = torch.zeros(n)

    local feval = function(param)
        local cost, grad = method.liner_reg_cost_function(X, y, param, lambda)
        return cost, grad
    end
    theta = optim.cg(feval, theta, {maxIter = 200})
    return theta
end

function method.learning_curve(X, y, X_val, y_val, lambda)
    local m = X:size(1)
    local err_train = {} 
    local err_val = {}
    for i = 1, m, 1 do
        local train_X = X[{{1, i}, {}}]
        local train_y = y[{{1, i}, {}}]
        local theta = method.train_linear_reg(train_X, train_y, lambda)
        local t_err = method.liner_reg_cost_function(train_X, train_y, theta, 0)
        local v_err = method.liner_reg_cost_function(X_val, y_val, theta, 0)
        table_insert(err_train, t_err)
        table_insert(err_val, v_err)
    end
    return torch.Tensor(err_train), torch.Tensor(err_val)
end

function method.poly_features(X, p)
    local cols = {}
    for i = 1, p, 1 do
        table_insert(cols, torch.pow(X, i))
    end
    return torch.cat(cols, 2)
end

function method.feature_normalize(X, mu, sigma)
    mu = mu or torch.mean(X, 1)
    sigma = sigma or torch.std(X, 1)
    return torch.cdiv((X - mu:expandAs(X)), sigma:expandAs(X)), mu, sigma
end

function method.plot_fit(min_x, max_x, mu, sigma, theta, p, plots)
    local x = torch.range(min_x - 15, max_x + 25, 0.05)
    local X_poly = method.poly_features(x, p)
    X_poly = method.feature_normalize(X_poly, mu, sigma)
    X_poly = torch.cat(torch.ones(x:size(1)), X_poly)
    local y_hat = X_poly * theta
    table_insert(plots, {"", x, y_hat, "-"})
end

function method.validation_curve(X, y, Xval, yval, lambda)
    local lambdas = {0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10}
    local error_train = {}
    local error_val = {}

    for _, lambda in ipairs(lambdas) do
        local theta = method.train_linear_reg(X, y, lambda)
        local cost = method.liner_reg_cost_function(X, y, theta, 0)
        table_insert(error_train, cost)
        cost = method.liner_reg_cost_function(Xval, yval, theta, 0)
        table_insert(error_val, cost) 
    end
    return torch.Tensor(lambdas), 
        torch.Tensor(error_train), 
        torch.Tensor(error_val)
end

function method.random_learning_curve(X, y, Xval, yval, lambda)
    local m = X:size(1)
    local err_train = {} 
    local err_val = {}
    local try_count = 50
    for i = 1, m, 1 do
        local all_err_train = 0
        local all_err_cv = 0
        for j = 1, try_count, 1 do
            local rand_train = torch.randperm(m)[{{1, i}}]:long()
            local X_train = X:index(1, rand_train)
            local y_train = y:index(1, rand_train)

            local rand_cv = torch.randperm(m)[{{1, i}}]:long()
            local X_cv = Xval:index(1, rand_cv)
            local y_cv = yval:index(1, rand_cv)
             
            local theta = method.train_linear_reg(X_train, y_train, lambda)
            local cost_train = method.liner_reg_cost_function(X_train, y_train, theta, 0)
            local cost_cv = method.liner_reg_cost_function(X_cv, y_cv, theta, 0)
            all_err_train = all_err_train + cost_train
            all_err_cv = all_err_cv + cost_cv
        end
        table_insert(err_train, all_err_train / try_count)
        table_insert(err_val, all_err_cv / try_count)
    end
    return torch.Tensor(err_train), torch.Tensor(err_val)
end

return method