local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"
local loader = require"utils.loader"
local optim = require"optim"

local method = {}

misc.extend_method(method, "ex2-logistic_regression.method2")
misc.extend_method(method, "ex4-NN_back_propagation.method4")
misc.extend_method(method, "ex7-kmeans_and_PCA.method7")

function method.estimate_gaussian(X)
    return X:mean(1), X:var(1, true)
end

function method.multi_variate_gaussian(X, mu, sigma2)
    local p1 = torch.sqrt(2 * math.pi * sigma2):pow(-1)
    local p2 = (-(X - mu:expandAs(X)):pow(2):cdiv((2 * sigma2):expandAs(X))):exp()
    return p2:cmul(p1:expandAs(p2)):prod(2)
end

function method.visualize_fit(X, mu, sigma2)
    local plot_tbl = {
        {"", X[{{}, 1}], X[{{}, 2}], "+"},
    }
    local x1 = torch.range(0, 35, 0.5)
    local x2 = x1
    local x_tbl = {}
    for i = 1, x1:size(1), 1 do
        local x = x1[i]
        table.insert(x_tbl, torch.cat(torch.ones(x2:size()) * x, x2, 2))
    end
    local x_tmp = torch.cat(x_tbl, 1)
    local z = method.multi_variate_gaussian(x_tmp, mu, sigma2)
    z = z:view(x1:size(1), x2:size(1))
    
    local vals = torch.pow(10, torch.range(-20, 0, 3))
    for i = 1, vals:numel(), 1 do
        local val = vals[i]
        local plot_x, plot_y = method.contour(z, x1, x2, val)
        if plot_x:dim() > 0 and plot_y:dim() > 0 then
            table.insert(plot_tbl, {"", plot_x, plot_y, "+"})
        end
    end

    return plot_tbl
end

function method.select_threshold(yval, pval)
    local pmax = torch.max(pval)
    local pmin = torch.min(pval)
    local step_size = (pmax - pmin) / 1000
    local yval_byte = yval:byte()

    local best_epsilon, best_f1
    for epsilon = pmin, pmax, step_size do
        local ptrue = torch.lt(pval, epsilon)
        local tp = torch.cbitand(ptrue, yval_byte):sum()
        local fp = ptrue:sum() - tp
        local fn = yval_byte:sum() - tp
        local prec = tp / (tp + fp)
        local rec = tp / (tp + fn)
        local f1 = 2 * prec * rec / (prec + rec)
        if f1 > (best_f1 or 0) then
            best_f1 = f1
            best_epsilon = epsilon
        end
    end
    return best_epsilon, best_f1
end

function method.cofi_cost_func(params, Y, R, num_users, num_movies, num_features, lambda)
    local X = params[{{1, num_movies * num_features}}]:view(num_movies, num_features)
    local Theta = params[{{num_movies * num_features + 1, -1}}]:view(num_users, num_features)
    local R_double = R:double()
    local J = (X * Theta:t() - Y):cmul(R_double):pow(2):sum() / 2
    J = J + (lambda / 2) * torch.sum(torch.pow(Theta, 2)) 
        + (lambda / 2) * torch.sum(torch.pow(X, 2))
    local grad_X = (X * Theta:t() - Y):cmul(R_double) * Theta + lambda * X
    local grad_theta = (X * Theta:t() - Y):cmul(R_double):t() * X  + lambda * Theta
    local grad = torch.cat(torch.Tensor(grad_X:storage()), torch.Tensor(grad_theta:storage()))
    return J, grad
end

function method.check_cost_function(lambda)
    lambda = lambda or 0

    -- Create small problem
    local X_t = torch.rand(4, 3)
    local Theta_t = torch.rand(5, 3)

    -- Zap out most entries
    local Y = X_t * Theta_t:t()
    Y[torch.gt(torch.rand(Y:size()), 0.5)] = 0
    local R = torch.zeros(Y:size())
    R[torch.ne(Y, 0)] = 1


    -- Run Gradient Checking
    local X = torch.randn(X_t:size())
    local Theta = torch.randn(Theta_t:size())
    local num_users = Y:size(2)
    local num_movies = Y:size(1)
    local num_features = Theta_t:size(2)
    
    local cost_func = function(param)
        return method.cofi_cost_func(param, Y, R, num_users, num_movies, num_features, lambda)
    end
    local cost_params = torch.cat(X, Theta, 1)
    local numgrad = method.compute_numberical_gradient(cost_func, 
        cost_params:view(cost_params:numel()))

    local cost, grad = method.cofi_cost_func(cost_params:view(cost_params:numel()),
        Y, R, num_users, num_movies, num_features, lambda)

    misc.printf(tostring(torch.cat(numgrad, grad, 2)))
    misc.printf('he above two columns you get should be very similar.\n(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    local diff = torch.norm(numgrad - grad) / torch.norm(numgrad + grad)
    misc.printf('If your cost function implementation is correct, then \nthe relative difference will be small (less than 1e-9). \n\nRelative Difference: %g\n', diff);
end

function method.load_movie_list()
    local file = io.open("movie_ids.txt", "r")
    if not file then
        return {}
    end
    local rlt = {}
    for line in file:lines() do
        local name = line:match("%d+ (.*) %(%d+%)$")
        table.insert(rlt, name)
    end
    return rlt
end

function method.normalize_ratings(Y, R)
    local Ymean = torch.zeros(Y:size(1))
    local Ynorm = torch.zeros(Y:size())
    for i = 1, Ymean:size(1), 1 do
        local idx = torch.eq(R[i], 1)
        Ymean[i] = Y[i][idx]:mean()
        Ynorm[i][idx] = Y[i][idx] - Ymean[i]
    end
    return Ynorm, Ymean
end

return method