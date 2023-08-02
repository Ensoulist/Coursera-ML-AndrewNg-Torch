local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"

local table_insert = table.insert

local method = {}

function method.warm_up_exercise()
    local A = torch.eye(5)
    print(A)
end

function method.plot_data(x, y)
    plot.pngfigure("plot.png")
    plot.plot("Training data", x, y, "+")
    plot.xlabel("Profit in $10,000s")
    plot.ylabel("Population of City in 10,000s")
    plot.plotflush()
end

function method.compute_cost(X, y, theta)
    local m = y:size(1)
    return torch.sum((X * theta - y):pow(2)) / ( 2 * m)
end

function method.gradient_descent(X, y, theta, alpha, num_iters, with_history)
    local m = y:size(1)
    local history = with_history and {}
    for _ = 1, num_iters, 1 do
        local gradient = X:t() * (X * theta - y) / m
        theta = theta - alpha * gradient
        
        if history then
            table_insert(history, method.compute_cost(X, y, theta))
        end
    end
    return theta, history
end

function method.feature_normalize(X)
    local mu = torch.mean(X, 1)
    local sigma = torch.std(X, 1)
    return torch.cdiv((X - mu:expandAs(X)), sigma:expandAs(X)), mu, sigma
end

function method.normal_eqn(X, y)
    return calc.pinverse(X:t() * X) * X:t() * y
end

return method