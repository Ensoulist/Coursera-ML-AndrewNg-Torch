local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"

local table_insert = table.insert

local method = {}

function method.plot_data(X, y, param_only, legend, labels, file_name)
    legend = legend or {"Admitted", "Not admitted"}
    labels = labels or {"Exam 1 score", "Exam 2 score"}

    local pos = torch.eq(y, 1)
    local neg = torch.eq(y, 0)
    local pt_tbl = {
        {legend[1], X[{{}, 1}][pos], X[{{}, 2}][pos], "+"}, 
        {legend[2], X[{{}, 1}][neg], X[{{}, 2}][neg], "+"},
    }
    if param_only then
        return pt_tbl
    end

    plot.pngfigure(file_name or "plot.png")
    plot.plot(pt_tbl)
    plot.xlabel(labels[1])
    plot.ylabel(labels[2])
    plot.plotflush()
end

function method.cost_function(theta, X, y)
    local g = method.sigmoid(X * theta)
    local cost = (torch.dot(y, torch.log(g)) + torch.dot((1 - y), torch.log(1 - g))) / -y:size(1)
    local grad = (X:t() * (g - y)) / y:size(1)
    return cost, grad
end

function method.sigmoid(x)
    -- return torch.sigmoid(x)
    return torch.pow((1 + torch.exp(-x)), -1)
end

function method.plot_decision_boundary(theta, X, y, param_only, 
    legend, labels, file_name)

    local tbl = method.plot_data(X[{{}, {2, 3}}], y, true, legend, labels, file_name)
    local plot_x = torch.Tensor({torch.min(X[{{}, 2}]) - 2, torch.max(X[{{}, 2}]) + 2})
    local plot_y = (theta[2] * plot_x + theta[1]) * (-1 / theta[3])
    table_insert(tbl, {"Decision Boundary", plot_x, plot_y, "-"})

    plot.pngfigure(file_name or "plot2.png")
    plot.plot(tbl)
    plot.xlabel(labels and labels[1] or "Exam 1 score")
    plot.ylabel(labels and labels[2] or "Exam 2 score")
    plot.plotflush()
end

function method.predict(theta, X)
    local p = method.sigmoid(X * theta)
    return p:gt(0.5):double()
end

function method.map_feature(X, degree)
    degree = degree or 6
    local X1 = X[{{}, 1}]
    local X2 = X[{{}, 2}]
    local map = {torch.ones(X1:size(1), 1)}
    for i = 1, degree, 1 do
        for j = 0, i, 1 do
            local one = torch.cmul(torch.pow(X1, i - j), torch.pow(X2, j))
            table_insert(map, one)
        end
    end
    return torch.cat(map, 2)
end

function method.cost_function_reg(theta, X, y, lambda)
    local cost, grad = method.cost_function(theta, X, y)
    local m = y:size(1)
    local theta2 = theta:clone()
    theta2[1] = 0
    return cost + (torch.sum(theta2:pow(2))) * lambda / (2 * m),
        grad + lambda * theta2 / m
end

return method