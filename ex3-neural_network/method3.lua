local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"
local optim = require"optim"

local table_insert = table.insert
local math_floor = math.floor
local math_ceil = math.ceil
local math_sqrt = math.sqrt

local method = {}
misc.extend_method(method, "ex2-logistic_regression.method2")

function method.display_data(X, example_width, skip_draw)
    -- Set example_width automatically if not passed in
    if not example_width then
        example_width = math_floor(math_sqrt(X:size(2)))
    end

    -- Compute rows, cols
    local m = X:size(1)
    local n = X:size(2)
    local example_height = math_floor(n / example_width)

    -- compute number of items to display
    local display_rows = math_floor(math_sqrt(m))
    local display_cols = math_ceil(m / display_rows)

    -- between images padding
    local pad = 1;
    local display_array = torch.ones(pad + display_rows * (example_height + pad), 
        pad + display_cols * (example_width + pad))

    local curr_ex = 1
    for j = 1, display_rows, 1 do
        for i = 1, display_cols, 1 do
            if curr_ex > m then
                break
            end

            local x = X[curr_ex]
            local max_val = torch.abs(x):max()
            x = x / max_val
            local x_image = x:view(example_height, example_width)

            local height_begin = pad + (j - 1) * (example_height + pad) + 1
            local height_end = height_begin + example_height - 1
            local width_begin = pad + (i - 1) * (example_width + pad) + 1
            local width_end = width_begin + example_width - 1
            display_array[{{height_begin, height_end}, {width_begin, width_end}}] = x_image

            curr_ex = curr_ex + 1
        end
    end

    if skip_draw then
        return display_array
    end

    plot.pngfigure("display_data.png")
    plot.imagesc(display_array, "gray")
    plot.plotflush()
    misc.printf("see display_data.png")
end

function method.lr_cost_function(theta, X, y, lambda)
    return method.cost_function_reg(theta, X, y, lambda)
end

function method.one_vs_all(X, y, num_labels, lambda)
    local m = X:size(1)
    local n = X:size(2)

    local thetas = {}
    local X = torch.cat(torch.ones(m), X, 2) -- Add a column of ones to x

    for i = 1, num_labels, 1 do
        local this_y = torch.eq(y, i):double()

        local feval = function(t)
            local cost, grad = method.lr_cost_function(t, X, this_y, lambda)
            return cost, grad
        end
        local theta, J = optim.cg(feval, torch.zeros(n + 1), {maxIter = 400})
        thetas[i] = theta
    end
    return torch.cat(thetas, 2)
end

function method.predict_one_vs_all(all_theta, X)
    X = torch.cat(torch.ones(X:size(1)), X, 2)
    local pred = torch.sigmoid(X * all_theta)
    local _, rlt = torch.max(pred, 2) 
    return rlt:double()
end

return method