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

local method = {}
misc.extend_method(method, "ex2-logistic_regression.method2")

function method.display_data(X, example_width, skip_draw, file_name)
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

    file_name = file_name or "display_data.png"
    plot.pngfigure(file_name)
    plot.imagesc(display_array, "gray")
    plot.plotflush()
    misc.printf("see %s", file_name)
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

        misc.printf("training with lable %d", i)
        local theta, J = optim.cg(feval, torch.zeros(n + 1), {maxIter = 50})
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

function method.predict(Theta1, Theta2, X)
    local X_new = torch.cat(torch.ones(X:size(1)), X, 2)

    local Z2 = X_new * Theta1:t()
    local A2 = torch.sigmoid(Z2)

    local A2_tmp = torch.cat(torch.ones(A2:size(1)), A2, 2)
    local Z3 = A2_tmp * Theta2:t()
    local A3 = torch.sigmoid(Z3)

    local _, rlt = torch.max(A3, 2)
    return rlt:double()

    -- nn version
    -- local l1_num = Theta1:size(2) - 1
    -- local l2_num = Theta1:size(1) 
    -- local out_num = Theta2:size(1)

    -- local m = nn.Sequential()
    -- local l2 = nn.Linear(l1_num, l2_num)
    -- local param = l2:parameters()
    -- param[1]:copy(Theta1:narrow(2, 2, l1_num))
    -- param[2]:copy(Theta1[{{}, 1}])

    -- local l3 = nn.Linear(l2_num, out_num)
    -- param = l3:parameters()
    -- param[1]:copy(Theta2:narrow(2, 2, l2_num))
    -- param[2]:copy(Theta2[{{}, 1}])

    -- m:add(l2)
    -- m:add(nn.Sigmoid())
    -- m:add(l3)
    -- m:add(nn.Sigmoid())

    -- local output = m:forward(X):double()
    -- local _, rlt = torch.max(output, 2)
    -- return rlt:double()
end

return method