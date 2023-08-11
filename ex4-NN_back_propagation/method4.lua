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
misc.extend_method(method, "ex3-neural_network.method3")

function method.nn_cost_function(nn_params, input_layer_size,
    hidden_layer_size, num_labels, X, y, lambda)

    local Theta1 = nn_params[{{1, hidden_layer_size * (input_layer_size + 1)}}]
        :view(hidden_layer_size, input_layer_size + 1) -- 25 401
    local Theta2 = nn_params[{{hidden_layer_size * (input_layer_size + 1) + 1, -1}}]
        :view(num_labels, hidden_layer_size + 1) -- 10 26
    local m = X:size(1)

    local y2_tbl = {}
    for i = 1, num_labels, 1 do
        y2_tbl[i] = torch.eq(y, i):double()
    end
    local y2 = torch.cat(y2_tbl, 2) -- 5000 10

    -- Hand writing version
    --
    local a1 = torch.cat(torch.ones(m), X, 2) -- 5000 401
    local z2 = a1 * Theta1:t() -- 5000 25
    local a2 = torch.sigmoid(z2)
    a2 = torch.cat(torch.ones(m), a2, 2) -- 5000 26
    local h = torch.sigmoid(a2 * Theta2:t()) -- 5000 10

    local J = torch.sum(torch.cmul(-y2, torch.log(h))
            - torch.cmul(1 - y2, torch.log(1 - h))) / m 
        + lambda / (2 * m) *                                    -- regularition
            (torch.sum(torch.pow(Theta1[{{}, {2, -1}}], 2))
            + torch.sum(torch.pow(Theta2[{{}, {2, -1}}], 2)))
    
    local Delta3 = h - y2 -- 5000 10
    local Delta2 = Delta3 * Theta2 -- 5000 26
    Delta2 = Delta2[{{}, {2, -1}}]  -- 5000 25
    Delta2 = torch.cmul(Delta2, method.sigmoid_gradient(z2)) -- 5000 25
    local D1 = Delta2:t() * a1 -- 25 401
        + lambda * torch.cat(torch.zeros(hidden_layer_size), Theta1[{{}, {2, -1}}]) -- regularization
    local D2 = Delta3:t() * a2 -- 10 26
        + lambda * torch.cat(torch.zeros(num_labels), Theta2[{{}, {2, -1}}]) -- regularization
    local Theta1_grad = D1 / m
    local Theta2_grad = D2 / m

    local grad = torch.cat(torch.Tensor(Theta1_grad:storage()),
        torch.Tensor(Theta2_grad:storage()))

    return J, grad
end

function method.sigmoid_gradient(_input)
    return torch.cmul(torch.sigmoid(_input), (1 - torch.sigmoid(_input)))
end

function method.rand_initialize_weight(L_in, L_out)
    local eps = 0.12
    return torch.rand(L_out, L_in + 1) * (2 * eps) - eps
end

function method.check_nn_gradients(lambda)
    lambda = lambda or 0

    local input_layer_size = 3
    local hidden_layer_size = 5
    local num_labels = 3
    local m = 5

    -- We generate some 'random' test data
    local Theta1 = method.debug_initialize_weights(hidden_layer_size, input_layer_size)
    local Theta2 = method.debug_initialize_weights(num_labels, hidden_layer_size)

    -- Reusing debugInitializeWeights to generate X
    local X = method.debug_initialize_weights(m, input_layer_size - 1)
    local y = torch.fmod(torch.range(1, m), 2) + 1

    -- Unroll parameters
    local nn_params = torch.cat(torch.Tensor(Theta1:storage()),
        torch.Tensor(Theta2:storage()))

    local cost_func = function(param)
        local cost, grad = method.nn_cost_function(param, input_layer_size,
            hidden_layer_size, num_labels, X, y, lambda)
        return cost, grad
    end
    local cost, grad = cost_func(nn_params)
    local numgrad = method.compute_numberical_gradient(cost_func, nn_params)

    -- Visually examine the two gradient computations.  The two columns
    -- you get should be very similar. 
    print(torch.cat(numgrad, grad, 2))
    misc.printf('The above two columns you get should be very similar.\n (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n');

    -- Evaluate the norm of the difference between two solutions.  
    -- If you have a correct implementation, and assuming you used EPSILON = 0.0001 
    -- in computeNumericalGradient.m, then diff below should be less than 1e-9
    local diff = torch.norm(numgrad-grad)/torch.norm(numgrad+grad);
    misc.printf('If your backpropagation implementation is correct, then \nthe relative difference will be small (less than 1e-9). \n\nRelative Difference: %g\n', diff);
end

function method.debug_initialize_weights(fan_out, fan_in)
    return torch.sin(torch.range(1, fan_out * (fan_in + 1)))
        :resize(fan_in + 1, fan_out):t() / 10
end

function method.compute_numberical_gradient(J, theta)
    local numgrad = torch.zeros(theta:size())
    local perturb = torch.zeros(theta:size())
    local e = 1e-4;
    for p = 1, theta:numel(), 1 do
        perturb[p] = e;

        local loss1 = J(theta - perturb)
        local loss2 = J(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2 * e)

        perturb[p] = 0
    end
    return numgrad
end

function method.fmincg(cost_func, param, option, report_num)
    local max_iter = option and option.maxIter 
    if not max_iter then
        if not option then
            option = {}
        end
        max_iter = 100
        option.maxIter = max_iter
    end

    local report_num = report_num or math_floor(max_iter / 10)
    local batch_num = math_ceil(max_iter / report_num)

    local total_num = 0
    for i = 1, batch_num, 1 do
        local this_iter = math_min(max_iter - total_num, report_num)
        if this_iter <= 0 then
            break
        end
        total_num = total_num + this_iter

        local costs
        option.maxIter = this_iter
        param, costs = optim.cg(cost_func, param, option)
        local cost = costs[#costs]
        misc.printf("Trained Iter: %s, cost: %s", total_num, cost)
    end
    return param
end

return method