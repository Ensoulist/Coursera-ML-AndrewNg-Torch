local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"
local loader = require"utils.loader"
local optim = require"optim"
local nn = require"nn"
local svm = require"svm"
local stem = require"stem"

local table_insert = table.insert
local table_sort = table.sort
local math_floor = math.floor
local math_ceil = math.ceil
local math_sqrt = math.sqrt
local math_min = math.min
local string_format = string.format

local method = {}

misc.extend_method(method, "ex2-logistic_regression.method2")

function method.svm_train(X, y, C, kernel, tol, max_passes, sigma)
    local svm_data = loader.data_to_svm(X, y)
    local option = string_format("-t %d", kernel)
    if C then
        option = string_format("%s -c %f", option, C)
    end
    if tol then
        option = string_format("%s -e %f", option, tol)
    end
    if sigma then
        local gamma = 1 / (2 * sigma * sigma)
        option = string_format("%s -g %f", option, gamma)
    end

    local svm_model = libsvm.train(svm_data, option)
    return svm_model
end

function method.svm_predict(model, X, y)
    local svm_data = loader.data_to_svm(X, y or torch.zeros(X:size(1)))
    return libsvm.predict(svm_data, model)
end

function method.svm_predict_weight(model, vocab_list)
    local length = #vocab_list
    local X = torch.eye(length)
    local y = torch.zeros(length)
    local svm_data = loader.data_to_svm(X, y)
    local _, _, val = libsvm.predict(svm_data, model)
    local rlt = {}
    for i = 1, length, 1 do
        local weight = val[i][1]
        table_insert(rlt, {i, weight})
    end
    table_sort(rlt, function(fst, scd)
        return fst[2] > scd[2]
    end)
    return rlt
end

function method.visualize_boundary_linear(X, y, model, file_name)
    local plot_tbl = method.plot_data(X, y, true, {"", ""}, {"", ""}, file_name)

    local u = torch.linspace(X[{{}, 1}]:min(), X[{{}, 1}]:max(), 50)
    local v = torch.linspace(X[{{}, 2}]:min(), X[{{}, 2}]:max(), 50)
    local z = torch.zeros(u:size(1), v:size(1))
    for i = 1, u:size(1), 1 do
        for j = 1, v:size(1), 1 do
            local svm_data = loader.data_to_svm(torch.Tensor({{u[i], v[j]}}), torch.Tensor({1}))
            local _, _, val = libsvm.predict(svm_data, model)
            z[i][j] = val
        end
    end
    local plot_x, plot_y = method.contour(z, u, v, 0.5)
    if plot_x:dim() > 0 and plot_y:dim() > 0 then
        table_insert(plot_tbl, {"Decision Boundary", plot_x, plot_y, "+"})
    end

    plot.pngfigure(file_name or "vbl.png")
    plot.plot(plot_tbl)
    plot.plotflush()
end

function method.gaussian_kernel(x1, x2, sigma)
    return torch.exp(-torch.pow(x1 - x2, 2):sum() / (2 * sigma * sigma))
end

function method.dataset3_params(X, y, Xval, yval)
    local choices = {0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100}
    local choosen_C, choosen_sigma
    local accuracy_max = 0
    for _, C in ipairs(choices) do
        for _, sigma in ipairs(choices) do
            local model = method.svm_train(X, y, C, 2, 0.0001, 20, sigma)
            local val_data = loader.data_to_svm(Xval, yval)
            local _, accuracy = libsvm.predict(val_data, model)
            if accuracy[1] > accuracy_max then
                accuracy_max = accuracy[1]
                choosen_C = C
                choosen_sigma = sigma
            end
        end
    end
    return choosen_C, choosen_sigma
end

function method.process_email(email_contents)
    local vocab_list = method.get_vocab_list()

    -- =========================== Preprocess Email ===========================
    -- Lower case
    email_contents = string.lower(email_contents) 
    -- Strip all HTML
    email_contents = string.gsub(email_contents, "<[^<>]+>", " ")
    -- Handle Numbers
    email_contents = string.gsub(email_contents, "%d+", "number")
    -- Handle URLS
    email_contents = string.gsub(email_contents, "http[s]?://[^%s]*", "httpaddr")
    -- Handle Email Addresses
    email_contents = string.gsub(email_contents, "[^%s]+@[^%s]+", "emailaddr")
    -- Handle $ sign
    email_contents = string.gsub(email_contents, "[$]+", "dollar")

    -- ========================== Tokenize Email ===========================
    misc.printf("==== Processed Email ====")
    local word_indices = {}
    local str_tbl = {}
    for v in string.gmatch(email_contents, "[^ @$/#.%-:&*+=%[%]?!(){},'\">_<;%%\t\n]+") do
        local str = string.gsub(v, "[^a-zA-Z0-9]", "")
        str = method.proter_stemmer(str)
        table_insert(str_tbl, str)
        if vocab_list[str] then
            table_insert(word_indices, vocab_list[str])
        end
    end
    misc.printf(table.concat(str_tbl, " "))
    misc.printf('\n=========================');
    return word_indices
end

function method.get_vocab_list(need_arr)
    local f = io.open("vocab.txt", "r")
    local str = f:read("*a")
    local set = {}
    local idx = 1
    local arr 
    if need_arr then
        arr = {}
    end
    for w in string.gmatch(str, "[a-z]+") do
        set[w] = idx
        idx = idx + 1
        if need_arr then
            table_insert(arr, w)
        end
    end
    return set, arr
end

function method.proter_stemmer(str)
    return stem.stem(str)
end

function method.email_features(word_indices)
    local n = 1899
    local x = torch.zeros(n)
    for _, idx in ipairs(word_indices) do
        x[idx] = 1
    end
    return x
end

return method
