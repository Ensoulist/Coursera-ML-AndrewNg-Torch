local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"
local loader = require"utils.loader"
local optim = require"optim"

local method = {}

function method.find_closest_centroids(X, centroids)
    local idx = {}
    local t_c 
    if type(centroids) == "table" then
        t_c = torch.Tensor(centroids)
    else
        t_c = centroids
    end
    for i = 1, X:size(1), 1 do
        local sample = X[i]
        local min = math.huge
        for j = 1, t_c:size(1), 1 do
            local dist = torch.dist(sample, t_c[j])
            if dist < min then
                min = dist
                idx[i] = j
            end
        end
    end
    return torch.Tensor(idx)
end

function method.compute_centroids(X, idx, K)
    local centroids = torch.zeros(K, X:size(2))
    local k_num = {}
    for i = 1, X:size(1), 1 do
        local ci = idx[i]
        centroids[ci] = centroids[ci] + X[i]
        k_num[ci] = (k_num[ci] or 0) + 1
    end
    for i = 1, K, 1 do
        centroids[i] = centroids[i] / (k_num[i] or 1)
    end
    return centroids
end

function method.run_k_means(X, initial_centroids, max_iters, plot_grogress)
    local centroids
    if type(initial_centroids) == "table" then
        centroids = torch.Tensor(initial_centroids)
    else
        centroids = initial_centroids
    end

    local n = centroids:size(2)
    local k = centroids:size(1)
    local idx
    local history
    if plot_grogress then
        -- make it max_iters + 1 for including initial values
        history = torch.Tensor(k, max_iters + 1, n)
    end
    for i = 0, max_iters, 1 do
        if i > 0 then
            idx = method.find_closest_centroids(X, centroids)
            centroids = method.compute_centroids(X, idx, k)
        end
        if history then
            for j = 1, k, 1 do
                history[j][i + 1] = centroids[j]
            end
        end
    end

    if history then
        method.plot_progress_k_means(X, idx, history, k)
    end
    return centroids, idx
end

function method.plot_progress_k_means(X, idx, history, k)
    local m = X:size(1)
    local n = X:size(2)

    local plot_tbl = {}
    for i = 1, k, 1 do
        local mask = torch.eq(idx, i)
        table.insert(plot_tbl, {"group " .. i, X[{{}, 1}][mask], X[{{}, 2}][mask], "+"})
    end
    for i = 1, k, 1 do
        local this_his = history[i]
        table.insert(plot_tbl, {"progress" .. i, this_his[{{}, 1}], this_his[{{}, 2}], "+-"})
    end

    plot.pngfigure("k_means_progress.png")
    plot.plot(plot_tbl)
    plot.plotflush()
    plot.close()
    misc.printf("see k_means_progress.png")
end

function method.k_means_init_centroids(X, K)
    local idx = torch.randperm(X:size(1))
    local centroids = torch.zeros(K, X:size(2))
    for i = 1, K, 1 do
        centroids[i] = X[idx[i]]
    end
    return centroids
end

return method