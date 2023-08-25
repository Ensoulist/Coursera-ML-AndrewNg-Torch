local torch = require"torch"
local plot = require"gnuplot"
local misc = require"utils.misc"
local calc = require"utils.calc"
local loader = require"utils.loader"
local optim = require"optim"

local method = {}

misc.extend_method(method, "ex3-neural_network.method3")
misc.extend_method(method, "ex5-bias_vs_variance.method5")

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

function method.compute_centroids2(X, idx, K)
    local centroids = torch.zeros(K, X:size(2))
    for i = 1, K, 1 do
        local this_idx = torch.eq(idx, i)
        local this_X = X:index(1, torch.nonzero(this_idx)[{{}, 1}])
        centroids[i] = this_X:sum(1) / this_X:size(1)
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
            centroids = method.compute_centroids2(X, idx, k)
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

function method.plot(plot_tbl, file_name, labels, axis, plot_method)
    file_name = file_name or "plot.png"
    plot.pngfigure(file_name)
    local plot_method = plot_method or "plot"
    plot[plot_method](plot_tbl)
    if axis then
        plot.axis(axis)
    end
    if labels then
        plot.xlabel(labels[1])
        plot.ylabel(labels[2])
    end
    plot.plotflush()
    plot.close()
    misc.printf("plot. see %s", file_name)
end

function method.pca(X)
    local SGM = X:t() * X / X:size(1)
    return torch.svd(SGM)
end

function method.draw_line(plot_tbl, pt1, pt2)
    if pt1:dim() > 1 then
        pt1 = pt1[1]
    end
    if pt2:dim() > 1 then
        pt2 = pt2[1]
    end
    table.insert(plot_tbl, {"", torch.Tensor({pt1[1], pt2[1]}), 
        torch.Tensor({pt1[2], pt2[2]}), "-"}) 
    return plot_tbl
end

function method.project_data(X, U, K)
    return X * U[{{}, {1, K}}]
end

function method.recover_data(X, U, K)
    return X * U[{{}, {1, K}}]:t()
end

return method