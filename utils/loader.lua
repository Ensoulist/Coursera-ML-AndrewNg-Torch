local torch = require"torch"
local matio = require"matio"

local table_insert = table.insert
local string_char = string.char

local loader = {}

function loader.load_from_txt(_path)
    local file = torch.DiskFile(_path, "r")
    file:quiet()
    -- read line
    local all = {}
    local row = {}
    while true do
        local num = file:readDouble()
        if file:hasError() then
            break
        end
        table_insert(row, num)

        local pos = file:position()
        local try = file:readChar()
        if string_char(try) ~= "," then
            file:seek(pos)
            table_insert(all, row)
            row = {}
        end
    end
    file:close()
    return torch.Tensor(all)
end

function loader.load_from_mat(_file_name)
    return matio.load(_file_name)
end

function loader.data_to_svm(X, y)
    local svm_data = {}
    if X:dim() == 1 then
        X = X:view(1, X:numel())
    end
    local n = X:size(2)
    local idx_arr = {}
    for i = 1, n, 1 do
        table_insert(idx_arr, i)
    end

    local y_dim = y:dim()
    for i = 1, X:size(1), 1 do
        local lable = y_dim == 1 and y[i] or y[i][1]
        local idxcs = torch.IntTensor(idx_arr)
        table_insert(svm_data, {lable, {idxcs, X[i]:float()}})
    end
    return svm_data
end

return loader