local torch = require"torch"

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

function loader.load_from_mat()
end

return loader