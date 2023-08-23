local string_format = string.format
local string_len = string.len
local string_sub = string.sub
local math_ceil = math.ceil

local cjson = require'cjson'

local misc = {}

function misc.printf(_s, ...)
    print(string_format(_s, ...))
end

function misc.dbg_tbl(...)
    local args = {...}
    local tbl 
    if #args == 1 then
        tbl = args[1]
    else
        tbl = args
    end

    local str = cjson.encode(tbl)
    local sp_max = 3000
    local sp_num = math_ceil(string_len(str) / sp_max)
    for i = 1, sp_num, 1 do
        local sub = string_sub(str, sp_max * (i - 1) + 1, sp_max * i)
        misc.printf("$$--------- NO.%s: %s -----------", i, sub)
    end
end

function misc.clear_screen()
    os.execute("clear")
end

function misc.pause()
    io.stdin:read()
end

function misc.input(_hint)
    if _hint then
        print(_hint)
    end
    return io.stdin:read()
end

function misc.extend_method(_method, _path)
    local comming = require(_path)
    for k, v in pairs(comming) do
        _method[k] = v
    end
    return _method
end

function misc.table_check_key(_tbl, _key)
    local val = _tbl[_key]
    if not val then
        val = {}
        _tbl[_key] = val
    end
    return val
end

return misc
