local torch = require"torch"

local calc = {}

function calc.pinverse(A)
    local U, S, V = torch.svd(A)
    local threshold = 1e-6
    S[S:lt(threshold)] = 0
    S[S:gt(0)] = S[S:gt(0)]:pow(-1)
    return V * torch.diag(S) * U:t()
end

return calc