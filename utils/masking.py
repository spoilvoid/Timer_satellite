import torch


class TriangularCausalMask():
    def __init__(self, batch_size, n_tokens, device="cpu"):
        mask_shape = [batch_size, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask
    

class TimerMultivariateMask():
    def __init__(self, batch_size, n_vars, n_tokens, device="cpu"):
        mask_shape = [batch_size, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.ones((n_vars, n_vars), dtype=torch.bool).to(device)
            self._mask2 = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
            self._mask = torch.kron(self._mask1, self._mask2)
    @property
    def mask(self):
        return self._mask


class TimerCovariateMask():
    def __init__(self, B, n_vars, n_tokens, device="cpu"):
        mask_shape = [B, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.eye(n_vars, dtype=torch.bool).to(device)
            self._mask2 = torch.tril(torch.ones(mask_shape, dtype=torch.bool)).to(device)
            self._mask = ~torch.kron(self._mask1, self._mask2)
            self._mask[:, :, -n_tokens:, :-n_tokens] = False
            
    @property
    def mask(self):
        return self._mask
    

class TimerMultivariateWithFutureCovariateMask():
    def __init__(self, batch_size, n_vars, n_tokens, n_pred_vars=None, device="cpu"):
        if n_pred_vars is None:
            n_pred_vars = 1
        mask_shape = [batch_size, 1, n_tokens, n_tokens]
        with torch.no_grad():
            self._mask1 = torch.eye(n_vars, dtype=torch.bool).to(device)
            self._mask1[:n_pred_vars, :n_pred_vars] = True
            self._mask2 = torch.tril(torch.ones(mask_shape, dtype=torch.bool)).to(device)
            self._mask = ~torch.kron(self._mask1, self._mask2)
            
    @property
    def mask(self): 
        return self._mask