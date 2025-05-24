import torch 
import torch.nn as nn 
import dgl
import dgl.function as fn
from einops import repeat
import torch.nn.functional as F
import controldiffeq
#from ddefunc import OdeintAdjointMethod, AdjointFunc
#from ddeutil import _check_inputs, _flat_to_shape
import torch
import torch.backends.cudnn as cudnn
cudnn.enabled = False 
#torch.backends.cudnn.benchmark = True
class DDEFunc(nn.Module):
    def __init__(self, in_dim, hid_dim, step_size):
        super(DDEFunc, self).__init__()
        self.d = nn.Parameter(torch.ones(hid_dim))
        self.w = nn.Parameter(torch.eye(hid_dim))
        self.t = None 
        self.step_size = step_size
        self.auto_regress_length = 5
        self.wy = nn.Parameter(torch.FloatTensor(hid_dim, hid_dim))
        self.wc = nn.Parameter(torch.FloatTensor(in_dim, hid_dim))
        self.trans_y = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                     #nn.ReLU(),
                                     #nn.Linear(hid_dim, hid_dim)
                                     )
        self.trans_control = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                    #nn.ReLU(),
                                    #nn.Linear(hid_dim, hid_dim)
                                    )
        self.memory_size = 128
        self.memory = nn.Parameter(torch.FloatTensor(self.memory_size, hid_dim))
        self.auto_regress = nn.Parameter(torch.FloatTensor(self.auto_regress_length))
        self.num_heads=4
        self.self_attn = nn.MultiheadAttention(embed_dim=hid_dim, num_heads=self.num_heads, batch_first=True)
        self.gate = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())
        self.gate_out = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.Sigmoid())
        #self.conv = nn.Conv1d(64, 64, kernel_size=3,padding=1)
        self.norm = nn.BatchNorm1d(hid_dim)
        self.reset_param()
    def reset_param(self):
        nn.init.kaiming_normal_(self.wc)
        nn.init.kaiming_normal_(self.wy)
        nn.init.kaiming_normal_(self.memory)
        nn.init.normal_(self.auto_regress)

    
    def message(self, edges):
        delay = edges.data['delay']
        catch = (self.t + delay)/self.step_size
        #print(f"Catch before clamp: {catch}")
        catch[catch < 0] = 0    # if history state doesn't exist, use the initial state
        #print(f"Catch after clamp: {catch}")
        hist = edges.src['state']
        catch = catch.to(torch.long)
        choose_state = hist[range(hist.shape[0]), catch]
        #print(f"Chosen history state: {hist[range(hist.shape[0]), catch]}")
        '''
        weight = edges.data['w'].reshape(-1, 1, 1)
        return {'m': weight * choose_state}
        '''
        choose_state = choose_state.permute(1, 0, 2)  # 现在形状为 (batch, num_node, hid)

        attn_output, attn_weights = self.self_attn(choose_state, choose_state, choose_state)

        attn_output = attn_output.permute(1, 0, 2)  #

        weight = edges.data['w'].reshape(-1, 1, 1)
        message = weight * (choose_state.permute(1, 0, 2) + attn_output)  
        return {'m': message}
    def forward(self, g, x, funcx, t):
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        # use the former state as a temporary 
        g.ndata['state'][:, int(t/self.step_size)] = x 
        self.t = t 
        g.update_all(self.message, fn.sum('m', 's'))
        y = g.ndata['s']     
        gate = self.gate(y)
        y = (1 - gate) * (y - x)
        y = torch.einsum('ijk, kl->ijl', y, w)
        gate = self.gate(y)
        y = (1 - gate) * (y - x)
        '''
        y = y.transpose(0, 1)  # 转换为 [batch_size, num_node, hid_dim]
        y = y.transpose(1, 2)  # 转换为 [batch_size, hid_dim, num_node] 适配卷积输入
        conv_out = self.conv(y)  # 卷积操作
        conv_out = conv_out.transpose(1, 2)  # 恢复为 [batch_size, num_node, hid_dim]
        y = conv_out.transpose(0, 1)
        '''
        if funcx is not None:
            dx_dt = funcx(t).permute(1, 0, 2) 
            dx_dt = self.trans_control(dx_dt) 
            attention_weights = torch.softmax(torch.matmul(dx_dt, self.memory.t()), dim=2)
            attention_output = torch.matmul(attention_weights, self.memory) 
            y = y * attention_output
        y = self.trans_y(torch.relu(y))
        y = self.gate_out(y) * y 
        return y

class Euler():
    def __init__(self, func, funcx, y0,step_size):
        self.func = func
        self.funcx = funcx
        self.y0 = y0
        self.step_size = step_size 
        self.grid_constructor = self._grid_constructor_from_step_size(step_size)

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(t):
            t_reverse = None
            if t[0] > t[1]:
                t = t.flip(0)
                t_reverse = True 
            
            start_time = t[0]
            end_time = t[-1]
            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]
            if t_reverse:
                t_infer = t_infer.flip(0)
            return t_infer
        return _grid_constructor

    def _step_func(self, dt, g, y0, t0):
        f0 = self.func(g, y0, self.funcx, t0)
        return dt * f0

    def integrate(self, g, t):
        time_grid = self.grid_constructor(t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            dy = self._step_func(dt, g, y0, t0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1
            y0 = y1
        return solution
    
    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)


def ddeint(func, g, y0, funcx, t,step_size):
    shapes, func, y0 = controldiffeq._check_inputs(func, y0)
    solver = Euler(func, funcx, y0, step_size=step_size)
    solution = solver.integrate(g, t)
    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution


class DDEBlock(nn.Module):
    def __init__(self, in_dim, hid_dim, step_size):
        super(DDEBlock, self).__init__()
        self.ddefunc = DDEFunc(in_dim, hid_dim, step_size)
        self.step_size = step_size
    def forward(self, g, y0, funcx, t):
        ans = ddeint(self.ddefunc, g, y0, funcx, t, self.step_size)
        return ans