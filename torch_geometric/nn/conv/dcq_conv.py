from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Hardsigmoid, Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import OptTensor
from torch_geometric.utils import get_laplacian


class DCQConv(MessagePassing):
    r"""The convex-concave quadratic spectral graph convolutional operator
    from the `"Convex-Concave Quadratic Spectral Filtering for Graph Neural
    Networks" <https://arxiv.org/abs/2606.24956>`_ paper.

    .. math::
        \mathbf{X}^{\prime} = \mathbf{\Theta} \cdot
        \mathrm{BN}\left( \sum_{k=0}^{3} g_k(\mathbf{X}) \odot
        \mathbf{H}_k \right)

    where :math:`\mathbf{H}_k` are four parallel low-order polynomial
    spectral channels computed on the symmetric-normalized Laplacian
    :math:`\mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2} \mathbf{A}
    \mathbf{D}^{-1/2}` (with inserted self-loops), and :math:`g_k` are
    per-node, per-feature adaptive gates. The channels are

    .. math::
        \mathbf{H}_0 &= s_0 \frac{\beta_0}{\lambda_0^2} \left(
        \mathbf{L} - \lambda_0 \mathbf{I} \right)^2 \mathbf{X}

        \mathbf{H}_1 &= s_1 \frac{\beta_1}{\lambda_1^2}
        \mathbf{L}^2 \mathbf{X}

        \mathbf{H}_2 &= -\frac{4 \beta_2}{\lambda_2^2} \mathbf{L} \left(
        \mathbf{L} - \lambda_2 \mathbf{I} \right) \mathbf{X}

        \mathbf{H}_3 &= \beta_3 \mathbf{X}

    where :math:`\lambda_t` and :math:`\beta` are learnable parameters
    clamped to :obj:`[0, 2]` and :obj:`[0.01, 1]`, respectively.
    :math:`s_0 \in \{+1, -1\}` and :math:`s_1 \in \{+1, -1\}` select the
    convex (:obj:`"cvx"`) or concave (:obj:`"ccv"`) variant of the low-pass
    and high-pass channels via :obj:`opt`. Each gate is given by

    .. math::
        g_k(\mathbf{X}) = \mathrm{Hardsigmoid}\left(
        \mathbf{X} \mathbf{W}_k \right)

    so that the fusion of the four channels is adaptive on a per-node,
    per-feature basis.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        opt (list, optional): The convex/concave variant of the low-pass and
            high-pass channels, either :obj:`"cvx"` or :obj:`"ccv"` per
            entry. (default: :obj:`["cvx", "cvx"]`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})`,
          edge indices :math:`(2, |\mathcal{E}|)`,
          edge weights :math:`(|\mathcal{E}|)` *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})`
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        opt: Optional[List[str]] = None,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.opt = opt if opt is not None else ['cvx', 'cvx']
        assert all(o in ['cvx', 'ccv'] for o in self.opt), 'Invalid option'

        self.lambda_t = Parameter(torch.tensor([2.0, 2.0, 2.0]))
        self.beta = Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]))

        self.gates = torch.nn.ModuleList([
            Linear(in_channels, in_channels, bias=False,
                   weight_initializer='glorot') for _ in range(4)
        ])
        self.act = Hardsigmoid()
        self.bn = BatchNorm1d(in_channels)
        self.lin = Linear(in_channels, out_channels, bias=True,
                          weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        for gate in self.gates:
            gate.reset_parameters()
        self.bn.reset_parameters()
        self.lin.reset_parameters()
        self.lambda_t.data.fill_(2.0)
        self.beta.data.fill_(1.0)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
    ) -> Tensor:
        # Clamp the learnable spectral parameters to their valid ranges
        # (mirrors the pre-forward hook of the reference implementation).
        with torch.no_grad():
            self.lambda_t.clamp_(min=0.0, max=2.0)
            self.beta.clamp_(min=0.01, max=1.0)

        edge_index, norm = get_laplacian(  # yapf: disable
            edge_index, edge_weight, normalization='sym', dtype=x.dtype,
            num_nodes=x.size(self.node_dim))

        # propagate_type: (x: Tensor, norm: Tensor)
        lx = self.propagate(edge_index, x=x, norm=norm)
        l2x = self.propagate(edge_index, x=lx, norm=norm)

        lam = self.lambda_t
        beta = self.beta
        sign_low = -1.0 if self.opt[0] == 'ccv' else 1.0
        sign_high = -1.0 if self.opt[1] == 'ccv' else 1.0

        h0 = sign_low * (beta[0] / lam[0]**2) * (
            l2x - 2.0 * lam[0] * lx + lam[0]**2 * x)
        h1 = sign_high * (beta[1] / lam[1]**2) * l2x
        h2 = -(4.0 * beta[2] / lam[2]**2) * (l2x - lam[2] * lx)
        h3 = beta[3] * x

        out = (self.act(self.gates[0](x)) * h0 +
               self.act(self.gates[1](x)) * h1 +
               self.act(self.gates[2](x)) * h2 +
               self.act(self.gates[3](x)) * h3)

        return self.lin(self.bn(out))

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, opt={self.opt})')
