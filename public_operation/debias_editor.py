'''
This code defines a class called "AddedLayers" which is the structure of the debias editor
'''
import torch
import torch.nn as nn

class AddedLayers(nn.Module):
    def __init__(self, n_feature, hidden_output, n_output):
        super(AddedLayers, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=hidden_output),
            nn.GELU(),

            nn.Linear(in_features=hidden_output, out_features=n_output),
            nn.GELU(),

            nn.Linear(in_features=n_output, out_features=hidden_output),
            nn.GELU(),

            nn.Linear(in_features=hidden_output, out_features=n_output),
        )

    def linear_forward(self, input, weight, bias=None):
        """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

        Args:
            input (Tensor): input tensor of shape :math:`(N, *, in\_features)` where `*` means any number of additional dimensions
            weight (Tensor): weight tensor of shape :math:`(out\_features, in\_features)`
            bias (Tensor): optional bias tensor of shape :math:`(out\_features)`

        Returns:
            Tensor: output tensor of shape :math:`(N, *, out\_features)`
        """
        if input.dim() == 2 and bias is not None:
            # Fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            # Using torch.tensordot to perform the operation
            output = torch.tensordot(input, weight.t(), dims=([input.dim() - 1], [0]))
            if bias is not None:
                output += bias
            ret = output
        return ret

    def forward0(self, input):
        w1 = self.model[0].weight.t()
        b1 = self.model[0].bias
        net = torch.tensordot(input, w1, [[1], [0]]) + b1
        net = self.model[1](net)
        w2 = self.model[2].weight.t()
        b2 = self.model[2].bias
        output = torch.tensordot(net, w2, [[1], [0]]) + b2

        net = self.model[3](output)
        w3 = self.model[4].weight.t()
        b3 = self.model[4].bias
        output = torch.tensordot(net, w3, [[1], [0]]) + b3

        net = self.model[5](output)
        w4 = self.model[6].weight.t()
        b4 = self.model[6].bias
        output = torch.tensordot(net, w4, [[1], [0]]) + b4

        return output

    def forward1(self, input):
        w1 = self.model[0].weight.t().to(input.dtype) # self.model[0].weight.t().to(input.dtype)
        b1 = self.model[0].bias.to(input.dtype)
        net = torch.tensordot(input, w1, [[1], [0]]) + b1
        net = self.model[1](net)
        w2 = self.model[2].weight.t().to(input.dtype)
        b2 = self.model[2].bias.to(input.dtype)
        output = torch.tensordot(net, w2, [[1], [0]]) + b2

        net = self.model[3](output)
        w3 = self.model[4].weight.t().to(input.dtype)
        b3 = self.model[4].bias.to(input.dtype)
        output = torch.tensordot(net, w3, [[1], [0]]) + b3

        net = self.model[5](output)
        w4 = self.model[6].weight.t().to(input.dtype)
        b4 = self.model[6].bias.to(input.dtype)
        output = torch.tensordot(net, w4, [[1], [0]]) + b4
        return output

    def forward2(self, input):
        input = input.to(self.model[0].weight.device)
        w1 = self.model[0].weight.to(input.dtype) # self.model[0].weight.t().to(input.dtype)
        b1 = self.model[0].bias.to(input.dtype)
        # net = torch.tensordot(input, w1, [[1], [0]]) + b1
        # net = torch.tensordot(input, w1, [[2], [1]]) + b1
        net = self.linear_forward(input, w1, b1)
        # print(net)
        net = self.model[1](net)
        w2 = self.model[2].weight.to(input.dtype)
        b2 = self.model[2].bias.to(input.dtype)
        # output = torch.tensordot(net, w2, [[1], [0]]) + b2
        # output = torch.tensordot(net, w2, [[2], [1]]) + b2
        output = self.linear_forward(net, w2, b2)
        # print(output)

        net = self.model[3](output)
        w3 = self.model[4].weight.to(input.dtype)
        b3 = self.model[4].bias.to(input.dtype)
        # output = torch.tensordot(net, w3, [[1], [0]]) + b3
        # output = torch.tensordot(net, w3, [[2], [1]]) + b3
        output = self.linear_forward(net, w3, b3)
        # print(output)

        net = self.model[5](output)
        w4 = self.model[6].weight.to(input.dtype)
        b4 = self.model[6].bias.to(input.dtype)
        # output = torch.tensordot(net, w4, [[1], [0]]) + b4
        # output = torch.tensordot(net, w4, [[2], [1]]) + b4
        output = self.linear_forward(net, w4, b4)
        # print(output)

        # w1 = self.model[0].weight.to(input.dtype)
        # b1 = self.model[0].bias.to(input.dtype)
        # net1 = torch.tensordot(input, w1, [[2], [1]]) + b1
        # net = torch.tensordot(input, w1.t(), dims=([input.dim() - 1], [0])) + b1
        # net2 = self.linear_forward(input, w1, b1)
        # print("-->net", net.shape)
        # if torch.equal(net, net2) and torch.equal(net, net1):
        #     print("same")
        # # net = torch.tensordot(input, w1, [[1], [0]]) + b1
        # net = self.model[1](net)
        # w2 = self.model[2].weight.t().to(input.dtype)
        # b2 = self.model[2].bias.to(input.dtype)
        # output = torch.tensordot(net, w2, [[2], [1]]) + b2
        # net = self.model[3](output)
        # w3 = self.model[4].weight.t().to(input.dtype)
        # b3 = self.model[4].bias.to(input.dtype)
        # output = torch.tensordot(net, w3, [[1], [0]]) + b3
        # net = self.model[5](output)
        # w4 = self.model[6].weight.t().to(input.dtype)
        # b4 = self.model[6].bias.to(input.dtype)
        # output = torch.tensordot(net, w4, [[1], [0]]) + b4

        return output

    def forward(self, input):
        input = input.to(device=self.model[0].weight.device)
        # input = input.to(dtype=self.model[0].weight.dtype)
        w1 = self.model[0].weight  # self.model[0].weight.t().to(input.dtype)
        b1 = self.model[0].bias
        # print("input", input.dtype)
        # print("w1", w1.dtype)
        # print("b1", b1.dtype)
        net = self.linear_forward(input, w1, b1)
        net = self.model[1](net)
        w2 = self.model[2].weight
        b2 = self.model[2].bias
        output = self.linear_forward(net, w2, b2)

        net = self.model[3](output)
        w3 = self.model[4].weight
        b3 = self.model[4].bias
        output = self.linear_forward(net, w3, b3)

        net = self.model[5](output)
        w4 = self.model[6].weight
        b4 = self.model[6].bias
        output = self.linear_forward(net, w4, b4)

        # output = output.to(dtype=torch.float16)

        return output

class AddedLayers_twice(nn.Module):
    def __init__(self, n_feature, hidden_output, n_output):
        super(AddedLayers_twice, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(in_features=n_feature, out_features=hidden_output),
            nn.GELU(),

            nn.Linear(in_features=hidden_output, out_features=n_output),
            nn.GELU(),

            nn.Linear(in_features=n_output, out_features=hidden_output),
            nn.GELU(),

            nn.Linear(in_features=hidden_output, out_features=n_output),
            nn.GELU(),

            nn.Linear(in_features=n_output, out_features=hidden_output),
            nn.GELU(),

            nn.Linear(in_features=hidden_output, out_features=n_output),
            nn.GELU(),

            nn.Linear(in_features=n_output, out_features=hidden_output),
            nn.GELU(),

            nn.Linear(in_features=hidden_output, out_features=n_output)
        )

    def linear_forward(self, input, weight, bias=None):
        """
        Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.

        Args:
            input (Tensor): input tensor of shape :math:`(N, *, in\_features)` where `*` means any number of additional dimensions
            weight (Tensor): weight tensor of shape :math:`(out\_features, in\_features)`
            bias (Tensor): optional bias tensor of shape :math:`(out\_features)`

        Returns:
            Tensor: output tensor of shape :math:`(N, *, out\_features)`
        """
        if input.dim() == 2 and bias is not None:
            # Fused op is marginally faster
            ret = torch.addmm(bias, input, weight.t())
        else:
            # Using torch.tensordot to perform the operation
            output = torch.tensordot(input, weight.t(), dims=([input.dim() - 1], [0]))
            if bias is not None:
                output += bias
            ret = output
        return ret

    def forward(self, input):
        input = input.to(device=self.model[0].weight.device)
        w1 = self.model[0].weight  # self.model[0].weight.t().to(input.dtype)
        b1 = self.model[0].bias
        net = self.linear_forward(input, w1, b1)

        net = self.model[1](net)
        w2 = self.model[2].weight
        b2 = self.model[2].bias
        output = self.linear_forward(net, w2, b2)

        net = self.model[3](output)
        w3 = self.model[4].weight
        b3 = self.model[4].bias
        output = self.linear_forward(net, w3, b3)

        net = self.model[5](output)
        w4 = self.model[6].weight
        b4 = self.model[6].bias
        output = self.linear_forward(net, w4, b4)

        net = self.model[7](output)
        w5 = self.model[8].weight
        b5 = self.model[8].bias
        output = self.linear_forward(net, w5, b5)

        net = self.model[9](output)
        w6 = self.model[10].weight
        b6 = self.model[10].bias
        output = self.linear_forward(net, w6, b6)

        net = self.model[11](output)
        w7 = self.model[12].weight
        b7 = self.model[12].bias
        output = self.linear_forward(net, w7, b7)

        net = self.model[13](output)
        w8 = self.model[14].weight
        b8 = self.model[14].bias
        output = self.linear_forward(net, w8, b8)

        return output