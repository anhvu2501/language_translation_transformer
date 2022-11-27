import torch

# torch.permute => return a view of the original tensor input with its dimensions permuted (hoán vị dims của tensor)
x = torch.randn(2, 3, 5)
print(x)
print(x.size()) # ([2, 3, 5])

x_permute = torch.permute(x, (2, 0, 1))
print(x_permute)
print(x_permute.size()) # => permuted to ([5, 2, 3])


# torch.view => return a new tensor with the same data as the self tensor but of a different shape
y = torch.randn(1, 2, 3, 4) # total size = 1x2x3x4 = 24 => tensor created has to have this size 
print(y.size()) # ([1, 2, 3, 4])

y_view = y.view(2, -1, 2, 3) # if we set -1 as the second param here, shape of this new tensor will be calculated as ([2, 24/(2x2x3), 2, 3])
print(y_view.size()) # ([2, 2, 2, 3])