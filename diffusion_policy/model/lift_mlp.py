import sys
sys.path.append('../')
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np

from escnn import gspaces
from escnn import nn
from escnn import group
"""
x = torch.randn(1,23)
print("x is:", x)
gsapce=gspaces.no_base_space(group.so2_group())
in_type = nn.FieldType(gsapce, [gsapce.irrep(1)] +[gsapce.trivial_repr]+3*[gsapce.trivial_repr]+[gsapce.irrep(1)]+ 2*[gsapce.trivial_repr]+[gsapce.irrep(1)] +1*[gsapce.trivial_repr]+ 1*[gsapce.irrep(1)] +2*[gsapce.trivial_repr]+[gsapce.irrep(1)] )
x=in_type(x)
for g in gsapce.testing_elements:
    x_transformed = x.transform(g)
    print("g=",g)
    print('x transformed=',x_transformed)
"""
class LiftLowDimMLP(nn.EquivariantModule):

    def __init__(self):

        super(LiftLowDimMLP, self).__init__()

        # the model is equivariant to the group O(2)
        self.G = group.so2_group()

        # since we are building an MLP, there is no base-space
        self.gspace = gspaces.no_base_space(self.G)

        # the input contains the coordinates of a point in the 2D space
        self.in_type = nn.FieldType(self.gspace, [self.gspace.irrep(1)] +[self.gspace.trivial_repr]+[self.gspace.irrep(1)]+ 4*[self.gspace.trivial_repr]+[self.gspace.irrep(1)] +[self.gspace.trivial_repr]+[self.gspace.irrep(1)] +1*[self.gspace.trivial_repr]+ 1*[self.gspace.irrep(1)] +4*[self.gspace.trivial_repr]+2*[self.gspace.trivial_repr] )

        # Layer 1
        # We will use the regular representation of SO(2) acting on signals over SO(2) itself, bandlimited to frequency 1
        # Most of the comments on the previous SO(3) network apply here as well

        activation1 = nn.FourierELU(
            self.gspace,
            channels=36, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=1).irreps, # include all frequencies up to L=1
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 6 equally distributed points
            type='regular', N=6,
        )

        # map with an equivariant Linear layer to the input expected by the activation function, apply batchnorm and finally the activation
        self.block1 = nn.SequentialModule(
            nn.Linear(self.in_type, activation1.in_type),
            nn.IIDBatchNorm1d(activation1.in_type),
            activation1,
        )

        # Repeat a similar process for a few layers

        # 8 signals, bandlimited up to frequency 3
        activation2 = nn.FourierELU(
            self.gspace,
            channels=36, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=10).irreps, # include all frequencies up to L=3
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 16 equally distributed points
            type='regular', N=16,
        )
        self.block2 = nn.SequentialModule(
            nn.Linear(self.block1.out_type, activation2.in_type),
            nn.IIDBatchNorm1d(activation2.in_type),
            activation2,
        )

        # 8 signals, bandlimited up to frequency 3
        activation3 = nn.FourierELU(
            self.gspace,
            channels=36, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=8).irreps, # include all frequencies up to L=3
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 16 equally distributed points
            type='regular', N=16,
        )
        self.block3 = nn.SequentialModule(
            nn.Linear(self.block2.out_type, activation3.in_type),
            nn.IIDBatchNorm1d(activation3.in_type),
            activation3,
        )

        # 5 signals, bandlimited up to frequency 2
        activation4 = nn.FourierELU(
            self.gspace,
            channels=36, # specify the number of signals in the output features
            irreps=self.G.bl_regular_representation(L=5).irreps, # include all frequencies up to L=2
            inplace=True,
            # the following kwargs are used to build a discretization of the circle containing 12 equally distributed points
            type='regular', N=12,
        )
        self.block4 = nn.SequentialModule(
            nn.Linear(self.block3.out_type, activation4.in_type),
            nn.IIDBatchNorm1d(activation4.in_type),
            activation4,
        )

        # Final linear layer mapping to the output features
        # the output is a 2-dimensional vector rotating with frequency 2
        # self.out_type = nn.FieldType(self.gspace, [self.gspace.irrep(1)] +[self.gspace.trivial_repr]+6*[self.gspace.trivial_repr]+ [self.gspace.trivial_repr])
        self.out_type = nn.FieldType(self.gspace, [self.gspace.irrep(1)] +[self.gspace.trivial_repr]+[self.gspace.irrep(1)]+4*[self.gspace.trivial_repr]+ [self.gspace.trivial_repr])
        self.block5 = nn.Linear(self.block4.out_type, self.out_type)

    def forward(self, x):

        # check the input has the right type
        # apply each equivariant block

        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
        device=x.device
        x=self.in_type(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x.to(device).tensor
    def evaluate_output_shape(self, input_shape: tuple):
        shape = list(input_shape)
        assert len(shape) ==2, shape
        assert shape[1] == self.in_type.size, shape
        shape[1] = self.out_type.size
        return shape
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = LiftLowDimMLP().to(device)
# np.set_printoptions(linewidth=10000, precision=4, suppress=True)

# model.eval()



# # generates B random points in 3D and wrap them in a GeometricTensor of the right type
# x = torch.randn(1,23)

# print('##########################################################################################')
# with torch.no_grad():
#     y = model(x.to(device)).to(device)
#     print("Outputs' magnitudes")
#     print(torch.linalg.norm(y.tensor.cpu(), dim=0).numpy().reshape(-1))
#     print('##########################################################################################')
#     print("Errors' magnitudes")
#     for r in range(8):
#         # sample a random rotation
#         g = model.G.sample()

#         x_transformed = g @ model.in_type(x)
#         x_transformed = x_transformed.to(device)

#         y_transformed = model(x_transformed.tensor).to(device)

#         # verify that f(g@x) = g@f(x)=g@y
#         print(torch.linalg.norm(y_transformed.tensor.cpu() - (g@y).tensor.cpu(), dim=0).numpy().reshape(-1))

# print('##########################################################################################')
# print()
# def ea_to_quo(x):
#         x=x.tensor
#         angle_z = np.arctan2(x[0][3], x[0][4])  # z轴旋转角度
#         angle_y = np.arctan2(x[0][5], x[0][6])  # y轴旋转角度
#         angle_x = np.arctan2(x[0][7], x[0][8])  # x轴旋转角度
#         rotation = R.from_euler('zyx', [angle_z, angle_y, angle_x])
#         quaternion = rotation.as_quat()  # 返回格式为 [x, y, z, w]
#         x_ = np.concatenate((x[0][:3], quaternion))
#         x_tensor = torch.from_numpy(x_)
#         return x_tensor
