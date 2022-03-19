from torch import nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        
#         self.sequence_conv_1 = nn.Sequential(
#             nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(5,5), padding=2, stride=1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=4, out_channels=16, kernel_size=(5,5), padding=2, stride=1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=2)
#         )
        
#         self.sequence_conv_2 = nn.Sequential(
#             nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5,5), padding=2, stride=1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.Conv2d(in_channels=16, out_channels=64, kernel_size=(5,5), padding=2, stride=1, padding_mode="reflect"),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(2,2), stride=2)
#         )
        
#         self.sequence_deconv = nn.Sequential(
            
#         )

        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=(5,5), padding=2, stride=1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(in_channels=4, out_channels=1, kernel_size=(1,1), padding=0, stride=1),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        return self.sequence(x) * 255
    
    
    