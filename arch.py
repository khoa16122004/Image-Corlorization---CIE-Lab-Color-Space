import torch
import torch.nn as nn
import torch.nn.functional as F



def get_architecture(architecture: str, objective: str="reconstruction"):
    if architecture == "unet":
        model = UNet()
        
    elif architecture == "unetDecoder":
        return UNetDecoder()
    elif architecture == "discriminator":
        model = Discriminator()
    elif architecture == "simple_CNN":
        model = ColorizationModel(objective)
    elif architecture == "upscale":
        model = Upscale()
    elif architecture == "wgan":
        if objective == "reconstruction":
            G = Generator(feature_maps=96, output_channels=2)
            D = Discriminator(96, 96, input_channels=2)
        else:
            G = Generator(feature_maps=96, output_channels=361)
            D = Discriminator(96, 96, input_channels=1)
        return G, D
    
    return model

class ColorizationModel(nn.Module):
    def __init__(self, objective="reconstruction"):
        super(ColorizationModel, self).__init__()
        self.objective = objective
        if self.objective == "reconstruction":
            self.Q = 2
            
        elif self.objective == "classification":
            self.Q = 361
            
        elif self.objective == "upscale":
            self.Q = 3

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.Q, kernel_size=3, stride=1, padding=1)
        )

        if self.objective == "reconstruction":
            self.decoder.add_module('tanh', nn.Tanh())

        elif self.objective == "classificaiton":
            self.decoder.add_module('relu', nn.ReLU())
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder part
        for feature in features:
            self.encoder.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder part
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(DoubleConv(feature * 2, feature))

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        print(x.shape)
        
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_conv(x)




    def forward(self, latent):
        # Chuyển latent vector thành tensor 3D có kích thước phù hợp
        x = self.fc(latent)
        x = x.view(-1, 64, 8, 8)  # Chuyển thành tensor có shape (batch_size, 64, 8, 8)

        # Các lớp convolution
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Upsample (tăng kích thước ảnh)
        x = self.upconv1(x)  # Tăng lên 16x16
        x = self.upconv2(x)  # Tăng lên 32x32
        x = self.upconv3(x)  # Tăng lên 64x64
        x = self.upconv4(x)  # Tăng lên 128x128

        # Lớp cuối cùng để xuất ra hình ảnh với số kênh mong muốn
        x = self.final_conv(x)

        return x
    
class UNetDecoder(nn.Module):
    def __init__(self, in_channels=64, out_channels=3, latent_dim=128):
        super(UNetDecoder, self).__init__()
        self.latent_dim = latent_dim
        
        self.fc = nn.Linear(latent_dim, in_channels * 8 * 8) 
        
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)  # Tăng lên 16x16
        self.upconv2 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)  # Tăng lên 32x32
        self.upconv3 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)  # Tăng lên 64x64
        self.upconv4 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)  # Tăng lên 128x128
        
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, latent):

        x = self.fc(latent)
        x = x.view(-1, 64, 8, 8) 


        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.upconv1(x)  
        x = self.upconv2(x)  
        x = self.upconv3(x)  
        x = self.upconv4(x)  

        x = self.final_conv(x)

        return x
    
class Generator(nn.Module):
    def __init__(self, input_channels=1, output_channels=361, feature_maps=256):
        super(Generator, self).__init__()
        self.input_channels = input_channels 
        self.output_channels = output_channels
        self.feature_maps = feature_maps
        
        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True)
        )

        self.generator = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            nn.Conv2d(feature_maps, output_channels, kernel_size=3, stride=1, padding=1)
        )
        self.output = nn.Tanh()

    def forward(self, x):
        x = self.input_layer(x)
        x = self.generator(x)
        return self.output(x)


class Discriminator(nn.Module):
    def __init__(self, H=128, W=128, input_channels = 1):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.discriminator = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, kernel_size=4, stride=2, padding=1),  # (B, 1, H, W) -> (B, 64, H/2, W/2)
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> (B, 128, H/4, W/4)
             nn.InstanceNorm2d(128,  affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # -> (B, 256, H/8, W/8)
            nn.InstanceNorm2d(256,  affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # -> (B, 512, H/16, W/16)
            nn.InstanceNorm2d(512,  affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),  
            nn.Linear(512 * (H // 16) * (W // 16), 4), 
        )

    def forward(self, x):
        x = self.discriminator(x)
        x = x.view(-1, 1, 2, 2)
        return x



if __name__ == "__main__":
    x = torch.rand(5, 1, 96, 96)
    model = ColorizationModel()
    print(model(x).shape)