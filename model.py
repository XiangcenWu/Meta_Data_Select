import torch
import torch.nn as nn


class SelectionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        
        self.down_sample = DownSample([2, 32, 128, 256, 512, 1024])


        self.pool_to_vector = nn.AvgPool3d(2)


        self.attention = nn.Sequential(
            BatchAttentionModulle(1024, 2, 2),
            BatchAttentionModulle(1024, 2, 2, 512),

            BatchAttentionModulle(512, 2, 1),
            BatchAttentionModulle(512, 2, 1, 128),

            BatchAttentionModulle(128, 2, 1),
            BatchAttentionModulle(128, 2, 1, 1),
        )

        # self.without_sequence = nn.Sequential(
        #     nn.Linear(1024, 1),
        #     nn.Sigmoid(),
        # )


    def forward(self, x):
        skip_list = self.down_sample(x)
        test_feature_map = skip_list
        feature = self.pool_to_vector(test_feature_map).flatten(1) # [B, feature]
        output = self.attention(feature)
        
        return torch.sigmoid(output)

    





        


class DownSample(nn.Module):

    def __init__(self, feature_list=[2, 32, 128, 256, 512, 1024], unet_shape=False):
        super().__init__()
        self.feature_list = feature_list
        self.unet_shape = unet_shape
        self.init_bn = nn.BatchNorm3d(2)
        self.down_0 = self._make_down(0)
        self.down_1 = self._make_down(1)
        self.down_2 = self._make_down(2)
        self.down_3 = self._make_down(3)
        self.down_4 = self._make_down(4)

        


    def forward(self, x):
        if self.unet_shape == True:
            res_0 = self.init_bn(x)
            res_1 = self.down_0(res_0)
            res_2 = self.down_1(res_1)
            res_3 = self.down_2(res_2)
            res_4 = self.down_3(res_3)
            res_5 = self.down_4(res_4)

            return [res_0, res_1, res_2, res_3, res_4, res_5]
        else:
            x = self.init_bn(x)
            x = self.down_0(x)
            x = self.down_1(x)
            x = self.down_2(x)
            x = self.down_3(x)
            x = self.down_4(x)

            return x


    def _make_down(self, i_th_block):
        return nn.Sequential(
            nn.AvgPool3d(2),
            nn.Conv3d(self.feature_list[i_th_block], self.feature_list[i_th_block+1], 1),
            ResBlock(self.feature_list[i_th_block+1], 3),
            nn.BatchNorm3d(self.feature_list[i_th_block+1]),
            nn.ReLU()
        )
        

class ResBlock(nn.Module):

    def __init__(self, num_in, num_blocks):
        super().__init__()
        self.relu = nn.ReLU()
        self.main_block = nn.ModuleList([self._creat_block(num_in) for _ in range(num_blocks)])

    def forward(self, x):
        for block in self.main_block:
            x = self.relu(x + block(x))
        return x


    def _creat_block(self, num_in):
        return nn.Sequential(
            nn.Conv3d(num_in, num_in, 1),
            nn.Conv3d(num_in, num_in, 3, 1, 1),
            nn.Conv3d(num_in, num_in, 1),
        )


class BatchAttentionModulle(nn.Module):


    def __init__(self, vector_size, num_head, mlp_dim=2, down_dim=None):
        super().__init__()
        self.num_head = num_head
        self.vector_size = vector_size
        head_dim = vector_size // num_head
        self.scale = head_dim**-0.5


        self.layernorm_attn = nn.LayerNorm([vector_size])
        self.layernorm_ffd = nn.LayerNorm([vector_size])
                
        self.qkv = nn.Linear(vector_size, 3*vector_size)
        self.mlp = nn.Sequential(
            nn.Linear(vector_size, mlp_dim*vector_size),
            nn.Linear(mlp_dim*vector_size, vector_size),
            nn.GELU()
        )
        self.tanh = nn.Tanh()
        self.down = down_dim
        if down_dim:
            self.down = nn.Linear(vector_size, down_dim)
        


    def forward(self, x):
        
        # input shape -> [B, C]
        batch_size, vector_size = x.shape
        assert vector_size == self.vector_size

        skip_0 = x
        qkv = self.qkv(x).reshape(batch_size, 3, self.num_head, vector_size // self.num_head).permute(1, 2, 0, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # q, k = self.tanh(q), self.tanh(k)
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(-1)
        x = (attn @ v).transpose(0, 1).flatten(1) + skip_0
        x = self.layernorm_attn(x)

        skip_1 = x
        x = self.mlp(x) + skip_1
        x = self.layernorm_ffd(x)

        if self.down:
            x = self.down(x)
        
        return x


        

        

        


if __name__ == "__main__":
    import torch

    device = "cuda:2"
    model = SelectionNet().to(device)
    # model = DownSample([1, 32, 128, 256, 512, 2048]).to(device)
    x = torch.rand(16, 2, 64, 64, 64).to(device)
    print(model(x))

    # model = BatchAttentionModulle(1024, 2, 2, 512).to(device)
    # x = torch.rand(4, 1024).to(device)

    # print(model(x).shape)

    


    

