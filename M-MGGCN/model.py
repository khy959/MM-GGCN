import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, vec_dim, hidden_dim, dropout_p):
        super(Model, self).__init__()


        self.second_dim = 6
        self.third_dim =6
        self.fourth_dim = 6
        # ----------------------


        self.gc1 = GraphConvolution(vec_dim, hidden_dim)
        self.res_proj1 = nn.Linear(vec_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, self.second_dim)
        # 残差投影 2: 将输入维度 (hidden_dim) 对齐到 second_dim
        self.res_proj2 = nn.Linear(hidden_dim, self.second_dim)
        self.bn2 = nn.BatchNorm1d(self.second_dim)
        self.gc3 = GraphConvolution(self.second_dim, self.third_dim)
        self.res_proj3 = nn.Linear(self.second_dim, self.third_dim)
        self.bn3 = nn.BatchNorm1d(self.third_dim)
        self.gc4 = GraphConvolution(self.third_dim, self.fourth_dim)

        self.res_proj4 = nn.Linear(self.third_dim, self.fourth_dim)
        self.bn4 = nn.BatchNorm1d(self.fourth_dim)


        self.linear = nn.Linear(vec_dim, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, features_vec, adj):
        x = features_vec  # [N, vec_dim]

        # gcn1_out = self.gc1(x, adj)
        # res1_out = self.res_proj1(x)
        # x = gcn1_out + res1_out
        # x = self.leaky_relu(x)
        # x = self.dropout(x)


        # gcn2_out = self.gc2(x, adj)
        # res2_out = self.res_proj2(x)
        # x = gcn2_out + res2_out
        # x = self.bn2(x)
        # x = self.leaky_relu(x)
        # x = self.dropout(x)


        # gcn3_out = self.gc3(x, adj)
        # res3_out = self.res_proj3(x)
        # x = gcn3_out + res3_out
        # x = self.bn3(x)
        # x = self.leaky_relu(x)
        #x = self.dropout(x)  # [N, third_dim]


        # gcn4_out = self.gc4(x, adj)
        # res4_out = self.res_proj4(x)
        # x = gcn4_out + res4_out
        # x = self.bn4(x)
        # x = self.leaky_relu(x)
        #x = self.dropout(x)  # [N, fourth_dim ]



        # --- Output: Readout ---
        x = self.linear(x)  # [N, 1]

        return torch.sigmoid(x)
