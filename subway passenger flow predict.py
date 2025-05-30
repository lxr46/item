import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Parameter
import math


# ==================== TGC-LSTM 短时预测模型 ====================

class GraphConvolution(nn.Module):
    """图卷积层 - GCN的基本组成单元"""

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class TGC_LSTM_Cell(nn.Module):
    """时间-图卷积LSTM单元"""

    def __init__(self, input_size, hidden_size, adj_matrix):
        super(TGC_LSTM_Cell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.adj_matrix = adj_matrix

        # 图卷积层用于处理空间信息
        self.gc_input = GraphConvolution(input_size, hidden_size * 4)
        self.gc_hidden = GraphConvolution(hidden_size, hidden_size * 4)

        # 传统LSTM的时间门控
        self.linear_input = nn.Linear(input_size, hidden_size * 4)
        self.linear_hidden = nn.Linear(hidden_size, hidden_size * 4)

    def forward(self, input, hidden, cell):
        # 图卷积处理空间关系
        gi = self.gc_input(input, self.adj_matrix)
        gh = self.gc_hidden(hidden, self.adj_matrix)

        # 传统LSTM时间特征
        li = self.linear_input(input)
        lh = self.linear_hidden(hidden)

        # 融合时空特征
        combined = gi + gh + li + lh

        # LSTM门控机制
        i_gate, f_gate, o_gate, g_gate = combined.chunk(4, dim=1)

        i_gate = torch.sigmoid(i_gate)
        f_gate = torch.sigmoid(f_gate)
        o_gate = torch.sigmoid(o_gate)
        g_gate = torch.tanh(g_gate)

        # 更新细胞状态和隐藏状态
        new_cell = f_gate * cell + i_gate * g_gate
        new_hidden = o_gate * torch.tanh(new_cell)

        return new_hidden, new_cell


class TGC_LSTM(nn.Module):
    """TGC-LSTM短时人流量预测模型"""

    def __init__(self, input_size, hidden_size, num_layers, num_stations, adj_matrix, seq_len, pred_len):
        super(TGC_LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_stations = num_stations
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 邻接矩阵转换为稀疏张量
        self.adj_matrix = self._normalize_adj(adj_matrix)

        # 多层TGC-LSTM
        self.tgc_lstm_layers = nn.ModuleList([
            TGC_LSTM_Cell(input_size if i == 0 else hidden_size, hidden_size, self.adj_matrix)
            for i in range(num_layers)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_size, pred_len)
        self.dropout = nn.Dropout(0.2)

    def _normalize_adj(self, adj):
        """归一化邻接矩阵"""
        adj = adj + torch.eye(adj.size(0))  # 添加自环
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.
        degree_matrix = torch.diag(degree_inv_sqrt)
        return torch.mm(torch.mm(degree_matrix, adj), degree_matrix)

    def forward(self, x):
        batch_size = x.size(0)

        # 初始化隐藏状态和细胞状态
        hidden_states = [torch.zeros(batch_size * self.num_stations, self.hidden_size) for _ in range(self.num_layers)]
        cell_states = [torch.zeros(batch_size * self.num_stations, self.hidden_size) for _ in range(self.num_layers)]

        # 时间步循环
        for t in range(self.seq_len):
            input_t = x[:, t, :].contiguous().view(-1, self.input_size)

            for layer in range(self.num_layers):
                hidden_states[layer], cell_states[layer] = self.tgc_lstm_layers[layer](
                    input_t if layer == 0 else hidden_states[layer - 1],
                    hidden_states[layer],
                    cell_states[layer]
                )
                input_t = self.dropout(hidden_states[layer])

        # 输出预测
        output = self.output_layer(hidden_states[-1])
        output = output.view(batch_size, self.num_stations, self.pred_len)

        return output


# ==================== DCRNN 长期预测模型 ====================

class DiffusionConvolution(nn.Module):
    """扩散卷积层"""

    def __init__(self, supports, input_dim, output_dim, max_diffusion_step):
        super(DiffusionConvolution, self).__init__()
        self._supports = supports
        self._max_diffusion_step = max_diffusion_step
        self.weight = Parameter(torch.FloatTensor(input_dim * (max_diffusion_step + 1) * len(supports), output_dim))
        self.bias = Parameter(torch.FloatTensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.constant_(self.bias, 0)

    def forward(self, inputs):
        batch_size, num_nodes, input_dim = inputs.shape
        x = inputs
        x0 = x.permute(1, 2, 0).contiguous().view(num_nodes, -1)  # (num_nodes, input_dim * batch_size)

        diffusion_outputs = []

        for support in self._supports:
            x1 = x0
            diffusion_outputs.append(x1)

            for k in range(self._max_diffusion_step):
                x2 = torch.sparse.mm(support, x1)
                diffusion_outputs.append(x2)
                x1 = x2

        # 拼接所有扩散步骤的输出
        diffusion_output = torch.cat(diffusion_outputs, dim=1)
        diffusion_output = diffusion_output.view(num_nodes,
                                                 input_dim * (self._max_diffusion_step + 1) * len(self._supports),
                                                 batch_size)
        diffusion_output = diffusion_output.permute(2, 0, 1)  # (batch_size, num_nodes, features)

        # 线性变换
        output = torch.matmul(diffusion_output, self.weight) + self.bias
        return output


class DCRNNCell(nn.Module):
    """扩散卷积递归神经网络单元"""

    def __init__(self, input_dim, hidden_dim, adj_mx, max_diffusion_step=2):
        super(DCRNNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._max_diffusion_step = max_diffusion_step

        # 构建支持矩阵（前向和后向扩散）
        supports = []
        supports.append(self._calculate_random_walk_matrix(adj_mx).T)
        supports.append(self._calculate_random_walk_matrix(adj_mx.T).T)
        self._supports = supports

        # 扩散卷积层
        self._dc_update = DiffusionConvolution(self._supports, input_dim + hidden_dim, hidden_dim * 2,
                                               max_diffusion_step)
        self._dc_reset = DiffusionConvolution(self._supports, input_dim + hidden_dim, hidden_dim * 2,
                                              max_diffusion_step)
        self._dc_candidate = DiffusionConvolution(self._supports, input_dim + hidden_dim, hidden_dim,
                                                  max_diffusion_step)

    def _calculate_random_walk_matrix(self, adj_mx):
        """计算随机游走矩阵"""
        adj_mx = adj_mx + torch.eye(adj_mx.size(0))
        d = torch.sum(adj_mx, dim=1)
        d_inv = 1.0 / d
        d_inv[torch.isinf(d_inv)] = 0.
        d_mat_inv = torch.diag(d_inv)
        random_walk_mx = torch.mm(d_mat_inv, adj_mx)
        return random_walk_mx

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape[0], inputs.shape[1]

        # 拼接输入和隐藏状态
        concatenation = torch.cat([inputs, hidden_state], dim=2)

        # 更新门和重置门
        combined = self._dc_update(concatenation)
        u, r = torch.chunk(combined, chunks=2, dim=2)
        u = torch.sigmoid(u)
        r = torch.sigmoid(r)

        # 候选隐藏状态
        concatenation_reset = torch.cat([inputs, r * hidden_state], dim=2)
        hc = torch.tanh(self._dc_candidate(concatenation_reset))

        # 新隐藏状态
        new_hidden_state = u * hidden_state + (1 - u) * hc

        return new_hidden_state


class DCRNN(nn.Module):
    """扩散卷积递归神经网络 - 长期预测模型"""

    def __init__(self, input_dim, hidden_dim, num_layers, num_nodes, adj_mx, seq_len, horizon, max_diffusion_step=2):
        super(DCRNN, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_layers = num_layers
        self._num_nodes = num_nodes
        self._seq_len = seq_len
        self._horizon = horizon

        # 编码器层
        self.encoder_cells = nn.ModuleList([
            DCRNNCell(input_dim if i == 0 else hidden_dim, hidden_dim, adj_mx, max_diffusion_step)
            for i in range(num_layers)
        ])

        # 解码器层
        self.decoder_cells = nn.ModuleList([
            DCRNNCell(input_dim if i == 0 else hidden_dim, hidden_dim, adj_mx, max_diffusion_step)
            for i in range(num_layers)
        ])

        # 输出投影
        self.projection_layer = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(0.3)

    def encoder(self, inputs):
        """编码器：处理历史序列"""
        batch_size = inputs.size(0)
        encoder_hidden_states = [torch.zeros(batch_size, self._num_nodes, self._hidden_dim) for _ in
                                 range(self._num_layers)]

        for t in range(self._seq_len):
            for layer in range(self._num_layers):
                encoder_input = inputs[:, t, :, :] if layer == 0 else encoder_hidden_states[layer - 1]
                encoder_hidden_states[layer] = self.encoder_cells[layer](encoder_input, encoder_hidden_states[layer])
                encoder_hidden_states[layer] = self.dropout(encoder_hidden_states[layer])

        return encoder_hidden_states

    def decoder(self, encoder_hidden_states, labels=None):
        """解码器：生成未来预测"""
        batch_size = encoder_hidden_states[0].size(0)
        decoder_hidden_states = encoder_hidden_states
        decoder_input = torch.zeros(batch_size, self._num_nodes, self._input_dim)

        outputs = []

        for t in range(self._horizon):
            for layer in range(self._num_layers):
                decoder_input_layer = decoder_input if layer == 0 else decoder_hidden_states[layer - 1]
                decoder_hidden_states[layer] = self.decoder_cells[layer](decoder_input_layer,
                                                                         decoder_hidden_states[layer])
                decoder_hidden_states[layer] = self.dropout(decoder_hidden_states[layer])

            # 生成输出
            decoder_output = self.projection_layer(decoder_hidden_states[-1])
            outputs.append(decoder_output)

            # 教师强制 vs 自回归
            if self.training and labels is not None and t < labels.size(1):
                decoder_input = labels[:, t, :, :]
            else:
                decoder_input = decoder_output

        return torch.stack(outputs, dim=1)

    def forward(self, inputs, labels=None):
        """前向传播"""
        encoder_hidden_states = self.encoder(inputs)
        outputs = self.decoder(encoder_hidden_states, labels)
        return outputs


# ==================== 训练和使用示例 ====================

def create_sample_data(batch_size=32, seq_len=12, num_stations=50, feature_dim=1):
    """创建示例数据"""
    # 模拟地铁站人流量数据
    data = torch.randn(batch_size, seq_len, num_stations, feature_dim)

    # 创建随机邻接矩阵（实际应用中应基于地铁线路图）
    adj_matrix = torch.rand(num_stations, num_stations)
    adj_matrix = (adj_matrix + adj_matrix.T) / 2  # 对称化
    adj_matrix[adj_matrix < 0.3] = 0  # 稀疏化

    return data, adj_matrix


def train_models():
    """训练示例"""
    # 参数设置
    batch_size = 32
    seq_len = 12  # 输入序列长度
    num_stations = 50  # 地铁站数量
    feature_dim = 1  # 特征维度（人流量）
    hidden_dim = 64

    # 创建示例数据
    data, adj_matrix = create_sample_data(batch_size, seq_len, num_stations, feature_dim)

    # ===== TGC-LSTM 短时预测 =====
    print("训练 TGC-LSTM 短时预测模型...")
    tgc_lstm = TGC_LSTM(
        input_size=feature_dim,
        hidden_size=hidden_dim,
        num_layers=2,
        num_stations=num_stations,
        adj_matrix=adj_matrix,
        seq_len=seq_len,
        pred_len=6  # 预测未来6个时间步
    )

    optimizer_tgc = torch.optim.Adam(tgc_lstm.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 训练循环示例
    tgc_lstm.train()
    for epoch in range(10):
        optimizer_tgc.zero_grad()

        # 前向传播
        predictions = tgc_lstm(data)

        # 计算损失（这里用随机目标作为示例）
        targets = torch.randn(batch_size, num_stations, 6)
        loss = criterion(predictions, targets)

        # 反向传播
        loss.backward()
        optimizer_tgc.step()

        if epoch % 2 == 0:
            print(f"TGC-LSTM Epoch {epoch}, Loss: {loss.item():.4f}")

    # ===== DCRNN 长期预测 =====
    print("\n训练 DCRNN 长期预测模型...")
    dcrnn = DCRNN(
        input_dim=feature_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_nodes=num_stations,
        adj_mx=adj_matrix,
        seq_len=seq_len,
        horizon=24  # 预测未来24个时间步（天或月级别）
    )

    optimizer_dcrnn = torch.optim.Adam(dcrnn.parameters(), lr=0.001)

    # 训练循环示例
    dcrnn.train()
    for epoch in range(10):
        optimizer_dcrnn.zero_grad()

        # 前向传播
        future_targets = torch.randn(batch_size, 24, num_stations, feature_dim)
        predictions = dcrnn(data, future_targets)

        # 计算损失
        loss = criterion(predictions, future_targets)

        # 反向传播
        loss.backward()
        optimizer_dcrnn.step()

        if epoch % 2 == 0:
            print(f"DCRNN Epoch {epoch}, Loss: {loss.item():.4f}")

    print("\n模型训练完成！")
    return tgc_lstm, dcrnn, adj_matrix


def predict_flow(tgc_lstm, dcrnn, current_data, adj_matrix):
    """预测人流量"""
    tgc_lstm.eval()
    dcrnn.eval()

    with torch.no_grad():
        # 短时预测（小时级别）
        short_term_pred = tgc_lstm(current_data)
        print(f"短时预测结果形状: {short_term_pred.shape}")

        # 长期预测（天/月级别）
        long_term_pred = dcrnn(current_data)
        print(f"长期预测结果形状: {long_term_pred.shape}")

    return short_term_pred, long_term_pred


if __name__ == "__main__":
    # 训练模型
    tgc_lstm_model, dcrnn_model, adj_matrix = train_models()

    # 创建新的测试数据进行预测
    test_data, _ = create_sample_data(batch_size=1, seq_len=12, num_stations=50)

    # 进行预测
    short_pred, long_pred = predict_flow(tgc_lstm_model, dcrnn_model, test_data, adj_matrix)

    print(f"\n预测完成！")
    print(f"短时预测: {short_pred.shape} - 预测未来6个时间步的人流量")
    print(f"长期预测: {long_pred.shape} - 预测未来24个时间步的人流量")