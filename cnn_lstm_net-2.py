import torch
from torch.autograd import Variable
import torch.nn as nn

class CNNLSTMNetwork(nn.Module):
    def __init__(self, cnn_network, lstm_hidden_units, lstm_layers, lstm_dropout):
        super(CNNLSTMNetwork, self).__init__()

        self.cnn = cnn_network
        self.lstm_hidden_units = lstm_hidden_units
        self.lstm = nn.LSTM(input_size=lstm_hidden_units,
                            hidden_size=lstm_hidden_units,
                            num_layers=lstm_layers,
                            dropout=lstm_dropout,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden_units, 2)

    def forward(self, x, hx_cx=None):
        batches, seq_len, chans, _ = x.shape
        outputs = []
        cnn_out = self.cnn(x[:, 0, 0, :].unsqueeze(1)).squeeze()
        lstm_out, _ = self.lstm(cnn_out.unsqueeze(1), hx_cx)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        outputs.append(output)
        
        intermediate_outputs = []  # List to store intermediate tensors
        for i in range(1, seq_len):
            cnn_out = self.cnn(x[:, i, 0, :].unsqueeze(1)).squeeze()
            lstm_out, _ = self.lstm(cnn_out.unsqueeze(1))
            lstm_out = lstm_out[:, -1, :]
            output = self.fc(lstm_out)
            intermediate_outputs.append(output)

        outputs = torch.cat([outputs[0]] + intermediate_outputs, dim=0)

        return output
