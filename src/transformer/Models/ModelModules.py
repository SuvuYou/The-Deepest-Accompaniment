import torch

class ChordsLSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, cnn_feature_size):
        super(ChordsLSTM, self).__init__()

        # Define LSTM layer
        self.lstm = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Define CNN layer
        self.cnn = torch.nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2))
        self.relu_cnn = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Define second CNN layer
        self.cnn2 = torch.nn.Conv3d(64, 254, kernel_size=(1, 4, 4), stride=(1, 1, 1))
        self.relu_cnn2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        
        # Define fully connected layers
        self.fc_cnn_in = torch.nn.Linear(48260, cnn_feature_size)
        self.fc_lstm_in = torch.nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc_mid = torch.nn.Linear(hidden_size * 4 + cnn_feature_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, output_size)
        
        # Define dropout layers
        self.dropout = torch.nn.Dropout(p=0.3)
        
        # Define activation functions
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        
    def forward(self, chords, video):   
        # LSTM forward pass
        lstm_out, _ = self.lstm(chords)
        lstm_out = self.relu(lstm_out)
        
        lstm_output = self.fc_lstm_in(lstm_out[:, -1, :])
        lstm_output = self.relu(lstm_output)
        lstm_output = self.dropout(lstm_output)
        
        # CNN forward pass
        video = video.permute(0, 4, 1, 2, 3)
        cnn_out = self.cnn(video)
        cnn_out = self.relu_cnn(cnn_out)
        cnn_out = self.pool(cnn_out)
        
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.relu_cnn2(cnn_out)
        cnn_out = self.pool2(cnn_out)

        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        cnn_output = self.fc_cnn_in(cnn_out)
        cnn_output = self.relu(cnn_output)
        cnn_output = self.dropout(cnn_output)
        
        # Fully connected layers
        combined_output = torch.cat((cnn_output, lstm_output), dim=1)
        output = self.fc_mid(combined_output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_out(output)
        output = self.dropout(output)
        
        # Apply softmax activation
        output = self.softmax(output)

        return output
          
class MelodyLSTM(torch.nn.Module):
    def __init__(self, melody_input_size, chords_context_input_size, hidden_size, num_layers, output_size, cnn_feature_size, chords_feature_size):
        super(MelodyLSTM, self).__init__()

        # Define LSTM layer
        self.lstm = torch.nn.GRU(melody_input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Define CNN layer
        self.cnn = torch.nn.Conv3d(3, 64, kernel_size=(2, 3, 3), stride=(1, 2, 2))
        self.relu_cnn = torch.nn.ReLU()
        self.pool = torch.nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        
        # Define second CNN layer
        self.cnn2 = torch.nn.Conv3d(64, 254, kernel_size=(1, 4, 4), stride=(1, 1, 1))
        self.relu_cnn2 = torch.nn.ReLU()
        self.pool2 = torch.nn.AvgPool3d(kernel_size=(1, 4, 4), stride=(1, 4, 4))
        
        # Define fully connected layers
        self.fc_cnn_in = torch.nn.Linear(48260, cnn_feature_size)
        self.fc_chords_in = torch.nn.Linear(chords_context_input_size, chords_feature_size)
        self.fc_lstm_in = torch.nn.Linear(hidden_size * 2, hidden_size * 4)
        self.fc_mid = torch.nn.Linear(hidden_size * 4 + cnn_feature_size + chords_feature_size, hidden_size)
        self.fc_out = torch.nn.Linear(hidden_size, output_size)
        
        # Define dropout layers
        self.dropout = torch.nn.Dropout(p=0.3)
        
        # Define activation functions
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax()
        
    def forward(self, melody, chords_context, video):   
        # LSTM forward pass
        lstm_out, _ = self.lstm(melody)
        lstm_out = self.relu(lstm_out)
        
        lstm_output = self.fc_lstm_in(lstm_out[:, -1, :])
        lstm_output = self.relu(lstm_output)
        lstm_output = self.dropout(lstm_output)
        
        # CNN forward pass
        video = video.permute(0, 4, 1, 2, 3)
        cnn_out = self.cnn(video)
        cnn_out = self.relu_cnn(cnn_out)
        cnn_out = self.pool(cnn_out)
        
        cnn_out = self.cnn2(cnn_out)
        cnn_out = self.relu_cnn2(cnn_out)
        cnn_out = self.pool2(cnn_out)

        cnn_out = cnn_out.reshape(cnn_out.size(0), -1)
        cnn_output = self.fc_cnn_in(cnn_out)
        cnn_output = self.relu(cnn_output)
        cnn_output = self.dropout(cnn_output)
        
        # Chords forward pass
        chords_output = self.fc_chords_in(chords_context)
        chords_output = self.relu(chords_output)
        chords_output = self.dropout(chords_output)
        
        # Fully connected layers
        combined_output = torch.cat((cnn_output, lstm_output, chords_output), dim=1)
        output = self.fc_mid(combined_output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.fc_out(output)
        output = self.dropout(output)
        
        # Apply softmax activation
        output = self.softmax(output)

        return output