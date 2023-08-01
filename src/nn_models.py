import torch

class SimpleNet(torch.nn.Module):
    '''
    Simple Convolutional neural network architecture 
    '''
    
    def __init__(self, **kwargs):
        super(SimpleNet, self).__init__()
        
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=20, kernel_size=7, stride= 2, padding=3, bias=False)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(.25)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3, padding=0, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=20, out_channels=40, kernel_size=5, stride=2, padding=2, bias=False)
        self.conv3 = torch.nn.Conv2d(in_channels=40, out_channels=30, kernel_size=3, stride=1, padding=1, bias=False)
        self.lin = torch.nn.Linear(in_features=1470, out_features=100,)
        
    def forward(self, x):
        c1 = self.relu(self.conv1(x))
        c1 = self.pool1(c1)
        c2 = self.relu(self.conv2(c1))
        c2 = self.pool1(c2)
        c3 = self.relu(self.conv3(c2))
        c3 = self.pool1(c3)
        c3 = c3.view(-1, 1470)
        d1 = self.sigmoid(self.lin(c3))
        d1 = d1.view(-1, 20, 5)
        return d1
    
    def load_weights(self, load_path):
        'Load model weights from file at load_path'
        self.load_state_dict(torch.load(load_path))
        
    def save_weights(self, save_path):
        'Save model weights to save_path'
        torch.save(self.state_dict(), save_path)