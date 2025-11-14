import torch
import torch.nn as nn

class Action_Conditioned_FF(nn.Module):
    def __init__(self, input_size=6, hidden1_size=32, hidden2_size=16, output_size=1):
        super(Action_Conditioned_FF,self).__init__()
        self.layer1 = nn.Linear(input_size,hidden1_size)
        self.bn1 = nn.BatchNorm1d(hidden1_size)
        self.drop1 = nn.Dropout(0.2)

        self.layer2 = nn.Linear(hidden1_size,hidden2_size)
        self.bn2 = nn.BatchNorm1d(hidden2_size)
        self.drop2 = nn.Dropout(0.2)

        self.output_layer = nn.Linear(hidden2_size,output_size)
        self.relu_activation = nn.ReLU()
        self.sigmoid_activation = nn.Sigmoid()
# STUDENTS: __init__() must initiatize nn.Module and define your network's
# custom architecture

    def forward(self, input):
# STUDENTS: forward() must complete a single forward pass through your network
# and return the output which should be a tensor
        x = self.layer1(input)
        x = self.bn1(x)
        output1 = self.relu_activation(x)
        output1 = self.drop1(output1)


        x = self.layer2(output1)
        x = self.bn2(x)
        output2 = self.relu_activation(x)
        output2 = self.drop2(output2)

        output = self.output_layer(output2)
        output = self.sigmoid_activation(output)

        return output


    def evaluate(self, model, test_loader, loss_function):
        # test_input, labels = test_loader['input'], test_loader['labels']
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for idx, test_sample in enumerate(test_loader):
                test_input, test_label = test_sample['input'], test_sample['label']
                test_input = torch.tensor(test_input, dtype=torch.float32)
                test_label = torch.tensor(test_label, dtype=torch.float32)
                output = self.forward(test_input)
                output = output.squeeze(1)
                loss = loss_function(output,test_label)
                total_loss += loss.item()
        avg_loss = total_loss / len(test_loader)
# STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
# mind that we do not need to keep track of any gradients while evaluating the
# model. loss_function will be a PyTorch loss function which takes as argument the model's
# output and the desired output.
        return avg_loss

def main():
    model = Action_Conditioned_FF()

if __name__ == '__main__':
    main()
