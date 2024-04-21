import torch
import torch.nn.functional as F
import torch.nn as nn
from cross_entropy_model import FBankCrossEntropyNet
from cross_entropy_dataset import FBanksCrossEntropyDataset
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
import tqdm
import multiprocessing
import time
import numpy as np
from pt_util import restore_objects, save_model, save_objects, restore_model

MODEL_PATH = 'weights/triplet_loss_trained_model.pth'
model_instance = FBankCrossEntropyNet()
model_instance.load_state_dict(torch.load(MODEL_PATH, map_location=lambda storage, loc: storage))
# model_instance = model_instance.double()

use_cuda = False
kwargs = {'num_workers': multiprocessing.cpu_count(),
            'pin_memory': True} if use_cuda else {}



class LinearClassifier(nn.Module):
    def __init__(self, output_size,input_size=250):
        super(LinearClassifier, self).__init__()
        self.linear1 = nn.Linear(input_size, 1)
        self.linear2 = nn.Linear(1,output_size)
        self.loss_layer = nn.CrossEntropyLoss(reduction='mean')
    
    def forward(self, x):
        input = self.linear1(x)
        return self.linear2(input)
    
    def loss(self, predictions, labels):
        loss_val = self.loss_layer(predictions, labels)
        return loss_val



def train_classification(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    losses = []
    accuracy = 0
    for batch_idx, (x, y) in enumerate(tqdm.tqdm(train_loader)):
        x, y = x.to(device), y.to(device)
        x = model_instance(x)
        optimizer.zero_grad()
        out = model(x)
        print(x.shape)
        loss = model.loss(out, y)

        with torch.no_grad():
            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == y))

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('{} Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                time.ctime(time.time()),
                epoch, batch_idx * len(x), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    accuracy_mean = (100. * accuracy) / len(train_loader.dataset)

    return np.mean(losses), accuracy_mean




def test_classification(model, device, test_loader, log_interval=None):
    model.eval()
    losses = []

    accuracy = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
            x, y = x.to(device), y.to(device)
            x = model_instance(x)
            out = model(x)
            test_loss_on = model.loss(out, y).item()
            losses.append(test_loss_on)

            pred = torch.argmax(out, dim=1)
            accuracy += torch.sum((pred == y))

            if log_interval is not None and batch_idx % log_interval == 0:
                print('{} Test: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    time.ctime(time.time()),
                    batch_idx * len(x), len(test_loader.dataset),
                    100. * batch_idx / len(test_loader), test_loss_on))

    test_loss = np.mean(losses)
    accuracy_mean = (100. * accuracy) / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} , ({:.4f})%\n'.format(
        test_loss, accuracy, len(test_loader.dataset), accuracy_mean))
    return test_loss, accuracy_mean



def speaker_probability(tensor):
    counts = {}
    total = 0
    for value in tensor:
        value = int(value)
        counts[value] = counts.get(value, 0) + 1
        total += 1

    probabilities = {}
    for key, value in counts.items():
        probabilities['speaker '+str(key)] = value / total

    return probabilities



def inference_speaker_classification(file_speaker,num_class=6,model_instance= model_instance,use_cuda=False,model_path='saved_models_cross_entropy_classification/0.pth'):
    device = torch.device("cuda" if use_cuda else "cpu")
    from preprocessing import extract_fbanks
    fbanks = extract_fbanks(file_speaker)
    model = LinearClassifier(output_size=num_class)
    cpkt = torch.load(model_path)
    model.load_state_dict(cpkt)
    model = model.double()
    model.to(device)
    model_instance = model_instance.double()
    model_instance.eval()
    model_instance.to(device)
    with torch.no_grad():
        x = torch.from_numpy(fbanks)
        embedings = model_instance(x.to(device))
        # print(embedings.shape)  
        # embedings=embedings.unsqueeze(0)
        output = model(embedings)
        output = torch.argmax(output,dim=-1) 
        speaker_pro = speaker_probability(output)
        print(speaker_pro)
    return speaker_pro

def main_train():
    model_path = 'saved_models_cross_entropy_classification/'
    use_cuda = False
    device = torch.device("cuda" if use_cuda else "cpu")
    print('using device', device)

    import multiprocessing
    print('num cpus:', multiprocessing.cpu_count())

    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {} 

    train_dataset = FBanksCrossEntropyDataset('fbanks-test')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)

    test_dataset = FBanksCrossEntropyDataset('fbanks-test')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, **kwargs)

    model = LinearClassifier(output_size=test_dataset.num_classes).to(device)
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(model_path, (0, 0, [], [], [], []))
    start = last_epoch + 1 if max_accuracy > 0 else 0

    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    for epoch in range(start, 1):
        train_loss, train_accuracy = train_classification(model, device, train_loader, optimizer, epoch, 500)
        test_loss, test_accuracy = test_classification(model, device, test_loader)
        print('After epoch: {}, train_loss: {}, test loss is: {}, train_accuracy: {}, '
              'test_accuracy: {}'.format(epoch, train_loss, test_loss, train_accuracy, test_accuracy))

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        if test_accuracy > max_accuracy:
            max_accuracy = test_accuracy
            save_model(model, epoch, model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies), epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))



  

    
if __name__ == '__main__':
    device = torch.device("cuda" if use_cuda else "cpu")
    # data_test = FBanksCrossEntropyDataset('fbanks-test')
    # test_loader = DataLoader(data_test, batch_size=1, shuffle=True, **kwargs)
    # numclass=data_test.num_classes
    # #  print(numclass)
    # #  one_hot = F.one_hot(torch.LongTensor([1]),num_classes=numclass)
    # #  print(one_hot)
    # model = LinearClassifier(output_size=numclass)
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.01)
    # num_epochs = 10 
    # for epoch in range(num_epochs):
    #     model.train()
    #     for batch_idx, (x, y) in enumerate(tqdm.tqdm(test_loader)):
    #         optimizer.zero_grad()
    #         x, y = x.to(device), y.to(device)
    #         input = model_instance(x)
    #         labels  = F.one_hot(y,num_classes=numclass)
    #         outputs = model(input)
    #         print(outputs)
    #         print(labels)
    #         loss = criterion(outputs,y)
    #     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    # main_train() 
    inference_speaker_classification('sample-0.wav',device,6)
             

