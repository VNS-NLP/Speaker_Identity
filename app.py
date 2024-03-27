import os
from torch import optim
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates 
from fastapi.staticfiles import StaticFiles
from cross_entropy_dataset import FBanksCrossEntropyDataset,DataLoader
import uvicorn
import torch
import numpy as np
from predictions import get_embeddings,get_cosine_distance
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
from cross_entropy_train import test,train
from cross_entropy_model import FBankCrossEntropyNet
from preprocessing import extract_fbanks
from pt_util import restore_objects, save_model, save_objects, restore_model
DATA_DIR ='data_dir/'

@app.post('/validation_model')
async def validation_model(test_dataset_path: str=Form(...),model_name:str=Form(...),use_cuda:bool=Form,batch_size:int=Form(...)):
    device = torch.device("cuda" if use_cuda else "cpu")
    import multiprocessing
    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}
    try:
        test_dataset = FBanksCrossEntropyDataset(test_dataset_path)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    except:
        return 'path dataset test is not exist'
        
    if model_name == 'fbanks net':
        model = FBankCrossEntropyNet(reduction='mean').to(device)
    else:
        model = None 
        return {"model not exist in lab"}
    test_loss,accurancy_mean = test(model,device,test_loader)
    print(accurancy_mean)
    return {'test loss': test_loss,
           'accurancy mean': float(accurancy_mean.numpy())}


@app.post('/train_model')
async def train_model(train_dataset_path:str=Form(...),test_dataset_path:str=Form(...),model_name:str=Form(...),epoch:int=Form(...),lr:float=Form(...),use_cuda:bool=Form(...),batch_size:int=Form(...)):
    device = torch.device("cuda" if use_cuda else "cpu")
    import multiprocessing
    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}
    try:
        train_dataset = FBanksCrossEntropyDataset(train_dataset_path)
        train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True,**kwargs)
        test_dataset = FBanksCrossEntropyDataset(test_dataset_path)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    except:
        return 'path dataset test or train is not exist'
    if model_name == 'fbanks net':
        model = FBankCrossEntropyNet(reduction='mean').to(device)
    else:
        model = None 
        return {"model not exist in lab"}
    model_path = 'saved_models_cross_entropy/'
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(model_path, (0, 0, [], [], [], []))
    start = last_epoch + 1 if max_accuracy > 0 else 0

    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(start,2):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, epoch, 500)
        test_loss, test_accuracy = test(model, device, test_loader)
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
    return {
        'message': 'model training successful',
        'train accurancies': str(train_accuracies),
        'test accuracies': str(test_accuracies),
        'train losses':str(train_losses),
        'test losses':str(test_losses),
        'max accurancy': str(max_accuracy)
    }



@app.post('/add_speaker')
async def add_speaker(file_speaker:UploadFile=File(...),speaker_name:str=Form(...)):
    dir_ = DATA_DIR + speaker_name
    if not os.path.exists(dir_):
        os.makedirs(dir_)
    content = await file_speaker.read()
    audioPath = DATA_DIR + speaker_name + '/sample.wav'
    with open(audioPath, 'wb') as f: 
            f.write(content)  
    fbanks = extract_fbanks(audioPath)
    embeddings = get_embeddings(fbanks)
    print('shape of embeddings: {}'.format(embeddings.shape), flush=True)
    mean_embeddings = np.mean(embeddings, axis=0)
    np.save(DATA_DIR+speaker_name+'/embeddings.npy',mean_embeddings)
    
    return {'message': 'audio of '+speaker_name+' saved'}

@app.get('/all_speaker')
async def show_all_speaker():
    list_user=os.listdir(DATA_DIR)
    return {"all_user":(str(list_user))}


@app.post('/inferences')
async def inferences(file_speaker:UploadFile=File(...),name_speaker:str=Form(...),THRESHOLD:float=Form(...)):
    dir_ = DATA_DIR + name_speaker
    if not os.path.exists(dir_):
        return {'message': 'name speaker is not exist,please add speaker'}
    content = await file_speaker.read()
    file_temp = 'static/temp.wav'
    with open(file_temp, 'wb') as f: 
        f.write(content)
    fbanks = extract_fbanks(file_temp)
    embeddings = get_embeddings(fbanks)
    stored_embeddings = np.load(DATA_DIR + name_speaker + '/embeddings.npy')
    stored_embeddings = stored_embeddings.reshape((1, -1))
    distances = get_cosine_distance(embeddings, stored_embeddings)
    print('mean distances', np.mean(distances), flush=True)
    positives = distances < THRESHOLD
    positives_mean = np.mean(positives)
    if positives_mean>=THRESHOLD:
        return {"result": 'with distances mean '+str(positives_mean)+' is speaker '+ name_speaker}
    else:
        return {"result": 'with distances mean '+str(positives_mean)+' is not speaker '+ name_speaker}
if __name__ == '__main__':
    uvicorn.run(app,port=3000)
    
    
    