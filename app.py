import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from torch import optim
from fastapi import FastAPI, File, UploadFile, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates 
from fastapi.staticfiles import StaticFiles
from cross_entropy_dataset import FBanksCrossEntropyDataset,DataLoader
import uvicorn
import torch
import numpy as np
import faiss
from predictions import get_embeddings,get_cosine_distance
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
from cross_entropy_train import test,train
from cross_entropy_model import FBankCrossEntropyNet
from preprocessing import extract_fbanks
from pt_util import restore_objects, save_model, save_objects, restore_model
from fbankcross_classification import LinearClassifier,train_classification,test_classification,inference_speaker_classification
DATA_DIR ='data_dir/'




# api identifi speaker 

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

    models_path = []
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
            model_path = save_model(model, epoch, model_path)
            models_path.append(model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies), epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))
    return {
        'message': 'model training successful',
        'train accurancies': str(train_accuracies),
        'test accuracies': str(test_accuracies),
        'train losses':str(train_losses),
        'test losses':str(test_losses),
        'max accurancy': str(max_accuracy),
        'models path': models_path
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



# api classification speaker 
@app.post('speaker_classification_train/')
async def train_model_classification(train_dataset_path:str=Form(...),test_dataset_path:str=Form(...),model_name:str=Form(...),epochs:int=Form(...),lr:float=Form(...),use_cuda:bool=Form(...),batch_size:int=Form(...)):
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
    
    try:
        train_dataset.num_classes == test_dataset.num_classes
    except:
        return "The number of speakers in test and training sets must be equal "
    if model_name == 'fbanks net classification':
        try:
            model = LinearClassifier(output_size=test_dataset.num_classes).to(device)
        except:
            print('cuda load is error')
            use_cuda = False
            device = torch.device("cuda" if use_cuda else "cpu")
            model = LinearClassifier(output_size=test_dataset.num_classes).to(device)
    else:
        model = None 
        return {"model not exist in lab"}
    model_path = 'saved_models_cross_entropy_classification/'
    model = restore_model(model, model_path)
    last_epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies = restore_objects(model_path, (0, 0, [], [], [], []))
    start = last_epoch + 1 if max_accuracy > 0 else 0

    models_path = []
    optimizer = optim.Adam(model.parameters(), lr)
    for epoch in range(start,epochs):
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
            model_path = save_model(model, epoch, model_path)
            models_path.append(model_path)
            save_objects((epoch, max_accuracy, train_losses, test_losses, train_accuracies, test_accuracies), epoch, model_path)
            print('saved epoch: {} as checkpoint'.format(epoch))
    return {
        'message': 'model training successful',
        'train accurancies': str(train_accuracies),
        'test accuracies': str(test_accuracies),
        'train losses':str(train_losses),
        'test losses':str(test_losses),
        'max accurancy': str(max_accuracy),
        'models path': models_path
    }



@app.post('/speaker_classification_validation')
async def validation_model_classification(test_dataset_path: str=Form(...),model_name:str=Form(...),model_path:str=Form(...),use_cuda:bool=Form(...),batch_size:int=Form(...)):
    device = torch.device("cuda" if use_cuda else "cpu")
    import multiprocessing
    kwargs = {'num_workers': multiprocessing.cpu_count(),
              'pin_memory': True} if use_cuda else {}
    try:
        test_dataset = FBanksCrossEntropyDataset(test_dataset_path)
        test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=True,**kwargs)
    except:
        return 'path dataset test is not exist'
        
    if model_name == 'fbanks net classification':
        try:
            model = LinearClassifier(output_size=test_dataset.num_classes)
            cpkt = torch.load(model_path)
            model.load_state_dict(cpkt)
            model.to(device)
        except:
            print('cuda load is error')
            use_cuda = False
            device = torch.device("cuda" if use_cuda else "cpu")
            model = LinearClassifier(output_size=test_dataset.num_classes)
            cpkt = torch.load(model_path)
            model.load_state_dict(cpkt)
            model.to(device)
    else:
        model = None 
        return {"model not exist in lab"}
    test_loss,accurancy_mean = test_classification(model,device,test_loader)
    print(accurancy_mean)
    return {'test loss': test_loss,
           'accurancy mean': float(accurancy_mean.numpy())}

@app.post('/speaker_classification_inference')
async def inferences(file_speaker:UploadFile=File(...),use_cuda:bool=Form(...),model_path:str=Form(...)):
    content = await file_speaker.read()
    file_temp = 'static/temp.wav'
    with open(file_temp, 'wb') as f: 
        f.write(content)
    try:
        rs = inference_speaker_classification(file_speaker=file_temp,use_cuda=use_cuda,model_path=model_path)
        return rs
    except:
        print('cuda error')
        rs = inference_speaker_classification(file_speaker=file_temp,use_cuda=False)
        return rs 
    
# search speaker 
def load_data_speaker(data_dir = 'data_dir'):
    if os.path.exists(data_dir):
        data_dict = {}
        for dir_name in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, dir_name)
            if os.path.isdir(dir_path):
                sub_data = {}
                for file_name in os.listdir(dir_path):
                    if file_name.endswith('.npy'):
                        file_path = os.path.join(dir_path, file_name)
                        key = file_name.replace('.npy', '')  # Sử dụng tên file làm key
                        value = np.load(file_path)  # Load file .npy
                        sub_data[key] = value
                
                data_dict[dir_name] = sub_data
                
        return data_dict
    else:
        return "folder do not exist"
           

@app.post('/inferences_search_speaker')
async def inferences_searh(file_speaker:UploadFile=File(...),k:int=Form(...)):
    content = await file_speaker.read()
    file_temp = 'static/temp.wav'
    with open(file_temp, 'wb') as f: 
        f.write(content)
    fbanks = extract_fbanks(file_temp)
    embeddings = get_embeddings(fbanks)
    mean_embeddings = np.mean(embeddings, axis=0)
    mean_embeddings=mean_embeddings.reshape((1, -1))
    rs=load_data_speaker('data_dir')
    encodes = []
    person_ids = []
    for key,vectors in rs.items():
        for emb,vector in vectors.items():
            encodes.append(np.array(vector,dtype=np.float32))
            person_ids.append(key)
    encodes = np.vstack(encodes).astype(np.float32)
    index = faiss.IndexFlatL2(encodes.shape[1])
    index.add(encodes)
    distances, indices = index.search(mean_embeddings, k)
    
    rs_speaker = []
    for i in range(k):
        rs_speaker.append(f"speaker {i+1}: {person_ids[indices[0][i]]}, distances: {distances[0][i]}")
    return {'result':rs_speaker}



if __name__ == '__main__':
    uvicorn.run(app,port=3000)
    # rs=load_data_speaker('data_dir')
    # encodes = []
    # person_ids = []
    # for key,vectors in rs.items():
    #     for emb,vector in vectors.items():
    #         encodes.append(np.array(vector,dtype=np.float32))
    #         person_ids.append(key)
    
    # encodes = np.vstack(encodes).astype(np.float32)
    # index = faiss.IndexFlatL2(encodes.shape[1])
    # index.add(encodes)
    # print(encodes.shape[1])
    # query_vector = np.random.rand(1, encodes.shape[1]).astype(np.float32)  # Vector truy vấn ngẫu nhiên
    # k = 5  # Số lượng kết quả gần nhất cần tìm
    # print(k)
    # distances, indices = index.search(query_vector, k)
    # for i in range(k):
    #     print(f"Vector gần nhất {i+1}: {person_ids[indices[0][i]]}, khoảng cách: {distances[0][i]}")
    