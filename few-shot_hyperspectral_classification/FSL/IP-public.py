import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os
import math
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils
import models
import random
from FSL.query_attention import Transformer, Attention
import sys

def seed_torch(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 128)
parser.add_argument("-d","--tar_input_dim",type = int, default = 200)
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 16)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)

parser.add_argument("-m","--test_class_num",type=int, default=16)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

parser.add_argument("-r","--random_seed",type=int, default=10)
parser.add_argument("-i","--contrastive_weight",type=float, default=10)
parser.add_argument("-j","--fsl_weight",type=float, default=1)
parser.add_argument("-k","--domain_weight",type=float, default=1)
parser.add_argument("-x","--attention_heads",type=int, default=8)
parser.add_argument("-y","--attention_depth",type=int, default=2)
parser.add_argument("-a","--contrastive_temperature",type=float, default=0.5)
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit


TEST_CLASS_NUM = args.test_class_num 
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class 

utils.same_seeds(0)
def _init_():
    if not os.path.exists('../checkpoints'):
        os.makedirs('../checkpoints')
    if not os.path.exists('../classificationMap'):
        os.makedirs('../classificationMap')
_init_()


with open(os.path.join('../datasets', 'Chikusei_imdb_128.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])


data_train = source_imdb['data'] 
labels_train = source_imdb['Labels'] 
print(data_train.shape)
print(labels_train.shape)
keys_all_train = sorted(list(set(labels_train)))  
print(keys_all_train) 
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) 
print("Num classes of the number of class larger than 200: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  
        data[class_][i] = image_transpose


metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data


print(source_imdb['data'].shape) 
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) 
print(source_imdb['data'].shape) 
print(source_imdb['Labels'])
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb



test_data = '../datasets/IP/indian_pines_corrected.mat'
test_label = '../datasets/IP/indian_pines_gt.mat'

Data_Band_Scaler, GroundTruth = utils.load_data(test_data, test_label)


def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape) 
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = utils.flip(Data_Band_Scaler)
    groundtruth = utils.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)  
    
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    
    train = {}
    test = {}
    da_train = {} 
    m = int(np.max(G))  
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices:', len(train_indices))
    print('the number of test_indices:', len(test_indices))
    print('the number of train_indices after data argumentation:', len(da_train_indices))  
    print('labeled sample indices:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)  
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')

    train_dataset = utils.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=class_num * shot_num_per_class,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = utils.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)  
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):  
        imdb_da_train['data'][:, :, :, iSample] = utils.radiation_noise(
            data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
            Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)

    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1  
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain


def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)  
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape) 

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)  
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))  
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']  
    print('target data augmentation label:', target_da_labels)

    
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    
    print(imdb_da_train['data'].shape)  
    print(imdb_da_train['Labels'])
    target_dataset = utils.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=128, shuffle=True, num_workers=0)
    del target_dataset

    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain



def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        
    )
    return layer

class instance_contrastive_Loss(nn.Module):
    def __init__(self):
        super(instance_contrastive_Loss, self).__init__()

    def forward(self, out_1, label, temperature=0.5):
        
        out_1 = F.normalize(out_1, dim=-1)
        
        sim_matrix = torch.exp(torch.mm(out_1, out_1.t().contiguous()) / temperature)
        
        mask = (torch.ones_like(sim_matrix) - torch.eye(out_1.shape[0], device=device)).bool()
        
        negative_sim = sim_matrix.masked_select(mask).view(out_1.shape[0], -1)
        negative_sim = negative_sim.sum(dim=-1)

        
        mask = label.repeat(out_1.shape[0], 1)
        mask2 = mask.transpose(1, 0)
        mask = mask2 - mask
        mask[mask!=0] = -1
        mask[mask == 0] = 1
        mask[mask<0] = 0
        mask = (mask.to(device) - torch.eye(out_1.shape[0], device=device)).bool()
        pos_sim = sim_matrix.masked_select(mask).view(out_1.shape[0], -1).mean(dim=-1)

        return (- torch.log(pos_sim / negative_sim)).mean()



class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x): 
        x1 = F.relu(self.conv1(x), inplace=True) 
        x2 = F.relu(self.conv2(x1), inplace=True) 
        x3 = self.conv3(x2) 

        out = F.relu(x1+x3, inplace=True) 
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160
        

    def forward(self, x): 
        x = x.unsqueeze(1) 
        x = self.block1(x) 
        x = self.maxpool1(x) 
        x = self.block2(x) 
        x = self.maxpool2(x) 
        x = self.conv(x) #(16, 16, 7, 3, 3) #(304, 16, 7, 3, 3) #(128, 16, 7, 3, 3)
        x = x.view(x.shape[0],-1) 
        
        return x


class Mapping(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, out_dimension, 1, 1, bias=False)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn(x)
        return x

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.final_feat_dim = FEATURE_DIM  
        
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping(SRC_INPUT_DIMENSION, N_DIMENSION)

    def forward(self, x, domain='source'):  
        
        if domain == 'target':
            x = self.target_mapping(x)  
        elif domain == 'source':
            x = self.source_mapping(x)  

        feature = self.feature_encoder(x)  

        output = self.classifier(feature)
        return feature, output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:

        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits



nDataSet = args.random_seed
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None

crossEntropy = nn.CrossEntropyLoss().to(device)
domain_criterion = nn.BCEWithLogitsLoss().to(device)
contrastiveLoss = instance_contrastive_Loss().to(device)

seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]

weight_constrativeloss = args.contrastive_weight
weight_fsl = args.fsl_weight
weight_domain = args.domain_weight

for iDataSet in range(nDataSet):
    seed_torch(seeds[iDataSet])
    # np.random.seed(seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    feature_encoder = Network()
    domain_classifier = models.DomainClassifier()
    random_layer = models.RandomLayer([args.feature_dim, args.class_num], 1024)
    transformer_q = Transformer(dim=FEATURE_DIM, depth=args.attention_depth, heads=args.attention_heads, dim_head=64, mlp_dim=512, dropout=0.1)

    feature_encoder.apply(weights_init)
    domain_classifier.apply(weights_init)
    transformer_q.apply(weights_init)

    feature_encoder.to(device)
    domain_classifier.to(device)
    random_layer.to(device)
    transformer_q.to(device)

    feature_encoder.train()
    domain_classifier.train()
    transformer_q.train()

    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    domain_classifier_optim = torch.optim.Adam(domain_classifier.parameters(), lr=args.learning_rate)
    transformer_q_optim = torch.optim.Adam(transformer_q.parameters(), lr=args.learning_rate)

    print("Training...")
    print("The seed tested: ", seeds[iDataSet] )

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()
    for episode in range(EPISODE):

        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()


        if episode % 2 == 0:
            '''Few-shot claification for source domain data set'''

            task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)


            supports, support_labels = support_dataloader.__iter__().next()
            querys, query_labels = query_dataloader.__iter__().next()


            support_features, support_outputs = feature_encoder(supports.to(device))
            query_features, query_outputs = feature_encoder(querys.to(device))
            target_features, target_outputs = feature_encoder(target_data.to(device), domain='target')


            features_cats = torch.cat((support_features, query_features))
            label_cats = torch.cat((support_labels, query_labels))
            contrastive_Loss = contrastiveLoss(features_cats, label_cats, args.contrastive_temperature)


            query_features_new = transformer_q(query_features, support_features)
            support_features_new = support_features.clone()

            max_value = support_features_new.max()
            min_value = support_features_new.min()
            support_features_new = (support_features_new - min_value) * 1.0 / (max_value - min_value)
            query_features_new = (query_features_new - min_value) * 1.0 / (max_value - min_value)

            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features_new.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features_new


            logits = euclidean_metric(query_features_new, support_proto)
            query_labels_copy = torch.LongTensor(query_labels.numpy())
            f_loss = crossEntropy(logits, query_labels_copy.to(device))


            features = torch.cat([support_features, query_features, target_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, target_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)


            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + target_data.shape[0], 1]).to(device)
            domain_label[:supports.shape[0] + querys.shape[0]] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)


            f_loss = weight_fsl * f_loss
            domain_loss = weight_domain * domain_loss
            contrastive_Loss = weight_constrativeloss * contrastive_Loss
            loss = f_loss + domain_loss + contrastive_Loss


            feature_encoder.zero_grad()
            transformer_q.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            transformer_q_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        else:
            '''Few-shot classification for target domain data set'''

            task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)
            support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
            query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)


            supports, support_labels = support_dataloader.__iter__().next()
            querys, query_labels = query_dataloader.__iter__().next()


            support_features, support_outputs = feature_encoder(supports.to(device),  domain='target')
            query_features, query_outputs = feature_encoder(querys.to(device), domain='target')
            source_features, source_outputs = feature_encoder(source_data.to(device))

            features_cats = torch.cat((support_features, query_features))
            label_cats = torch.cat((support_labels, query_labels))
            contrastive_Loss = contrastiveLoss(features_cats, label_cats, args.contrastive_temperature)

            query_features_new = transformer_q(query_features, support_features)
            support_features_new = support_features.clone()

            max_value = support_features_new.max()
            min_value = support_features_new.min()
            support_features_new = (support_features_new - min_value) * 1.0 / (max_value - min_value)
            query_features_new = (query_features_new - min_value) * 1.0 / (max_value - min_value)


            if SHOT_NUM_PER_CLASS > 1:
                support_proto = support_features_new.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            else:
                support_proto = support_features_new


            logits = euclidean_metric(query_features_new, support_proto)
            query_labels_copy = torch.LongTensor(query_labels.numpy())
            f_loss = crossEntropy(logits, query_labels_copy.to(device))

            features = torch.cat([support_features, query_features, source_features], dim=0)
            outputs = torch.cat((support_outputs, query_outputs, source_outputs), dim=0)
            softmax_output = nn.Softmax(dim=1)(outputs)

            domain_label = torch.zeros([supports.shape[0] + querys.shape[0] + source_features.shape[0], 1]).to(device)
            domain_label[supports.shape[0] + querys.shape[0]:] = 1

            randomlayer_out = random_layer.forward([features, softmax_output])

            domain_logits = domain_classifier(randomlayer_out, episode)
            domain_loss = domain_criterion(domain_logits, domain_label)

            f_loss = weight_fsl * f_loss
            domain_loss = weight_domain * domain_loss
            contrastive_Loss = weight_constrativeloss * contrastive_Loss
            loss = f_loss + domain_loss + contrastive_Loss


            feature_encoder.zero_grad()
            transformer_q.zero_grad()
            domain_classifier.zero_grad()
            loss.backward()
            feature_encoder_optim.step()
            transformer_q_optim.step()
            domain_classifier_optim.step()

            total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
            total_num += querys.shape[0]

        if (episode + 1) % 100 == 0:
            train_loss.append(loss.item())
            print('episode {:>3d}: co_loss: {:6.4f},  domain loss: {:6.4f}, fsl loss: {:6.4f}, acc {:6.4f}, loss: {:6.4f}'.format(episode + 1, \
                                                                                                                contrastive_Loss.item(),
                                                                                                                domain_loss.item(),
                                                                                                                f_loss.item(),
                                                                                                                total_hit / total_num,
                                                                                                               loss.item()))
            if (episode + 1) % 1000 == 0 or episode == 0:
               with torch.no_grad():


                print("Testing ...")
                train_end = time.time()
                feature_encoder.eval()
                total_rewards = 0
                counter = 0
                accuracies = []
                predict = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)


                train_datas, train_labels = train_loader.__iter__().next()
                train_features, _ = feature_encoder(Variable(train_datas).to(device), domain='target')

                max_value = train_features.max()
                min_value = train_features.min()
                print(max_value.item())
                print(min_value.item())
                train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

                KNN_classifier = KNeighborsClassifier(n_neighbors=1)
                KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]

                    test_features, _ = feature_encoder(Variable(test_datas).to(device), domain='target')
                    test_features = transformer_q(test_features, train_features)
                    test_features = (test_features - min_value) * 1.0 / (max_value - min_value)

                    predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                    test_labels = test_labels.numpy()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter
                    accuracies.append(accuracy)

                test_accuracy = 100. * total_rewards / len(test_loader.dataset)

                print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),
                    100. * total_rewards / len(test_loader.dataset)))
                test_end = time.time()


                feature_encoder.train()
                if test_accuracy > last_accuracy:

                    torch.save(feature_encoder.state_dict(),str("DFSL_feature_encoder_" + "IP_" +str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    print("save networks for episode:",episode+1)
                    last_accuracy = test_accuracy
                    best_episdoe = episode

                    acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                    OA = acc
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')

AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))



for i in range(len(best_predict_all)):
    best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
        if best_G[i][j] == 8:
            hsi_pic[i, j, :] = [0.65, 0.35, 1]
        if best_G[i][j] == 9:
            hsi_pic[i, j, :] = [0.75, 0.5, 0.75]
        if best_G[i][j] == 10:
            hsi_pic[i, j, :] = [0.75, 1, 0.5]
        if best_G[i][j] == 11:
            hsi_pic[i, j, :] = [0.5, 1, 0.65]
        if best_G[i][j] == 12:
            hsi_pic[i, j, :] = [0.65, 0.65, 0]
        if best_G[i][j] == 13:
            hsi_pic[i, j, :] = [0.75, 1, 0.65]
        if best_G[i][j] == 14:
            hsi_pic[i, j, :] = [0, 0, 0.5]
        if best_G[i][j] == 15:
            hsi_pic[i, j, :] = [0, 1, 0.75]
        if best_G[i][j] == 16:
            hsi_pic[i, j, :] = [0.5, 0.75, 1]

utils.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24, "IP_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))