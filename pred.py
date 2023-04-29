import time
start = time.clock()    
from mof import MOF_CGCNN
import csv

with open('./tandv.csv') as f:
    readerv = csv.reader(f)
    trainandval = [row for row in readerv]
with open('./test.csv') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]
from sklearn.model_selection import train_test_split

train, val = train_test_split(trainandval, test_size=0.89, random_state=8)


with open('./hmof_d.csv') as f:
    readerv = csv.reader(f)
    pred = [row for row in readerv]


## dataset
train_root =  './dataset_cif/mofs-exp'
pred_root = './hmof_cif'
mof = MOF_CGCNN(works=8,root_file=train_root,trainset = train[:1], valset=val[:1],testset=test[:1],epoch = 1,lr=0,optim='Adam',batch_size=24,h_fea_len=480,n_conv=5,lr_milestones=[160],weight_decay=5e-8,dropout=0.2)
#pred_experiment
mof.pred_MOF(pred_root,pred,'./model_best.pth.tar')
end = time.clock()
runTime = end - start
print("运行时间：", runTime)
