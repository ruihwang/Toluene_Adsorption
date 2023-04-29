from mof import MOF_CGCNN
import csv

##read data
with open('./traning_val.csv') as f:
    readerv = csv.reader(f)
    trainandval = [row for row in readerv]
with open('./test.csv') as f:
    readerv = csv.reader(f)
    test = [row for row in readerv]
from sklearn.model_selection import train_test_split
train, val = train_test_split(trainandval, test_size=0.11, random_state=24)
#file path
root = './cif'
#create a class
mof = MOF_CGCNN(cuda=True,root_file=root,trainset = train, valset=val,testset=test,epoch = 500,lr=0.002,optim='Adam',batch_size=24,h_fea_len=480,n_conv=5,lr_milestones=[200],weight_decay=1e-7,dropout=0.2)
# train the model
mof.train_MOF()

