import sys
import importlib
if sys.version_info < (3,0):
    import cPickle as pickle
else:
    import pickle
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('metadata_path')
parser.add_argument('--temperature', type=float, default=1.0)

parser.add_argument('--seed')
parser.add_argument('--transcription_id', type=int, default=1)

args = parser.parse_args()

metadata_path = args.metadata_path
temperature = args.temperature
seed = args.seed
transcription_id = args.transcription_id

with open(metadata_path, "rb") as f:
    metadata = pickle.load(f)

token2idx = metadata['token2idx']

start_idx = token2idx['<s>']

LSTM_Wxi=[]
LSTM_Wxf=[]
LSTM_Wxc=[]
LSTM_Wxo=[]
LSTM_Whi=[]
LSTM_Whf=[]
LSTM_Whc=[]
LSTM_Who=[]
LSTM_bi=[]
LSTM_bf=[]
LSTM_bc=[]
LSTM_bo=[]
LSTM_cell_init=[]
LSTM_hid_init=[]
htm1=[]
ctm1=[]

numlayers=3 # hard coded for now, but this should be saved in the model pickle
for jj in range(numlayers):
    LSTM_Wxi.append(metadata['param_values'][2+jj*14-1])
    LSTM_Whi.append(metadata['param_values'][3+jj*14-1])
    LSTM_bi.append(metadata['param_values'][4+jj*14-1])
    LSTM_Wxf.append(metadata['param_values'][5+jj*14-1])
    LSTM_Whf.append(metadata['param_values'][6+jj*14-1])
    LSTM_bf.append(metadata['param_values'][7+jj*14-1])
    LSTM_Wxc.append(metadata['param_values'][8+jj*14-1])
    LSTM_Whc.append(metadata['param_values'][9+jj*14-1])
    LSTM_bc.append(metadata['param_values'][10+jj*14-1])
    LSTM_Wxo.append(metadata['param_values'][11+jj*14-1])
    LSTM_Who.append(metadata['param_values'][12+jj*14-1])
    LSTM_bo.append(metadata['param_values'][13+jj*14-1])
    LSTM_cell_init.append(metadata['param_values'][14+jj*14-1])
    LSTM_hid_init.append(metadata['param_values'][15+jj*14-1])
    htm1.append(LSTM_hid_init[jj])
    ctm1.append(LSTM_cell_init[jj])

FC_output_W = metadata['param_values'][43];
FC_output_b = metadata['param_values'][44];

def sigmoid(x): return 1/(1 + np.exp(-x))
def softmax(x,T): 
    expx=np.exp(x/T)
    sumexpx=np.sum(expx)
    if sumexpx==0:
       maxpos=x.argmax()
       x=np.zeros(x.shape, dtype=x.dtype)
       x[0][maxpos]=1
    else:
       x=expx/sumexpx
    return x

sizeofx=LSTM_Wxi[0].shape[0]
x = np.zeros(sizeofx, dtype=np.int8)
# Converting the seed passed as an argument into a list of idx
seed_sequence = [start_idx]
distribution_matrix=[]
if seed is not None:
    for token in seed.split(' '):
         seed_sequence.append(token2idx[token])
         
    # Running the annotation through the network
    for tok in seed_sequence[:-1]:
        x = np.zeros(sizeofx, dtype=np.int8)
        x[tok] = 1
        for jj in range(numlayers):
            it=sigmoid(np.dot(x,LSTM_Wxi[jj]) + np.dot(htm1[jj],LSTM_Whi[jj]) + LSTM_bi[jj])
            ft=sigmoid(np.dot(x,LSTM_Wxf[jj]) + np.dot(htm1[jj],LSTM_Whf[jj]) + LSTM_bf[jj])
            ct=np.multiply(ft,ctm1[jj]) + np.multiply(it,np.tanh(np.dot(x,LSTM_Wxc[jj]) + np.dot(htm1[jj],LSTM_Whc[jj]) + LSTM_bc[jj]))
            ot=sigmoid(np.dot(x,LSTM_Wxo[jj]) + np.dot(htm1[jj],LSTM_Who[jj]) + LSTM_bo[jj])
            ht=np.multiply(ot,np.tanh(ct))
            x=ht
            ctm1[jj]=ct
            htm1[jj]=ht
        
        #Prop distribution vector for next token 
        distribution_for_next_token = softmax(np.dot(x,FC_output_W) + FC_output_b,temperature)
        distribution_matrix.append(distribution_for_next_token[-1].squeeze())


#To numpy matrix
distribution_matrix = np.array(distribution_matrix)

#Save matrix to file
filename_matrix = "matrix_%d.txt" %transcription_id
np.savetxt(fname=filename_matrix, X=distribution_matrix)