import sys
import os
import time
import string
import numpy as np
from collections import defaultdict
import six
import shutil
import argparse
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
from torch import nn, optim
from collections import defaultdict

class DataIterator(object):
    def __init__(self, tune_lens, tune_idxs, batch_size, random_lens=False):
        self.batch_size = batch_size
        self.ntunes = len(tune_lens)
        self.tune_idxs = tune_idxs

        self.len2idx = defaultdict(list)
        for k, v in zip(tune_lens, tune_idxs):
            self.len2idx[k].append(v)

        self.random_lens = random_lens
        self.rng = np.random.RandomState(42)

    def __iter__(self):
        if self.random_lens:
            for batch_idxs in self.__iter_random_lens():
                yield np.int32(batch_idxs)
        else:
            for batch_idxs in self.__iter_homogeneous_lens():
                yield np.int32(batch_idxs)

    def __iter_random_lens(self):
        available_idxs = np.copy(self.tune_idxs)
        while len(available_idxs) >= self.batch_size:
            rand_idx = self.rng.choice(range(len(available_idxs)), size=self.batch_size, replace=False)
            yield available_idxs[rand_idx]
            available_idxs = np.delete(available_idxs, rand_idx)

    def __iter_homogeneous_lens(self):
        for idxs in six.itervalues(self.len2idx):#.itervalues():
            self.rng.shuffle(idxs)

        progress = defaultdict(int)
        available_lengths = list(self.len2idx.keys())

        batch_idxs = []
        b_size = self.batch_size

        get_tune_len = lambda: self.rng.choice(available_lengths)
        k = get_tune_len()

        while available_lengths:
            batch_idxs.extend(self.len2idx[k][progress[k]:progress[k] + b_size])
            progress[k] += b_size
            if len(batch_idxs) == self.batch_size:
                yield batch_idxs
                batch_idxs = []
                b_size = self.batch_size
                k = get_tune_len()
            else:
                b_size = self.batch_size - len(batch_idxs)
                i = available_lengths.index(k)
                del available_lengths[i]
                if not available_lengths:
                    break
                if i == 0:
                    k = available_lengths[0]
                elif i >= len(available_lengths) - 1:
                    k = available_lengths[-1]
                else:
                    k = available_lengths[i + self.rng.choice([-1, 0])]

#config 
one_hot = True
embedding_size = 256  # is ignored if one_hot=True
num_layers = 3
rnn_size = 512
dropout = 0.2

learning_rate = 0.003
learning_rate_decay_after = 20
learning_rate_decay = 0.97

batch_size = 64
max_epoch = 100
grad_clipping = 5
validation_fraction = 0.05
validate_every = 1000  # iterations

save_every = 10  # epochs

data_path = "allabcwrepeats_parsed"
with open(data_path, 'r') as f:
    data = f.read()
    
def remove_title(tune):
    return ('\n').join(tune.split('\n')[1:])

tunes = data.split('\n\n')

# Remove all the tiles to reduce vocab size
tunes = [remove_title(tune) for tune in tunes]

tokens_set = set('\n\n'.join(tunes).split())

#tokens_set = set(data.split())
start_symbol, end_symbol = '<s>', '</s>'
tokens_set.update({start_symbol, end_symbol})

idx2token = sorted(list(tokens_set)) # needs to be sorted to be the same after reloading the vocab
vocab_size = len(idx2token)
print('vocabulary size:', vocab_size)
token2idx = dict(zip(idx2token, range(vocab_size)))

start_symbol_id = token2idx['<s>']
end_symbol_id = token2idx['</s>']


tunes = [[token2idx[c] for c in [start_symbol] + t.split() + [end_symbol]] for t in tunes]

# set maximum length to 500
tunes = [t for t in tunes if len(t) <= 500]

tunes.sort(key=lambda x: len(x), reverse=True)
ntunes = len(tunes)

tune_lens = np.array([len(t) for t in tunes])
max_len = max(tune_lens)


nvalid_tunes = ntunes * validation_fraction
nvalid_tunes = int(batch_size * max(1, np.rint(
    nvalid_tunes / float(batch_size))))  # round to the multiple of batch_size

rng = np.random.RandomState(42)
valid_idxs = rng.choice(np.arange(ntunes), nvalid_tunes, replace=False)

ntrain_tunes = ntunes - nvalid_tunes
train_idxs = np.delete(np.arange(ntunes), valid_idxs)

print('n tunes:', ntunes)
print('n train tunes:', ntrain_tunes)
print('n validation tunes:', nvalid_tunes)
print('min, max length', min(tune_lens), max(tune_lens))

def create_batch(idxs):
    max_seq_len = max([len(tunes[i]) for i in idxs])
    x = np.zeros((batch_size, max_seq_len), dtype='float32')
    mask = np.zeros((batch_size, max_seq_len - 1), dtype='float32')
    for i, j in enumerate(idxs):
        x[i, :tune_lens[j]] = tunes[j]
        mask[i, : tune_lens[j] - 1] = 1
    return x, mask

def np_onehot(indices, depth, dtype=np.int32):
    """Converts 1D array of indices to a one-hot 2D array with given depth."""
    onehot_seq = np.zeros((len(indices), depth), dtype=dtype)
    onehot_seq[np.arange(len(indices)), indices] = 1.0
    return onehot_seq

def np_onehots(seqs, depth, dtype=np.int32):
    """Converts 2D array of indices to a one-hot 3D array with given depth."""
    return np.array([np_onehot(s, depth) for s in seqs])

class FolkRNN(nn.Module):
    def __init__(self, hparams, device):
        
        super().__init__()
        
        self.device = device
        
        self.global_step = 0
        self.best_val_loss = np.inf
        
        self.input_dim = hparams['input_dim']
        self.lstm_size = hparams['lstm_size']
        self.lstm_layers = hparams['num_lstm_layers']
        self.embedding_size = hparams['embedding_size']
        self.dropout = nn.Dropout(hparams['dropout'])
        self.vocab = hparams['vocab']
        self.vocab_size = len(self.vocab)
        self.start_symbol_id = hparams['start_symbol_id']
        self.end_symbol_id = hparams['end_symbol_id']
        self.max_len = hparams['max_len']
        
        # Make the input projection size the same as the lstm size
        #self.input_projection = nn.Linear(self.input_dim, self.embedding_size)
        
        initial_embeddings = np.random.normal(scale=0.01, size=(self.vocab_size, self.embedding_size))
        self.embedding_layer = nn.Embedding(vocab_size, self.embedding_size)
        self.embedding_layer.load_state_dict({'weight': torch.from_numpy(initial_embeddings)})
        
        self.lstm = nn.LSTM(self.embedding_size, hidden_size=self.lstm_size,
                            num_layers=self.lstm_layers, dropout=hparams['dropout'], batch_first=False)
        
        self.output_projection = nn.Linear(self.lstm_size, self.input_dim)
        
        self.to(device)
    
    def _sample(self, rnn_output, temperature=1.0):
        sampler = torch.distributions.Categorical(logits=rnn_output)
        return sampler.sample()

    def forward(self, inputs=None, targets=None, sampling_probability=0., batch_size=1):
        # input shape [sequence_length, batch_size]
        # target shape  [sequence_length, batch_size]
            
        if inputs is not None:
            length = min(inputs.shape[0], self.max_len)
            batch_size = inputs.shape[1]
            # get input embeddings - input shape  [batch_size, sequence_length, embedding_size]
            input_embeddings = self.embedding_layer(inputs)
            lstm_input = input_embeddings[0:1,:,:] # input feature at time t0 is start symbol
        else:
            length = self.max_len
            start_input = torch.full(size=(1,batch_size), fill_value=self.start_symbol_id, dtype=torch.int64)
            lstm_input = self.embedding_layer(start_input.to(self.device))     
        
        outputs = []
        logits = []
        losses = []
        
        state = None        
         
        # Loop over timesteps
        for i in range(1, length):
            lstm_output, state = self.lstm(lstm_input, state)
            lstm_out = self.output_projection(lstm_output)
            logits.append(lstm_out)
            out = self._sample(lstm_out)
            outputs.append(out)
            
            if inputs is None or np.random.uniform(0,1) < sampling_probability:
                lstm_input = self.embedding_layer(out)
            else:
                lstm_input = input_embeddings[i:i+1,:,:]              
        
        outputs = torch.cat(outputs,dim=0)
        logits = torch.cat(logits,dim=0)
        
        return outputs, logits
    
def make_reverse_vocab(vocab, default_type=str, merge_fn=None):
    # Flip the keys and values in a dict.
    """ Straightforward function unless the values of the vocab are 'unhashable'
        i.e. a list. For example, a phoneme dictionary maps 'SAY' to
        ['S', 'EY1']. In this case, pass in a function merge_fn, which specifies
        how to combine the list items into a hashable key. This could be a
        lambda fn, e.g merge_fn = lambda x: '_'.join(x).
        It's also possible that there could be collisions - e.g. with
        homophones. If default_type is list, collisions will be combined into
        a list. If not, they'll be overwritten.
        Args:
            merge_fn: a function to combine lists into hashable keys
    """
    rv = defaultdict(default_type)
    for k in vocab.keys():
        if merge_fn is not None:
            if default_type is list:
                rv[merge_fn(vocab[k])].append(k)
            else:
                rv[merge_fn(vocab[k])] = k
        else:
            if default_type is list:
                rv[vocab[k]].append(k)
            else:
                rv[vocab[k]] = k
    return rv

def readable_outputs(seq, reverse_vocab, end_symbol_id=None):
    """ Convert a sequence of output indices to readable string outputs
    from a given (reverse) vocab """
    if end_symbol_id is None:
        return ' '.join([reverse_vocab[s] for s in seq])
    else:
        outputs = []
        for s in seq:
            outputs.append(reverse_vocab[s])
            if s == end_symbol_id:
                break
        return ' '.join(outputs)
    
rv = make_reverse_vocab(token2idx)

# pytorch stuff

def check_zero_grads(model):
    zero_grads = sum([float((p.grad==0).sum().detach().cpu()) for p in model.parameters()])
    nonzero_grads = sum([float((p.grad!=0).sum().detach().cpu()) for p in model.parameters()])
    print(f"Params with zero gradient: {zero_grads}")
    print(f"Params with Nonzero gradient: {nonzero_grads}")

def save_checkpoint(state, is_best, checkpoint_dir, name='last.pth.tar'):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    Modified from: https://github.com/cs230-stanford/cs230-code-examples/
    """
    filepath = os.path.join(checkpoint_dir, name)
    if not os.path.exists(checkpoint_dir):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint_dir))
        os.mkdir(checkpoint_dir)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint_dir, 'best.pth.tar'))


def load_checkpoint(checkpoint_dir, model, optimizer=None, name='last.pth.tar'):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    Modified from: https://github.com/cs230-stanford/cs230-code-examples/
    """
    checkpoint = os.path.join(checkpoint_dir, name)
    if not os.path.exists(checkpoint):
        raise Exception("File doesn't exist {}".format(checkpoint))
    else:
        print("Loading checkpoint at:", checkpoint)
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    if 'epoch' in checkpoint:
        model.epoch = checkpoint['epoch']

    if 'global_step' in checkpoint:
        model.global_step = checkpoint['global_step'] + 1
        print("Loading checkpoint at step: ", model.global_step)

    if 'best_val_loss' in checkpoint:
        model.best_val_loss = checkpoint['best_val_loss']

    return checkpoint

def make_state_dict(model, optimizer=None, epoch=None, global_step=None,
    best_val_loss=None):
    return {'epoch': epoch, 'global_step': global_step,
        'best_val_loss': best_val_loss, 'state_dict': model.state_dict(),
        'optim_dict' : optimizer.state_dict()
    }

def count_parameters(model):
    counts = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {counts:,} trainable parameters')

def init_weights(model):
    for name, param in model.named_parameters():
        nn.init.normal_(param.data, mean=0, std=0.01)

def train(model, epochs=20, sampling_probability=0., tb_writer=None):
    model.train()
    t0 = time.time()
    train_loss_buffer = []
    for e in range(epochs):
        print(f"Epoch: {e}")
        for i, train_batch_idxs in enumerate(train_data_iterator):
            optimizer.zero_grad()
            x_batch, mask_batch = create_batch(train_batch_idxs)
            lstm_inputs = torch.from_numpy(x_batch.T).long().to(device)
            targets = torch.from_numpy(x_batch.T[1:]).long().to(device)
            outputs, logits = model(lstm_inputs, sampling_probability=sampling_probability)
            losses = [criterion(logits[i], targets[i]) for i in range(len(targets))]
            loss = torch.stack(losses).mean()
            train_loss_buffer.append(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            #check_zero_grads(model)
            optimizer.step(); model.global_step += 1
            if model.global_step % 100 == 0:
                train_loss_at_step = torch.stack(train_loss_buffer).mean()
                print(f"Train Loss at step {model.global_step}: {train_loss_at_step}, Time: {time.time()-t0}")
                train_loss_buffer = []
                eval_loss = evaluate(model, sampling_probability=sampling_probability)
                print(f"Eval Loss at step {model.global_step}: {eval_loss}, Time: {time.time()-t0}")
                if eval_loss < model.best_val_loss:
                    state = make_state_dict(model, optimizer=optimizer, 
                        global_step=model.global_step, best_val_loss=model.best_val_loss)
                    save_checkpoint(state, is_best=True, checkpoint_dir=checkpoint_dir)
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/train", train_loss_at_step, model.global_step)
                    tb_writer.add_scalar("loss/eval", eval_loss, model.global_step)
                model.train()
                
def evaluate(model, sampling_probability):
    model.eval()
    for i, val_batch_idxs in enumerate(valid_data_iterator):
        total_loss = []
        with torch.no_grad():
            x_batch, mask_batch = create_batch(val_batch_idxs)
            lstm_inputs = torch.from_numpy(x_batch.T).long().to(device)
            targets = torch.from_numpy(x_batch.T[1:]).long().to(device)
            outputs, logits = model(lstm_inputs, sampling_probability=sampling_probability)
            losses = [criterion(logits[i], targets[i]) for i in range(len(targets))]
            loss = torch.stack(losses).mean()
            total_loss.append(loss.detach().cpu())
    return np.mean(total_loss)
        
def sample(model, n_samples, batch_size=64):
    all_samples = []
    n_batches = int(np.ceil(n_samples/batch_size))
    for i in range(n_batches):
        batch = model.forward(batch_size=batch_size)[0].detach().cpu().numpy().T
        for j in range(batch_size):
            txt_sample = readable_outputs(batch[j], rv, end_symbol_id = token2idx['</s>'])
            all_samples.append(txt_sample)
    return all_samples[0:n_samples]

train_data_iterator = DataIterator(tune_lens[train_idxs], train_idxs, batch_size, random_lens=False)
valid_data_iterator = DataIterator(tune_lens[valid_idxs], valid_idxs, batch_size, random_lens=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hparams = {}
hparams['input_dim'] = len(idx2token)
hparams['lstm_size'] = rnn_size
hparams['num_lstm_layers'] = num_layers
hparams['embedding_size'] = embedding_size
hparams['dropout'] = dropout
hparams['vocab'] = token2idx
hparams['start_symbol_id'] = start_symbol_id
hparams['end_symbol_id'] = end_symbol_id
hparams['max_len'] = max_len



parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_dir', type=str)
parser.add_argument('--sampling_probability', type=str, default='0.1')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_samples', type=str, default=10) # In sample mode, number of samples to generate
parser.add_argument('--output_file', type=str) # In sample mode, file to save the samples to

args = parser.parse_args()
checkpoint_dir = args.checkpoint_dir
sampling_probability = float(args.sampling_probability)
mode = args.mode
n_samples = int(args.n_samples)
output_file = args.output_file


# Initialize Model
model = FolkRNN(hparams, device)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters())

count_parameters(model)

if os.path.exists(checkpoint_dir):
    load_checkpoint(checkpoint_dir, model, optimizer)
else:
    print("Saving checkpoints to ", checkpoint_dir)
    
writer = SummaryWriter(checkpoint_dir)
    
if mode == 'sample':
    samples = sample(model, n_samples=n_samples)
    txt_out = "\n\n".join(samples)
    if output_file is not None:
        with open(output_file, 'w') as f:
            f.write(txt_out)
    else:
        print(txt_out)
    
elif mode == 'train':
    train(model, epochs=50, sampling_probability=sampling_probability, tb_writer=writer)
    
# How to use
# For training:
# python folk_rnn.py --mode=train --checkpoint_dir=folkrnn_samp0 --sampling_probability=0.0

# For sampling
# python folk_rnn.py --mode=sample --n_samples=5 --output_file=mysamples_0 --checkpoint_dir=folkrnn_samp0
