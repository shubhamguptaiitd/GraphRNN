#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import networkx as nx  
import matplotlib.pyplot as plt
import operator
import numpy as np
import numpy
import sys
import pdb
from sklearn.model_selection import train_test_split
import random
import sys
import numpy as np
import random
np.set_printoptions(threshold=sys.maxsize)
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from read_graph import read_graphs_in_networkx
from utils import calculate_M,graphs_db,encode_M_matrix,decode_M_matrix
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# In[2]:


data_file = "../data/yeast.txt"
(graphs,node_dict,edge_dict,node_label_freq_dict,edge_label_freq_dict) = read_graphs_in_networkx(data_file,True,100000)
print("Number of graphs loaded " , len(graphs))

(train_graphs,val_graphs) = train_test_split(graphs,test_size=0.4, random_state=42)
(train_graphs,test_graphs) = train_test_split(train_graphs,test_size=0.4, random_state=42)


# In[3]:


len_node_labels = len(node_dict)+1
len_edge_labels = len(edge_dict)+1
weight_vector_node_label = np.ones(len_node_labels)
weight_vector_edge_label = np.ones(len_edge_labels)

node_max_value= max(node_label_freq_dict.items(), key=operator.itemgetter(1))[1]


for label, count in node_label_freq_dict.items():
    weight_vector_node_label[label] = min(100,node_max_value /count)

weight_vector_node_label[0]=1
edge_count_max_value= max(edge_label_freq_dict.items(), key=operator.itemgetter(1))[1]


for label, count in edge_label_freq_dict.items():
    weight_vector_edge_label[label] = min(100,edge_count_max_value /count)

print(weight_vector_node_label,weight_vector_edge_label)
print(edge_label_freq_dict,node_label_freq_dict)


# In[4]:


max_num_edges = max([graph.number_of_edges() for graph in graphs])
min_num_edges = min([graph.number_of_edges() for graph in graphs])
max_num_nodes = max([graph.number_of_nodes() for graph in graphs])
min_num_nodes = min([graph.number_of_nodes() for graph in graphs])
M = int(calculate_M(graphs,len(graphs)))
print("Number of training graphs, max nodes, min nodes , max edges , min edges , max prev nodes", len(train_graphs), max_num_nodes, min_num_nodes, max_num_edges, min_num_edges,M)


# In[ ]:





# In[ ]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# In[ ]:


for batch_ndx, sample in enumerate(graph_loader):
    print(batch_ndx)
    break


# In[ ]:





# ### Training code ###

# In[ ]:



graph_db = graphs_db(train_graphs,max_num_nodes,M)
(a, b,c,d) = graph_db.__getitem__(10)
graph_loader = DataLoader(graph_db,batch_size= 16,num_workers=2,shuffle=True)


# In[55]:



        
graph_db = graphs_db(train_graphs,max_num_nodes,M)
(a, b,c,d) = graph_db.__getitem__(10)
graph_loader = DataLoader(graph_db,batch_size= 32,num_workers=2,shuffle=True)


class CUSTOM_RNN_NODE(torch.nn.Module):
    def __init__(self, input_size, embedding_size=64, hidden_size=32,output_size =None,number_layers=4,name="",len_unique_node_labels=None,len_unique_edge_labels=None):
        super(CUSTOM_RNN_NODE, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.name = name
        self.len_unique_node_labels = len_unique_node_labels
        self.len_unique_edge_labels = len_unique_edge_labels
        
        self.sequence_embedding_size = embedding_size*input_size + embedding_size*4
        self.input = nn.Embedding(self.len_unique_edge_labels, embedding_size)
        self.input2 = nn.Embedding(self.len_unique_node_labels, embedding_size*4)
        self.rnn = nn.GRU(input_size=self.sequence_embedding_size,hidden_size = self.hidden_size,
                                num_layers=self.number_layers,bias=True,batch_first=True,dropout=0)
        self.hidden_n = None
        #self.out = nn.Sequential(nn.Linear(self.hidden_size,self.embedding_size),nn.ReLU(),nn.Linear(self.embedding_size,self.output_size))
        self.out = nn.Sequential(nn.Linear(self.hidden_size,self.sequence_embedding_size),nn.ReLU(),nn.Linear(self.sequence_embedding_size,self.output_size))
        self.relu = nn.ReLU()
        
        ###MLP for loss
        self.Linear = nn.Sequential(nn.ReLU(),nn.Linear(self.output_size,self.len_unique_node_labels))
    def forward(self,input,x_node_label, seq_lengths = None,is_packed=True,is_MLP=False):
        
        input = self.input(input)
        input = self.relu(input)
        input = input.reshape(input.shape[0],input.shape[1],-1)
        input2 = self.input2(x_node_label)
        input_concat =torch.cat((input, input2), 2)
        if is_packed:
            input_concat = pack_padded_sequence(input_concat,seq_lengths,batch_first=True,enforce_sorted=False)
        output,self.hidden_n = self.rnn(input_concat,self.hidden_n)
        
        if is_packed:
            output = pad_packed_sequence(output,batch_first=True)[0]
        output = self.out(output)
        if not is_MLP:
            return output
        
        mlp_output= self.Linear(output)
        return output,mlp_output
        
    def init_hidden(self, batch_size):
        return Variable(torch.zeros(self.number_layers, batch_size, self.hidden_size))
    
class CUSTOM_RNN_EDGE(torch.nn.Module):
    def __init__(self, input_size, embedding_size=64, hidden_size=32,output_size =None,number_layers=4,name="",len_unique_edge_labels=None):
        super(CUSTOM_RNN_EDGE, self).__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.number_layers = number_layers
        self.name = name
        self.len_unique_edge_labels = len_unique_edge_labels
        
        self.embedding= nn.Embedding(self.len_unique_edge_labels,embedding_size)
        self.linear = nn.Linear(self.input_size,self.embedding_size)
        self.rnn = nn.GRU(input_size=self.embedding_size,hidden_size = self.hidden_size,
                                num_layers=self.number_layers,bias=True,batch_first=True,dropout=0)
        self.hidden_n = None
        self.out = nn.Sequential(nn.Linear(self.hidden_size,self.embedding_size),nn.ReLU(),nn.Linear(self.embedding_size,self.output_size))
        
        self.relu = nn.ReLU()
        self.Linear_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.output_size,self.len_unique_edge_labels))
    def forward(self,input, seq_lengths = None,is_mlp=False):
        #print("doing forward loop for rnn ," ,self.name)
        input = self.embedding(input)
        input = self.relu(input)
        input = input.reshape(input.size(0),input.size(1),-1)
        output,self.hidden_n = self.rnn(input,self.hidden_n)
        output = self.out(output)
        
        if not is_mlp:
            return output
        output_mlp = self.Linear_mlp(output)
        return output,output_mlp

        
### Define two RNNs 1 for graph level and 2nd for edge level 
hidden_size_node_rnn = 128
hidden_size_edge_rnn = 32
embedding_size_node_rnn = 64
embedding_size_edge_rnn = 16
num_layers = 4
node_rnn = CUSTOM_RNN_NODE(input_size=M, embedding_size=embedding_size_node_rnn,
                hidden_size=hidden_size_node_rnn, number_layers=num_layers,output_size=len_node_labels,
            name="node",len_unique_node_labels=len_node_labels,len_unique_edge_labels=len_edge_labels)
edge_rnn = CUSTOM_RNN_EDGE(input_size=1, embedding_size=embedding_size_edge_rnn,
                   hidden_size=len_node_labels, number_layers=num_layers, output_size=len_edge_labels,
                    name="edge",len_unique_edge_labels=len_edge_labels)

lr = 0.001
optimizer_node = optim.Adam(list(node_rnn.parameters()), lr=lr)
optimizer_edge = optim.Adam(list(edge_rnn.parameters()),lr=lr)


# In[ ]:





# In[57]:



epochs = 2
most_frequent_edge_label = 2
for epoch in range(0,epochs):
    print("####Epoch#### ", epoch)
    node_rnn.train()
    edge_rnn.train()
    loss = 0
    for ndx, data in enumerate(graph_loader):
        node_rnn.zero_grad()
        edge_rnn.zero_grad()
        
        max_seq_len = max(data[2])
        node_labels = data[3].long()
        node_rnn.hidden_n = node_rnn.init_hidden(batch_size=list(data[0].size())[0]) 
        x = data[0].float()[:,0:max_seq_len,:].long()
        x[:,0] = x[:,0]*most_frequent_edge_label
        y = data[1].float()[:,0:max_seq_len,:].long()
        
        y_node_labels = node_labels
        node_labels = node_labels[:,0:max_seq_len]
        y_node_labels[:,0:-1] = y_node_labels[:,1:]
        y_node_labels[:,-1] = 0
        y_node_labels = y_node_labels[:,0:max_seq_len]
        h = node_rnn(x,node_labels, seq_lengths=data[2])
        h_ce = h.view(-1,h.size(2))
        h = pack_padded_sequence(h,data[2],batch_first=True,enforce_sorted=False).data
        
        #criterion = F.cross_entropy(weight=torch.FloatTensor(weight_vector_node_label))
        #criterion = nn.CrossEntropyLoss()
        #y_node_labels = pack_padded_sequence(y_node_labels,data[2],batch_first=True,enforce_sorted=False).data
        y_node_labels = y_node_labels.reshape(-1)
        loss_node_label = F.cross_entropy(input=h_ce,target=y_node_labels,weight=torch.Tensor(weight_vector_node_label) ) #weight=torch.Tensor(weight_vector_node_label)
        #print(max_seq_len,loss_node_label)
        
        
  
#         ## initialized edge rnn with node rnn hiddent state
        h_edge_tmp = torch.zeros(num_layers-1, h.size(0), h.size(1))
        edge_rnn.hidden_n = torch.cat((h.view(1,h.size(0),h.size(1)),h_edge_tmp),dim=0)
        y_packed = pack_padded_sequence(y,data[2],batch_first=True,enforce_sorted=False).data
        edge_rnn_y = y_packed.view(y_packed.size(0),y_packed.size(1),1)
        edge_rnn_x = torch.cat((torch.ones(edge_rnn_y.size(0),1,1).long()*most_frequent_edge_label,edge_rnn_y[:,0:-1,0:1]),dim=1)
        
        
        edge_rnn_y_pred = edge_rnn(edge_rnn_x)
        
        #edge_rnn_y_pred = pack_padded_sequence(edge_rnn_y_pred,data[2],enforce_sorted=False).data
        #edge_rnn_y = pack_padded_sequence(edge_rnn_y,data[2],enforce_sorted=False).data.reshape(-1)
        #print(edge_rnn_y_pred.size(),edge_rnn_y.size())
        
        #edge_rnn_y_pred = F.sigmoid(edge_rnn_y_pred)
        edge_rnn_y = edge_rnn_y.reshape(-1)
        edge_rnn_y_pred = edge_rnn_y_pred.view(-1,edge_rnn_y_pred.size(2))
        loss_edge_label = F.cross_entropy(edge_rnn_y_pred, edge_rnn_y,weight=torch.Tensor(weight_vector_edge_label))
        #print(loss_edge_label)
        
        total_loss = loss_edge_label+loss_node_label
        total_loss.backward()
        optimizer_edge.step()
        optimizer_node.step()
        print(ndx,total_loss,loss_edge_label,loss_node_label)

        #print(h.size())
        
        
        break


# In[26]:


data[2].sum()


# In[63]:


h_ce[0:10]


# In[62]:


y_node_labels[0:10]


# In[ ]:


edge_rnn_y_pred[104:121]


# In[ ]:





# ### TESTING ####

# In[ ]:





# In[34]:


def pick_random_label(label_freq_dict):
    freq_dict = [(key,value) for key,value in label_freq_dict.items() ]
    freq_dict =sorted(freq_dict,key=lambda val:val[1],reverse=True)
    return random.choice((freq_dict[0][0],freq_dict[1][0]))
def sample_multi(y,num_of_samples=1):
    #print(y)
    y = F.softmax(y,dim=2)
    sampled_y = torch.multinomial(y.view(y.size(0),y.size(2)),num_samples=num_of_samples,replacement=True,)
    #print(sampled_y)
    return sampled_y


# In[ ]:





# In[64]:


num_graphs_to_be_generated = 1000
node_rnn.hidden_n = node_rnn.init_hidden(num_graphs_to_be_generated)
node_rnn.eval()
edge_rnn.eval()
generated_graphs =torch.zeros(num_graphs_to_be_generated, max_num_nodes-1, M)
generated_graphs_labels = torch.zeros(num_graphs_to_be_generated,max_num_nodes-1,1)
node_x = torch.ones(num_graphs_to_be_generated,1,M).long()*most_frequent_edge_label
node_x_label = torch.ones(num_graphs_to_be_generated,1).long()
for i in range(0,num_graphs_to_be_generated):
    node_x_label[i,0]=pick_random_label(node_label_freq_dict)
    #node_x_label[i,0] = 4
node_x_label_1st_node = node_x_label

print("generating")
for i in range(0,max_num_nodes-1):
    print(i)
    h = node_rnn(node_x,node_x_label,None,is_packed=False)
    node_label_sampled = sample_multi(h,1)
    h_edge_tmp = torch.zeros(num_layers-1, h.size(0), h.size(2))
    edge_rnn.hidden_n = torch.cat((h.permute(1,0,2),h_edge_tmp),dim=0)
    edge_x = torch.ones(num_graphs_to_be_generated,1,1).long()*most_frequent_edge_label
    
    
    node_x = torch.zeros(num_graphs_to_be_generated,1,M).long()
    node_x_label = node_label_sampled.long()
    
    
    for j in range(min(M,i+1)):
        edge_rnn_y_pred = edge_rnn(edge_x)
        edge_rnn_y_pred_sampled = sample_multi(edge_rnn_y_pred)
        #print(edge_rnn_y_pred_sampled.size(),node_x[:,:,j:j+1].size())
        node_x[:,:,j:j+1] = edge_rnn_y_pred_sampled.view(edge_rnn_y_pred_sampled.size(0),
                                                         edge_rnn_y_pred_sampled.size(1),1)
        edge_x = edge_rnn_y_pred_sampled.long()
    
    
    #print(node_label_sampled.size(),generated_graphs_labels[:,i+1,:].size())
    generated_graphs_labels[:,i] = node_label_sampled
    generated_graphs[:, i:i + 1, :] = node_x


# In[67]:


generated_graphs_labels[5].numpy()


# In[ ]:


generated_graphs_labels[2]


# In[68]:


a,b = np.unique(generated_graphs_labels,return_counts=True)
print([i for i in zip(a,b)])


# In[ ]:


generated_graphs[10].numpy()


# In[69]:


# # remove all zeros rows and columns
# adj = adj[~np.all(adj == 0, axis=1)]
# adj = adj[:, ~np.all(adj == 0, axis=0)]
# adj = np.asmatrix(adj)
# G = nx.from_numpy_matrix(adj)
# return G

def cut_graph(g,labels):
    tp = np.where(~g.any(axis=1))[0]
    if tp.shape[0] >0:
        g = g[0:tp[0],:]
        labels = labels[0:tp[0],:]
    ct = 0
    index = None
    ct = 0
    labels_list = []
    for i in labels:
        if i[0] == 0:### terminal node
            index = ct
            break
        labels_list.append(i[0])
        ct += 1
    return (g[0:ct,:],labels_list)
predicted_graphs = []
predicted_graphs_x = []
predicted_graphs_x_labels = []
for i in range(num_graphs_to_be_generated):
    pred_graph,pred_labels = cut_graph(generated_graphs[i].numpy(),generated_graphs_labels[i].numpy())
    #pred_labels = [item[0] for item in generated_graphs_labels[i]]
    #pred_graph = generated_graphs[i].numpy()
    predicted_graphs.append(decode_M_matrix(pred_graph,M))
    predicted_graphs_x.append(nx.from_numpy_matrix(predicted_graphs[i]))
    start_label = node_x_label_1st_node[i,0].tolist()
    pred_labels.insert(0,start_label)
    predicted_graphs_x_labels.append(pred_labels)
max_num_edges_test = max([graph.number_of_edges() for graph in predicted_graphs_x])
min_num_edges_test = min([graph.number_of_edges() for graph in predicted_graphs_x])
max_num_nodes_test = max([graph.number_of_nodes() for graph in predicted_graphs_x])
min_num_nodes_test = min([graph.number_of_nodes() for graph in predicted_graphs_x])
print("Number of training graphs, max nodes, min nodes , max edges , min edges , max prev nodes", max_num_nodes_test, min_num_nodes_test, max_num_edges_test, min_num_edges_test,M)


# In[70]:


[graph.number_of_nodes() for graph in predicted_graphs_x]


# In[71]:


predicted_graphs_x = [graph for graph in predicted_graphs_x if graph.number_of_edges() > 20 and graph.number_of_edges() < 80]
print(len(predicted_graphs_x))


# In[ ]:


# save model checkpoint
if args.save:
    if epoch % args.epochs_save == 0:
        fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
        torch.save(rnn.state_dict(), fname)
        fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
        torch.save(output.state_dict(), fname)
        
        
         # check if load existing model
    if args.load:
fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
rnn.load_state_dict(torch.load(fname))
fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
output.load_state_dict(torch.load(fname))

args.lr = 0.00001
epoch = args.load_epoch
print('model loaded!, lr: {}'.format(args.lr))


# In[ ]:





# In[76]:


nx.draw(predicted_graphs_x[20], cmap = plt.get_cmap('jet'))
plt.show()


# In[46]:


nx.draw(graphs[10], cmap = plt.get_cmap('jet'))
plt.show()


# In[ ]:




