COL 761 - Project - : Graph generative models



Team consist of -
Raj Kamal - 2018CSZ8013
Sahil Manchanda - 2018CSZ8551
Shubham Gupta - 2019CSZ8470


Approach - 

For the first version of generative graph modelling we have utilized the approach of the GraphRNN paper i.e modeling it as a sequence  generating problem.

Since the GraphRNN paper isn't directly capable of handling node labels and edge labels we have utilized the idea present in a paper on Arxiv titled: 

MolecularRNN: Generating realistic molecular graphs with optimized properties
 https://arxiv.org/abs/1905.13372 


WE HAVE NOT UTILIZED ANY SPECIFIC FEATURE RELATED TO MOLECULES.


The idea is to maintain each node label and edge label as a categorical value and learn the embedding of that label based upon the neural network task.
The idea is frequently used in NLP and other problems where there could be a relationship between different categorical variables. 

Similar to GraphRNN, we maintain 2 rnns, one is to create nodes and other is to help in creating edges.

There are few structural changes to graph rnn to support node label and edge label.


1. The node level rnn is fed with sequence of past edges( max M previous, similar to graphrnN) and the previous node's generated label.
The embedding of edge sequence( which is a concatenation of embedding of edge labels in the sequence)  and embedding of  previous node label  is concatenated to create a new embedding. 
This new embedding is
 passed to Node Rnn. We are not sure if Molecular RNN is doing the same thing or not as we didn't have access to its code.

2. A new MLP layer is added to the output of nodeRNN. This was a design choice to get better feature representation and it also improved accuracy.


3. This MLP layer is connected to another layer to get the output of nodeRNN as a set of scores. The size of the output layer depends on number of unique node labels.

4. The edge level RNN is initialized with the MLP layer of nodeRNN.

5. The edge level RNN is fed with previous edge label instead of just existence/absence of edge in GraphRNN. The embedding of previous edge label is passed to edgeRNN.

6. The output is now having multiple classes depending on the number of unique edge labels.

7. The absence of an edge is given special category (0).  

8. If there are X unique labels of node, we have added another label called Terminal node. This actually helped since for every training graph, the node next to the last node in training graph should be given some symbol and it should not be a symbol of actually node label.

So, when sequence to sequence learning is happening while training, if we are at  time T( for terminal node and T is the last node in the graph)  and predict label T+1( since its sequence to sequence task) , the  true label at T+1 is kept as the terminal node label.

The softmax activation is applied to the outputs( PyTorch does it automatically while doing cross entropy loss) .

We utilized a weighted cross entropy loss for both node level and edge level RNN.
The weights were added since some node/edges appear rarely.

We have used batch training similar to graphRNN.


Future work:

1. The idea is to use another loss component which will provide loss  based on the metrics ( orbit, clustering )  to improve the quality  of generated graphs. 
I.e The loss could be delayed by some steps while graph is getting created. If the graph formation is definitely going to be improper it should be caught early rather than at the end.


2. Initialized the node embeddings with a pretrained network like gsage and see it improves.


Code structure - 

python train.py <datafile> to train on the datafile
python test.py "epoch" <num_of_graphs_to_generated> <name of file where graphs to be stored>