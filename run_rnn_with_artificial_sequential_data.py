#from basic_rnn import BasicRNN
from basic_rnn_using_tensorflow_api import BasicRNN
from sequential_data import SequentialData

# Data generations
num_epochs = 10
data_size = 1000000
batch_size = 200
num_steps = 10
num_classes = 2

# RNN
state_size = 16
learning_rate = 0.001

sequential_data = SequentialData(data_size=data_size, batch_size=batch_size,
                                 num_steps=num_steps, num_classes=num_classes)

basic_rnn = BasicRNN(state_size=5, num_steps=num_steps, num_classes=num_classes,
                    learning_rate=learning_rate)

for epoch in sequential_data.gen_epoch(num_epochs):
    for batch in epoch:
        basic_rnn.update_params(batch)
