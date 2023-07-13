import torch 
import torch.nn.functional as F

torch.manual_seed(123)
sentence = torch.tensor([ 0, 7, 1, 2, 5, 6, 4, 3])
embed = torch.nn.Embedding(10, 16)
embedded_sentence = embed(sentence).detach()

d = embedded_sentence.shape[1]
one_U_query = torch.rand(d, d)

h = 8
multihead_U_query = torch.rand(h, d, d)
multihead_U_key = torch.rand(h, d, d)
multihead_U_value = torch.rand(h, d, d)


stacked_inputs = embedded_sentence.T.repeat(8, 1, 1)
multihead_keys = torch.bmm(multihead_U_key, stacked_inputs)
multihead_keys = multihead_keys.permute(0, 2, 1)
print(multihead_keys.shape) 

multihead_values = torch.matmul( multihead_U_value, stacked_inputs)

multihead_values = torch.matmul( multihead_U_value, stacked_inputs)
multihead_values = multihead_values.permute(0, 2, 1)

multihead_z_2 = torch.rand(8, 16)

linear = torch.nn.Linear(8*16, 16)
context_vector_2 = linear(multihead_z_2.flatten())

print(context_vector_2.shape)








