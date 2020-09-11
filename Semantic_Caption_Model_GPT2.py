import random
import torch.nn.functional as F
class Semantic_Caption_Model_GPT2(nn.Module):
  def __init__(self):
    super(Semantic_Caption_Model_GPT2, self).__init__()
    self.GPT2_Model = GPT2_Model
    self.GPT2_wte = self.GPT2_Model.wte
    self.GPT2_wpe = self.GPT2_Model.wpe
    self.GPT2_drop = self.GPT2_Model.drop
    self.GPT2_h = self.GPT2_Model.h
    self.GPT2_ln_f = self.GPT2_Model.ln_f

    self.GPT2_Hidden = 768
    self.GPT2_layers = 12
    self.Model_insize = 768
    self.Model_outsize = 768
    self.Model_lmsize = 768
    self.Sequence_Length = 15
    self.Vocab_size = 50260
    
    self.ReLU = nn.ReLU(inplace = True)
    self.Dropout = nn.Dropout(0.01, inplace = True)
    
    self.Linear_FC = nn.Linear(8 * 8 * 512, self.GPT2_Hidden)
    self.Linear_HH = nn.Linear(8 * 8 * 512, self.Model_insize)
    self.Linear_CC = nn.Linear(8 * 8 * 512, self.Model_insize)
    self.Linear_LM = nn.Linear(self.Model_lmsize, self.Vocab_size)
    
    self.Linear_IAW = nn.Linear(self.GPT2_Hidden, self.Model_insize)
    self.Linear_IAU = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_Indiag = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_Indomain = nn.Linear(self.GPT2_Hidden, self.Model_insize)
    self.Linear_OAV = nn.Linear(self.Model_outsize, self.GPT2_Hidden)
    self.Linear_Outdiag = nn.Linear(self.GPT2_Hidden, self.Model_outsize)
    self.Linear_Outdomain = nn.Linear(self.Model_outsize, self.Model_lmsize)

  def forward(self, caption_ids, position_ids, feature_map, attributes):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    first_inputs = self.Linear_FC(feature_map) 
    first_inputs = self.ReLU(first_inputs)
    input_wte = self.GPT2_wte(caption_ids)[:, :-1]
    input_wpe = self.GPT2_wpe(position_ids)
    
    model_inputs_init = self.Input_Attention_Model_init(first_inputs)
    model_inputs = self.Input_Attention_Model(input_wte, attributes)
    input_Embedding = torch.cat([model_inputs_init.unsqueeze(1), model_inputs], dim = 1)
    input_Embedding = input_Embedding + input_wpe
    ## input_Embedding = self.GPT2_drop(input_Embedding)
    
    for i in range(self.GPT2_layers):
      input_Embedding = self.GPT2_h[i](input_Embedding)[0]

    Hidden_states = self.GPT2_ln_f(input_Embedding)
    Logits = self.Output_Attention_Model(Hidden_states, attributes)
    Logits = self.Dropout(Logits)
    return Logits
    
  def Input_Attention_Model_init(self, image_v):
    model_inputs = self.Linear_IAW(image_v)
    return model_inputs
  
  def Input_Attention_Model(self, before_words, attributes):
    attributes = self.GPT2_wte(attributes).transpose(2, 1)
    Matmul1 = self.Linear_IAU(before_words)
    Matmul2 = torch.matmul(Matmul1, attributes)
    attributes_weight_a = F.softmax(Matmul2, dim = 2)
    
    Matmul3 = torch.matmul(attributes_weight_a, attributes.transpose(1, 2))
    model_inputs = before_words + self.Linear_Indiag(Matmul3)
    model_inputs = self.Linear_Indomain(model_inputs)
    return model_inputs

  def Output_Attention_Model(self, hidden_states, attributes):
    attributes = self.GPT2_wte(attributes).transpose(2, 1)
    Matmul1 = self.Linear_OAV(hidden_states)
    Matmul2 = torch.matmul(Matmul1,attributes)
    attributes_weight_b = F.softmax(Matmul2, dim = 2)

    Matmul3 = torch.matmul(attributes_weight_b, attributes.transpose(1, 2))
    model_outputs = hidden_states + self.Linear_Outdiag(Matmul3)
    model_outputs = self.Linear_Outdomain(model_outputs)
    model_outputs = self.ReLU(model_outputs)
    model_outputs = self.Linear_LM(model_outputs)
    return model_outputs
  
  def Semantic_Caption_Sampling(self, feature_map, attributes, max_len):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    feature_inputs = self.Linear_FC(feature_map) 
    feature_inputs = self.ReLU(feature_inputs)
    feature_inputs = self.Input_Attention_Model_init(feature_inputs) 
    feature_inputs = feature_inputs.unsqueeze(1)
    
    Sample_Ids = []
    for i in range(max_len):
      Addition_inputs = torch.Tensor(Sample_Ids)
      Addition_inputs = Addition_inputs.long()
      Addition_inputs = Addition_inputs.to(device)
      Addition_inputs = self.GPT2_wte(Addition_inputs)
      Addition_inputs = Addition_inputs.unsqueeze(0) 
      Addition_inputs = self.Input_Attention_Model(Addition_inputs, attributes)
      
      Sampling_inputs = torch.cat((feature_inputs, Addition_inputs), dim = 1)
      inputs_length = Sampling_inputs.size(1)
      
      Position_inputs = torch.Tensor(np.arange(inputs_length))
      Position_inputs = Position_inputs.long()
      Position_inputs = Position_inputs.to(device)
      Position_inputs = self.GPT2_wpe(Position_inputs) 
      Model_inputs = Sampling_inputs + Position_inputs
      ## Model_inputs = self.GPT2_drop(Model_inputs)
      
      for j in range(self.GPT2_layers):
        Model_inputs = self.GPT2_h[j](Model_inputs)[0] 

      hidden_states = self.GPT2_ln_f(Model_inputs)
      Sampling_outputs = self.Output_Attention_Model(hidden_states, attributes)

      Words_Index = self.Next_Word_Index(Sampling_outputs[:, -1, :])
      Words_Index = torch.Tensor([Words_Index])
      Words_Index = Words_Index.long().to(device)
      Sample_Ids.append(Words_Index) 
    Sample_Ids = torch.stack(Sample_Ids, dim = 1)
    return Sample_Ids
      
  def Next_Word_Index(self, logits):
    Last_Word_Embedding = logits[0, :]
    Softmax_logits = F.softmax(Last_Word_Embedding, dim = 0)
    Words_Probability = Softmax_logits.tolist()
    Words_Sorted = sorted(Words_Probability)

    First_value = Words_Sorted[-1]
    Second_value = Words_Sorted[-2]
    Third_value = Words_Sorted[-3]

    First_Index = Words_Probability.index(First_value)
    Second_Index = Words_Probability.index(Second_value)
    Third_Index = Words_Probability.index(Third_value)

    Index_List = [(First_Index, First_value),
                  (Second_Index, Second_value),
                  (Third_Index, Third_value)]

    Index_Filtered = []
    for i in range(len(Index_List)):
      if Index_List[i][1] >= 0.35:
        Index_Filtered.append(Index_List[i][0])
    
    if len(Index_Filtered) == 0:
      Index_Filtered = [First_Index]
    
    Words_Index = random.choice(Index_Filtered)
    return Words_Index  

Semantic_Caption_Decoder = Semantic_Caption_Model_GPT2().to(device)
