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
    self.Sequence_Length = 30
    self.Vocab_size = 50260
    
    self.ReLU = nn.ReLU(inplace = True)
    self.Dropout = nn.Dropout(0.001, inplace = True)
    
    self.Linear_FC = nn.Linear(8 * 8 * 512, self.GPT2_Hidden)
    self.Linear_LM = nn.Linear(self.GPT2_Hidden, self.Vocab_size)
    
    self.Linear_IAU = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_Indiag = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_BAU = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_Bodydiag = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_OAV = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)
    self.Linear_Outdiag = nn.Linear(self.GPT2_Hidden, self.GPT2_Hidden)

  def forward(self, caption_ids, position_ids, feature_map, attributes):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    first_inputs = self.Linear_FC(feature_map) 
    input_wte = self.GPT2_wte(caption_ids)[:, :-1]
    input_wpe = self.GPT2_wpe(position_ids)
    
    first_inputs = first_inputs.unsqueeze(1)
    Orig_Embedding = torch.cat([first_inputs, input_wte], dim = 1)
    Attr_Embedding = self.Head_Attention_Model(Orig_Embedding, attributes)
    input_Embedding = Orig_Embedding + Attr_Embedding + input_wpe
    
    for i in range(self.GPT2_layers):
      input_Embedding = self.GPT2_h[i](input_Embedding)[0]
      input_Embedding = self.Body_Attention_Model(input_Embedding, attributes)

    Orig_Logits = self.GPT2_ln_f(input_Embedding)
    Attr_Logits = self.Foot_Attention_Model(Orig_Logits, attributes)

    Logits = Orig_Logits + Attr_Logits
    Logits = self.Linear_LM(Logits)
    Logits = self.Dropout(Logits)
    return Logits
  
  def Head_Attention_Model(self, before_words, attributes):
    attributes = self.GPT2_wte(attributes).transpose(2, 1)
    Matmul1 = self.Linear_IAU(before_words)
    Matmul2 = torch.matmul(Matmul1, attributes)
    attributes_weight_a = F.softmax(Matmul2, dim = 2)
    
    Matmul3 = torch.matmul(attributes_weight_a, attributes.transpose(1, 2))
    model_inputs = before_words + self.Linear_Indiag(Matmul3)
    return model_inputs

  def Body_Attention_Model(self, body_states, attributes):
    attributes = self.GPT2_wte(attributes).transpose(2, 1)
    Matmul1 = self.Linear_BAU(body_states)
    Matmul2 = torch.matmul(Matmul1, attributes)
    attributes_weight_a = F.softmax(Matmul2, dim = 2)
    
    Matmul3 = torch.matmul(attributes_weight_a, attributes.transpose(1, 2))
    model_inputs = body_states + self.Linear_Bodydiag(Matmul3)
    return model_inputs

  def Foot_Attention_Model(self, hidden_states, attributes):
    attributes = self.GPT2_wte(attributes).transpose(2, 1)
    Matmul1 = self.Linear_OAV(hidden_states)
    Matmul2 = torch.matmul(Matmul1,attributes)
    attributes_weight_b = F.softmax(Matmul2, dim = 2)

    Matmul3 = torch.matmul(attributes_weight_b, attributes.transpose(1, 2))
    model_outputs = hidden_states + self.Linear_Outdiag(Matmul3)
    return model_outputs
  
  def Semantic_Caption_Sampling(self, feature_map, attributes, max_len):
    feature_map = feature_map.view(-1, 8 * 8 * 512)
    feature_inputs = self.Linear_FC(feature_map) 
    feature_inputs = feature_inputs.unsqueeze(1)
    
    Sample_Ids = []
    for i in range(max_len):
      Addition_inputs = torch.Tensor(Sample_Ids)
      Addition_inputs = Addition_inputs.long()
      Addition_inputs = Addition_inputs.to(device)
      Addition_inputs = self.GPT2_wte(Addition_inputs)
      Addition_inputs = Addition_inputs.unsqueeze(0) 
      
      Orig_inputs = torch.cat((feature_inputs, Addition_inputs), dim = 1)
      Attr_inputs = self.Head_Attention_Model(Orig_inputs, attributes)
      
      inputs_length = Orig_inputs.size(1)
      Position_inputs = torch.Tensor(np.arange(inputs_length))
      Position_inputs = Position_inputs.long()
      Position_inputs = Position_inputs.to(device)
      Position_inputs = self.GPT2_wpe(Position_inputs) 
      Model_inputs = Orig_inputs + Attr_inputs + Position_inputs
      
      for j in range(self.GPT2_layers):
        Model_inputs = self.GPT2_h[j](Model_inputs)[0] 
        Model_inputs = self.Body_Attention_Model(Model_inputs, attributes)

      Orig_Logits = self.GPT2_ln_f(Model_inputs)
      Attr_Logits = self.Foot_Attention_Model(Orig_Logits, attributes)

      Logits = Orig_Logits + Attr_Logits
      Sampling_outputs = self.Linear_LM(Logits)

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
    print(Index_List)
    Index_Filtered = []
    for i in range(len(Index_List)):
      if Index_List[i][1] >= 0.35:
        Index_Filtered.append(Index_List[i][0])
    
    if len(Index_Filtered) == 0:
      Index_Filtered = [First_Index]
    
    Words_Index = random.choice(Index_Filtered)
    return Words_Index  

Semantic_Caption_Decoder = Semantic_Caption_Model_GPT2().to(device)
