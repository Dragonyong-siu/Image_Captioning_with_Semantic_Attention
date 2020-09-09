import torch
import nltk
import numpy as np
from collections import Counter
nltk.download('stopwords')
nltk.download('punkt')
class Semantic_Caption_Dataset(torch.utils.data.Dataset):
  def __init__(self, data, max_len, feature_extractor):
    self.data = data
    self.max_len = max_len
    self.feature_extractor = feature_extractor
    self.config = GPT2_Config
    self.model = GPT2_Model
    self.tokenizer = GPT2_Tokenizer
    self.Encoded_PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')
    self.Encoded_START = self.tokenizer.convert_tokens_to_ids('[START]')
    self.Encoded_END = self.tokenizer.convert_tokens_to_ids('[END]')
  
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    Dictionary = {}

    PIL_Image = self.data[index][0]
    Image_Array = np.asarray(PIL_Image)
    Copied_Image_Array = Image_Array.copy()
    Image_Tensor = torch.Tensor(Copied_Image_Array)

    # Normalize
    Image_Tensor = Image_Tensor.view(3, 256, 256)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    Normalization = torchvision.transforms.Normalize(mean, std)
    Image_Tensor = Normalization(Image_Tensor)
    
    # Feature_Map
    Image_Tensor = Image_Tensor.unsqueeze(0)
    Image_Tensor = Image_Tensor.to(device)
    Feature_Map = self.feature_extractor(Image_Tensor)

    Target_List = self.data[index][1]
    Target_List = [Target_List[0]]
    Target_Caption = " ".join(Target_List).lower()
    Tokenized_Caption = self.tokenizer.tokenize(Target_Caption)
    Encoded_Caption = self.tokenizer.encode(Tokenized_Caption)
    
    # Caption_Ids
    # Caption_Target
    if len(Encoded_Caption) >= (self.max_len - 2):
      Caption_Ids = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)]
      Caption_Target = [self.Encoded_START] + Encoded_Caption[:(self.max_len - 2)] \
      + [self.Encoded_END]
    else:
      Caption_Ids = [self.Encoded_START] + Encoded_Caption
      Caption_Target = [self.Encoded_START] + Encoded_Caption + [self.Encoded_END]
  
    # Padding
    Padding_Length = self.max_len - len(Caption_Ids)
    Caption_Ids = Padding(Caption_Ids, self.Encoded_PAD, Padding_Length)
    Padding_Length = self.max_len - len(Caption_Target)
    Caption_Target = Padding(Caption_Target, self.Encoded_PAD, Padding_Length)

    # Position_Encoding
    Position_Ids = list(range(self.max_len))
    
    # Image_Attr & Stopwords
    Image_Attr = self.data[index][1]
    Image_Attr = " ".join(Image_Attr).lower()
    Attr_tokens = nltk.tokenize.word_tokenize(Image_Attr)
    Stopwords = set(nltk.corpus.stopwords.words('english'))
    Stopwords.add('.')
    Stopwords.add(',')

    # Nonstop_tokens
    Nonstop_tokens = []
    for token in Attr_tokens:
      if token not in Stopwords:
        Nonstop_tokens.append(token)
    
    # Attr_Candidates
    Attr_Candidates = Counter(Nonstop_tokens)
    Attr_number = 10
    if len(Attr_Candidates) < 10:
      Attr_number = len(Attr_Candidates)

    # Attributes_Tokens
    # Attributes_Ids
    Attributes_Tokens = []
    for i in range(Attr_number):
      Attr_List = Attr_Candidates.most_common()
      Attributes_Tokens.append(Attr_List[i][0])
    
    Attributes_Tokens = " " + " " + " ".join(Attributes_Tokens)
    Attributes_Tokens = self.tokenizer.tokenize(Attributes_Tokens)[1:]
    Attributes_Ids = self.tokenizer.encode(Attributes_Tokens)

    # Padding
    Attr_padding = 20 - len(Attributes_Ids)
    Attributes_Ids = Padding(Attributes_Ids, self.Encoded_PAD, Attr_padding)

    # Rezister to Dictionary
    Dictionary['Feature_Map'] = Feature_Map.squeeze(0)
    Dictionary['Caption_Ids'] = torch.Tensor(Caption_Ids).long()
    Dictionary['Caption_Target'] = torch.Tensor(Caption_Target).long()
    Dictionary['Attributes_Ids'] = torch.Tensor(Attributes_Ids).long()

    return Dictionary

def Padding(X, padding_value, padding_length):
  return X + [padding_value] * padding_length

from torch.utils.data import DataLoader
input_data = train_data
caption_length = 20
Train_dataset = Semantic_Caption_Dataset(input_data, 
                                         caption_length,
                                         Semantic_Caption_Encoder)
Train_dataloader = DataLoader(Train_dataset,
                              batch_size = 16,
                              shuffle = True,
                              drop_last = True)
