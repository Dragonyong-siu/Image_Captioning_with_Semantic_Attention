from tqdm import tqdm
def Semantic_Caption_Train(dataloader, model, optimizer, device):
  model.train()
  Book = tqdm(dataloader, total = len(dataloader))
  total_loss = 0.0
  for bi, Dictionary in enumerate(Book):
    Caption_Ids = Dictionary['Caption_Ids']
    Feature_Map = Dictionary['Feature_Map']
    Caption_Target = Dictionary['Caption_Target']
    Attributes_Ids = Dictionary['Attributes_Ids']

    Caption_Ids = Caption_Ids.to(device)
    Feature_Map = Feature_Map.to(device)
    Caption_Target = Caption_Target.to(device)
    Attributes_Ids = Attributes_Ids.to(device)

    model.zero_grad()
    Logits = model(Caption_Ids, Feature_Map, Attributes_Ids)
    
    Logits = Logits.view(-1, 50260)
    Caption_Target = Caption_Target.view(-1)

    Caption_Loss = Semantic_Caption_Loss(Logits, Caption_Target)
    Caption_Loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    total_loss += Caption_Loss

  Average_Caption_Loss = total_loss / len(dataloader)
  print(" Average_Caption_Loss: {0:.2f}".format(Average_Caption_Loss))

def FIT(model, Epoch, Learning_Rate):
  optimizer = torch.optim.AdamW(model.parameters(), lr = Learning_Rate)
  for i in range(Epoch):
    print(f"EPOCHS:{i+1}")
    print('TRAIN')
    Semantic_Caption_Train(Train_dataloader, model, optimizer, device)
  torch.save(model, '/content/gdrive/My Drive/' + f'Semantic_Caption_Decoder')
    
FIT(Semantic_Caption_Decoder, Epoch = 5, Learning_Rate = 0.001)
