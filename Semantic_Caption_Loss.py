def Semantic_Caption_Loss(logits, targets):
  Loss_Function = nn.CrossEntropyLoss()
  Caption_Loss = Loss_Function(logits, targets) 
  return Caption_Loss
