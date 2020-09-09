device = 'cuda'
import torch.nn as nn
from collections import OrderedDict
from torchvision import models
class Semantic_Caption_Model_VGG19(nn.Module):
  def __init__(self):
    super(Semantic_Caption_Model_VGG19, self).__init__()
    #self.Feature_Extraction_Layer = nn.Sequential(
    #    OrderedDict([('Conv_1', nn.Conv2d(3, 64, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_1', nn.BatchNorm2d(64)),
    #                 ('ReLU_1', nn.ReLU(inplace = True )),
    #                 ('Conv_2', nn.Conv2d(64, 64, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_2', nn.BatchNorm2d(64)),
    #                 ('ReLU_2', nn.ReLU(inplace = True)),
    #                 ('MaxPool_1', nn.MaxPool2d(2, stride = 2)),
    #
    #                 ('Conv_3', nn.Conv2d(64, 128, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_3', nn.BatchNorm2d(128)),
    #                 ('ReLU_3', nn.ReLU(inplace = True)),
    #                 ('Conv_4', nn.Conv2d(128, 128, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_4', nn.BatchNorm2d(128)),
    #                 ('ReLU_4', nn.ReLU(inplace = True)),
    #                 ('MaxPool_2', nn.MaxPool2d(2, stride = 2)),
    #
    #                 ('Conv_5', nn.Conv2d(128, 256, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_5', nn.BatchNorm2d(256)),
    #                 ('ReLU_5', nn.ReLU(inplace = True)),
    #                 ('Conv_6', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_6', nn.BatchNorm2d(256)),
    #                 ('ReLU_6', nn.ReLU(inplace = True)),
    #                 ('Conv_7', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_7', nn.BatchNorm2d(256)),
    #                 ('ReLU_7', nn.ReLU(inplace = True)),
    #                 ('Conv_8', nn.Conv2d(256, 256, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_8', nn.BatchNorm2d(256)),
    #                 ('ReLU_8', nn.ReLU(inplace = True)),
    #                 ('MaxPool_3', nn.MaxPool2d(2, stride = 2)),
    #
    #                 ('Conv_9', nn.Conv2d(256, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_9', nn.BatchNorm2d(512)),
    #                 ('ReLU_9', nn.ReLU(inplace = True)),
    #                 ('Conv_10', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_10', nn.BatchNorm2d(512)),
    #                 ('ReLU_11', nn.ReLU(inplace = True)),
    #                 ('Conv_12', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_12', nn.BatchNorm2d(512)),
    #                 ('ReLU_12', nn.ReLU(inplace = True)),
    #                 ('Conv_13', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_13', nn.BatchNorm2d(512)),
    #                 ('ReLU_13', nn.ReLU(inplace = True)),
    #                 ('Maxpool_4', nn.MaxPool2d(2, stride = 2)),
    #
    #                 ('Conv_14', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_14', nn.BatchNorm2d(512)),
    #                 ('ReLU_14', nn.ReLU(inplace = True)),
    #                 ('Conv_15', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_15', nn.BatchNorm2d(512)),
    #                 ('ReLU_15', nn.ReLU(inplace = True)),
    #                 ('Conv_16', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_16', nn.BatchNorm2d(512)),
    #                 ('ReLU_16', nn.ReLU(inplace = True)),
    #                 ('Conv_17', nn.Conv2d(512, 512, 3, stride = 1, padding = 1)),
    #                 ('Batch_Norm_17', nn.BatchNorm2d(512)),
    #                 ('ReLU_17', nn.ReLU(inplace = True)),
    #                 ('Maxpool_5', nn.MaxPool2d(2, stride = 2)),]))
    self.Feature_Extraction_Layer = models.vgg19_bn(pretrained = True).features
    
  def forward(self, Image):
    Feature_Map = self.Feature_Extraction_Layer(Image).to(device)
    return Feature_Map

Semantic_Caption_Encoder = Semantic_Caption_Model_VGG19().to(device)
