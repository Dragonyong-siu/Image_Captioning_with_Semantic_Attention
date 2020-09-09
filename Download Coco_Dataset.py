!wget http://images.cocodataset.org/zips/val2014.zip
!unzip val2014.zip

import torchvision
from torchvision import transforms
Data_Transforms = torchvision.transforms.Compose([transforms.Resize((256, 256))])
Train_data = torchvision.datasets.CocoCaptions(root = 'val2014/',
                                               annFile = '/content/gdrive/My Drive/captions_val2014.json',
                                               transform = Data_Transforms,
                                               target_transform = None,
                                               transforms = None)
