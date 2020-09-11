# Image_Captioning_with_Semantic_Attention

0) Download Coco_Dataset.zip and Unzip

1) Data_Tranforms 
 1.1) Resize to (256, 256, 3)
 1.2) Make couple :(Image, Caption_Target)

2) Prepare GPT2 & Add special tokens

3) Semantic_Caption_Encoder
  CNN_Feature_Extractor : VGG19 pretrained

4) Semantic_Caption_Dataset
 4.1) Feature_Map from Image
 4.2) Encoded Target that will be captions : GPT2
 4.3) Attributes from Attribute_Detector
   NIC_GPT2 (Before train, Use original_data) : Trash Stopwords

5) Semantic_Caption_Decoder
 5.1) Using LSTMCell as Model
 5.2) Using GPT2 as Model

6) Semantic_Caption_Loss : CrossEntropyLoss

7) Semantic_Caption_Train
