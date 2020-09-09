from transformers import GPT2Tokenizer, GPT2Config, GPT2Model
special_tokens = '[PAD]', '[START]', '[END]'
GPT2_Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
GPT2_Config = GPT2Config.from_pretrained('gpt2')
GPT2_Model = GPT2Model(GPT2_Config).from_pretrained('gpt2', config = GPT2_Config)
GPT2_Tokenizer.add_tokens(special_tokens)
GPT2_Model.resize_token_embeddings(len(GPT2_Tokenizer))
