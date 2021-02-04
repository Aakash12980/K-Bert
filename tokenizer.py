from transformers import BertTokenizer


class Tokenizer():
    def __init__(self, max_len=512):
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

    def encode_batch(self, text_batch):
        src_tokens = self.tokenizer(text_batch, add_special_tokens=True,
                return_token_type_ids=False, padding="longest", truncation=True,
                return_attention_mask=True, return_tensors="pt")
        
        return src_tokens.input_ids, src_tokens.attention_mask