# hw1-build: caspian-7
import json
class BPETokenizer:
    def __init__(self):
        self.merges=[]
        self.token_bytes={}
        for i in range(256):
            self.token_bytes[i]=bytes([i])
    def train(self,text: str,vocab_size: int)->None:
        if vocab_size<256:
            raise ValueError("vocab_size must be at least 256")
        self.merges=[]
        self.token_bytes={}
        for i in range(256):
            self.token_bytes[i]=bytes([i])
        tokens=list(text.encode("utf-8"))
        num_merges=vocab_size-256
        for merge_index in range(num_merges):
            pair_counts={}
            for i in range(len(tokens)-1):
                pair=(tokens[i],tokens[i+1])
                if pair not in pair_counts:
                    pair_counts[pair]=0
                pair_counts[pair]+=1
            if not pair_counts:
                raise ValueError("Not enough tokens to perform all merges")
            best_pair=None
            best_count=-1
            for pair,count in pair_counts.items():
                if count>best_count:
                    best_count=count
                    best_pair=pair
            new_id=256+merge_index
            self.merges.append(best_pair)
            self.token_bytes[new_id]=self.token_bytes[best_pair[0]]+self.token_bytes[best_pair[1]]
            new_tokens=[]
            i=0
            while i<len(tokens):
                if i<len(tokens)-1 and (tokens[i],tokens[i+1])==best_pair:
                    new_tokens.append(new_id)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens=new_tokens
    def encode(self,text: str)->list:
        tokens=list(text.encode("utf-8"))
        for merge_index,pair in enumerate(self.merges):
            new_id=256+merge_index
            new_tokens=[]
            i=0
            while i<len(tokens):
                if i<len(tokens)-1 and (tokens[i],tokens[i+1])==pair:
                    new_tokens.append(new_id)
                    i+=2
                else:
                    new_tokens.append(tokens[i])
                    i+=1
            tokens=new_tokens
        return tokens
    def decode(self,ids: list)->str:
        output=bytearray()
        for token_id in ids:
            if token_id not in self.token_bytes:
                raise ValueError("Unknown token ID: {}".format(token_id))
            output.extend(self.token_bytes[token_id])
        return output.decode("utf-8")
    def save(self,path: str)->None:
        data={"vocab_size":256+len(self.merges),"merges":self.merges}
        with open(path,"w",encoding="utf-8") as f:
            json.dump(data,f,indent=4)
    @classmethod
    def load(cls,path: str):
        with open(path,"r",encoding="utf-8") as f:
            data=json.load(f)

        if not isinstance(data,dict) or "merges" not in data:
            raise ValueError("Invalid tokenizer file")

        if not isinstance(data["merges"],list):
            raise ValueError("Merges must be a list")

        tokenizer=cls()

        for pair in data["merges"]:
            if not isinstance(pair,list) or len(pair)!=2:
                raise ValueError("Invalid merge pair")

            if not all(isinstance(token_id,int) and token_id in tokenizer.token_bytes for token_id in pair):
                raise ValueError("Invalid token ID in merge pair")

            pair=tuple(pair)
            new_id=256+len(tokenizer.merges)

            tokenizer.merges.append(pair)
            tokenizer.token_bytes[new_id]=tokenizer.token_bytes[pair[0]]+tokenizer.token_bytes[pair[1]]

        if data.get("vocab_size")!=256+len(tokenizer.merges):
            raise ValueError("Vocabulary size mismatch")

        return tokenizer
