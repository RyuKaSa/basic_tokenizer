from tokenizers import Tokenizer
from tokenizers import pre_tokenizers
from tokenizers import trainers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel


# instantiate empty untrained BPE model, wrap inside Tokenizer object
tokenizer = Tokenizer(BPE())


# assign Whitespace pretokenizer as attribute
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)


# instantiate BpeTrainer, size 16000 with special tokens
special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
trainer = trainers.BpeTrainer(vocab_size=16000, special_tokens=special_tokens)


# call tokenizer.train(), whilst passing the corpus.txt file, and trainer object
tokenizer.train(["corpus.txt"], trainer=trainer)


# save with tokenizer.save as tokenizer.json
tokenizer.save("tokenizer.json")


# sanity check, encode this sentence, and print
test = "This has led to the recent banning of Neonics in the EU, however the US and Canada are still using this chemical pesticide." 
print(tokenizer.encode(test).ids)
print(tokenizer.encode(test).tokens)

'''
[581, 435, 2639, 247, 220, 1971, 2450, 650, 236, 4567, 219, 685, 240, 220, 4072, 14, 2300, 220, 1483, 243, 3333, 298, 1408, 1018, 450, 2290, 7932, 448, 16]
['This', 'Ġhas', 'Ġled', 'Ġto', 'Ġthe', 'Ġrecent', 'Ġban', 'ning', 'Ġof', 'ĠNe', 'on', 'ics', 'Ġin', 'Ġthe', 'ĠEU', ',', 'Ġhowever', 'Ġthe', 'ĠUS', 'Ġand', 'ĠCanada', 'Ġare', 'Ġstill', 'Ġusing', 'Ġthis', 'Ġchemical', 'Ġpestic', 'ide', '.']
'''