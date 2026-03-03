# loads dataset
# use load_dataset from datasets library
from datasets import load_dataset

# access the train split and store in a variable
dataset = load_dataset("agentlans/high-quality-english-sentences", split="train")


# open a new file called corpus.txt in write mode, UTF8 encoding, explicit,
import os
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
corpus_file_name = "corpus.txt"
corpus_file = os.path.join(THIS_FOLDER, corpus_file_name)

f = open(corpus_file_name, "w", encoding="utf-8")
f.write("This dataset comes from agentlans, it is named high quality english sentences, and will be used here to train a simple tokenizer.")    # Write inside file 
# this first line serves as credit, we shall keep it in the corpus

# we can check the format like this, and make sure its a dict format
print(type(dataset[2]))
print(dataset[2])
'''
{'text': 'This has led to the recent banning of Neonics in the EU, however the US and Canada are still using this chemical pesticide.'} 
'''

count = 0
# iterate over every row from the train split, format is a dict,
# pull from the text field, and copy it to corpus file, add a newline character
for row in dataset:
    text = str(row["text"]) + "\n"
    f.write(text)
    count += 1
f.close() 


# after loop, sanity check, print total lines of text file, and print loop number
print("count of loop   : ", count)
print("count of dataset: ", len(dataset))