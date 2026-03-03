from datasets import load_dataset

# first call will download dataset and expose in local huging face cache
dataset = load_dataset("agentlans/high-quality-english-sentences")