from datasets import load_dataset
from dotenv import load_dotenv
from matplotlib import pyplot as plt

load_dotenv()
eval_dataset = load_dataset("UCSC-VLAA/MedVLThinker-Eval")
print(eval_dataset['test'][:5])
print(eval_dataset['test'][0]['images'])
plt.imshow(eval_dataset['test'][0]['images'][0])
plt.show()