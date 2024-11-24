"""
# Part 2: Import Libraries
"""

# !pip install datasets -q

# importing required libraries
import torch
import torch.nn as nn
import collections
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import warnings

from torch.optim import AdamW
from typing import List
from torch.nn import functional as F
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, T5Tokenizer, T5ForSequenceClassification
from torch.utils.data import DataLoader

warnings.simplefilter("ignore")
print(torch.__version__)

device = "cuda" if torch.cuda.is_available() else "cpu"

""" LoRA Adapters

We will inject LoRA into the **key, query, and value** matrices of each transformer block.

Recall from the LoRA paper that LoRA enhances model training efficiency by reducing the need to retrain all pretrained weights.
Instead, it introduces two smaller matrices, A and B, which capture the necessary adaptations for the new task.
This significantly reduces computational overhead while maintaining high performance.

For more information, read the [paper](https://arxiv.org/pdf/2106.09685).

By using LoRA in our causal model, we aim to achieve
efficient fine-tuning with minimal computational cost,
focusing on the key, query, and value matrices within each transformer block.

LoRA class
First, let's implement the LoRA class based on how it is defined in the paper.
"""

class LoRALayer():
    def __init__(
        self,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x

class LoRAAdapter(nn.Module, LoRALayer):
    def __init__(
        self,
        existing_layer: nn.Module,
        in_features,
        out_features,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.,
        **kwargs
    ):
        nn.Module.__init__(self)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.existing_layer = existing_layer

        self.r = r

        existing_dtype = next(existing_layer.parameters()).dtype
        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros((in_features, r), dtype=existing_dtype))
            self.lora_B = nn.Parameter(torch.zeros((r, out_features), dtype=existing_dtype))
            self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    ## Resets the two matrices (A and B) based on how the paper does it
    def reset_parameters(self):
        if hasattr(self, 'lora_A'):
            nn.init.normal_(self.lora_A, mean=0, std=0.02) # mean=0, std=0.02 by recommendation
        if hasattr(self, 'lora_B'):
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        self.existing_layer.train(mode)

    def forward(self, x: torch.Tensor):
        if self.r > 0:
            return self.existing_layer(x) + (self.lora_dropout(x) @ self.lora_A @ self.lora_B) * self.scaling
        else:
            return self.existing_layer(x)

"""## 5.2 Inject into the model

Recall in LoRA that we want to freeze the pre-trained model and only train our adapter weights `lora_A` and `lora_B`.

Complete `mask_only_lora_as_trainable` so that only those weights require gradients.
"""

# TODO: Finish the method
def mark_only_lora_as_trainable(model: nn.Module) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LoRAAdapter):
            for param in module.parameters():
                param.requires_grad = True
        else:
            for param in module.parameters():
                param.requires_grad = False

"""Finally, we want to write the code that will inject the LoRA adapters into our causal model.
Complete the following methods so that we can correctly inject our LoRA adapters into the model.
`match_submodules`: Returns a list of names of layers in a model whose names match a specified key.
`get_submodule`: Retrieves a specific submodule from a model based on its name.
`replace_submodule`: Replaces a specific submodule in a model with a new module at a given path.
```
Code Hint:
You can use the set_attr and get_attr methods to get and replace submodules.
```

`inject_adapter`: Replaces all submodules in a model that match any string in a list with a new module created by an adapter function.

```
Code Hint:
Remember to put the adapters onto GPU
```

```
Code Hint:
Here is an example of `inject_adapter` usage:
inject_adapter(model, ["query_key_value"], lambda x: LoRAAdapter(x, r=8,lora_alpha=8, in_features=x.in_features, out_features=x.out_features))
```

"""

def match_submodules(model: nn.Module, key:str) -> List[str]:
    return [name for name, _ in model.named_modules() if key in name]

def get_submodule(model: nn.Module, module_name:str):
    return model.get_submodule(module_name)

def replace_submodule(model: nn.Module, module_path: str, new_module):
    # model.set_attr(module_path, new_module)
    # Split the path into parts
    parts = module_path.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    # Set the new module on the parent
    setattr(parent, parts[-1], new_module)

def inject_adapter(model: nn.Module, match_on: List[str], adapter_fn):
    for key in match_on:
        for module_name in match_submodules(model, key):
            module = get_submodule(model, module_name)
            replace_submodule(model, module_name, adapter_fn(module))

"""## 5.3 Evaluation on a benchmark

Next, we want to inject the LoRA adapter into our causal model we defined earlier.
Let's also check to see how many parameters are in this model, as well as how many of these parameters are considered trainable.

Re-initialize the causal model and chck the model architecture.

```
Code Hint:
The name of the model is "facebook/opt-125m"
```
"""

causal_model_name = "facebook/opt-125m"
causal_model = AutoModelForCausalLM.from_pretrained(causal_model_name, torch_dtype=torch.bfloat16, device_map="auto")
causal_tokenizer = AutoTokenizer.from_pretrained(causal_model_name)

# TODO: Check the model architecture
print(causal_model)

"""Next, we want to call the inject_adapter method on our causal model and see how this changed our model architecture.
Calculate and print the total number of parameters as well as the number of trainable parameters after we inject LoRA into our model.
"""

inject_adapter(causal_model, ["q_proj","k_proj","v_proj"], lambda x: LoRAAdapter(x, r=8, lora_alpha=8, in_features=x.in_features, out_features=x.out_features))
mark_only_lora_as_trainable(causal_model)

total_params = sum(p.numel() for p in causal_model.parameters())
trainable_params = sum(p.numel() for p in causal_model.parameters() if p.requires_grad)

print(f"Total Parameters: {total_params}")
print(f"Trainable Parameters: {trainable_params}")

"""Finally, run the cell below to check the new model's architecture. If the key, value, and query matrices are all now replaced by a LoRA adapter, you are good to go!"""

# Check the new model architecture
print(causal_model)
"""## 5.4: Finetuning your LoRA adapters on Wikitext
In this next part, we will finally finetune the LoRA adapter of our causal model on a small subset of the training set of Wikitext.
If all went correctly, we should notice that the perplexity over our test set went down!
Since we are only using a small subset of the training set and a low chunk size, you shouldn't expect the perplexity to go down by much (<1 point).
**Note:** Please be aware that this code may take some time to run (you are literally training a large language model), so please be fully confident in your completed code above.  With this being said **please ensure that you record the final perplexity score** (you may even want to screenshot it for proof).
First, let's define our finetuning function:
"""

def finetune_causal_model(model, train_dataset, epochs=1, learning_rate=1e-4):
        def tokenize_function(examples):
            result = causal_tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256) #256 chosen for Colab's GPU size
            result["labels"] = result["input_ids"].copy()
            return result

        train_dataset = Dataset.from_dict(train_dataset)
        tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
        data_collator = DataCollatorForLanguageModeling(causal_tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir="/content",
            evaluation_strategy="epoch",
            per_device_train_batch_size=8,
            learning_rate=learning_rate,
            weight_decay=0.01,
            num_train_epochs=epochs,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            eval_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        trainer.train()

"""Next, let's load our training dataset.

A few interesting things to note: The training dataset can be quite large with respect to our compute resources, so we're only going to use a small fraction of it.  Also, we are going to split our text into chunks so that the attention gradients can fit on Colab's GPU.

"""

wiki_training_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
chunks = []

# As big as Colab's GPU can fit
chunk_size = 256


def split_into_chunks(text, chunk_size):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

for example in wiki_training_dataset:
    text = example['text']
    text_chunks = split_into_chunks(text, chunk_size)
    chunks.extend(text_chunks)

processed_train_dataset = {'text':chunks[:len(chunks)//10]}

"""Finally, calculate the score of our new model."""

# Import wikitext dataset
causal_test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
causal_test_encodings = causal_tokenizer("\n\n".join(causal_test["text"]), return_tensors="pt")

def calc_perplexity(model, encodings, stride):
    max_length = 1024
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return torch.exp(torch.stack(nlls).mean())

finetune_causal_model(causal_model, processed_train_dataset)
calc_perplexity(causal_model, causal_test_encodings, 256)

"""## 5.5 Conceptual Questions
**Question:** What do you think the benefits of using LoRA are?  What might be some drawbacks?

### Benefits:
1. **Reduced Memory and Storage Usage**:
   - By freezing most parameters and adapting only a few (using low-rank matrices A and B), LoRA reduces memory usage substantially. For instance, in large models like GPT-3 (175B parameters), VRAM usage drops from 1.2TB to 350GB, which saves resources during training.

2. **Efficient Optimizer State Management**:
   - Because only a subset of parameters is updated (A and B matrices), VRAM requirements are cut by up to 2/3, making training more feasible for large models on fewer GPUs.

3. **Smaller Checkpoint Sizes**:
   - LoRA compresses model checkpoints by storing just the adapter parameters instead of all weights, which reduces I/O demands and storage costs.

4. **Task Switching Flexibility**:
   - LoRA allows for cost-effective switching between tasks in a deployed model by swapping small LoRA weights rather than the entire model. This is especially helpful in applications needing multiple task-specific adaptations without reloading full model parameters.

5. **Speedup During Training**:
   - LoRA speeds up training by roughly 25% on large models since gradient calculations are only done for adapted parameters, not for all parameters. This can make training both faster and less resource-intensive.

### Drawbacks:
1. **Batching for Mixed Tasks is Challenging**:
   - When dealing with a batch containing samples from multiple tasks, it’s difficult to adapt LoRA weights in a single forward pass. If weights are merged into the main model weights \( W \) for efficiency, changing A and B within a batch is challenging. The only alternative is to keep A and B separate, adding latency for dynamic selection of LoRA modules.

2. **Potential Latency in Dynamic LoRA Selection**:
   - For applications requiring rapid inference across multiple tasks, dynamically switching LoRA modules can introduce latency.


<hr>

**Question:** Discuss the trade-offs between model size, speed, and accuracy when using LoRA in LLMs.

### Model Size
   - **Reduced Storage**: LoRA minimizes model size by adapting only a few parameters, cutting storage and memory costs significantly.
   - **Trade-Off**: Too few adapted parameters might limit accuracy on complex tasks, especially if only a low-rank approximation is used.

### Speed
   - **Efficiency**: LoRA speeds up training and inference by focusing updates on specific layers, reducing computational load.
   - **Latency**: For multi-task settings, dynamically switching LoRA modules can add latency, affecting real-time applications.

### Accuracy
   - **Task Adaptability**: LoRA achieves near full fine-tuning accuracy for related tasks. But as said in the drawbacks, we need to switch modules for distinct tasks.

"""

