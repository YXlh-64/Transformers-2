# Training Fix - Tensor Dimension Mismatch Resolution

## Problem
The training was failing with the following error:
```
ValueError: expected sequence of length 80 at dim 1 (got 103)
ValueError: Unable to create tensor, you should probably activate truncation and/or padding 
with 'padding=True' 'truncation=True' to have batched tensors with the same length.
```

## Root Cause
The `DataCollatorForLanguageModeling` from HuggingFace was not properly handling the padding of labels for causal language modeling tasks. When batching samples of different lengths, it failed to create tensors with consistent dimensions.

## Solution Implemented

### 1. Custom Data Collator
Created a custom `DataCollatorForCausalLM` class that:
- **Extracts input_ids and labels** from each sample in the batch
- **Finds the maximum length** in the current batch
- **Pads to multiple of 8** (optional, for GPU efficiency)
- **Pads input_ids** with the tokenizer's `pad_token_id`
- **Pads labels** with `-100` (PyTorch's ignore index for CrossEntropyLoss)
- **Creates attention_mask** (1 for real tokens, 0 for padding)
- **Converts everything to PyTorch tensors** with consistent dimensions

### 2. Key Features of the Custom Collator

```python
@dataclass
class DataCollatorForCausalLM:
    tokenizer: AutoTokenizer
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        # Handles variable-length sequences properly
        # Pads labels with -100 instead of pad_token_id
        # Creates proper attention masks
```

### 3. Changes Made to train_qwen.py

#### A. Added Custom Data Collator (lines 38-86)
```python
@dataclass
class DataCollatorForCausalLM:
    """Custom data collator that properly handles padding for causal language modeling"""
    # ... implementation
```

#### B. Updated train() method (line 287)
Changed from:
```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=self.tokenizer,
    mlm=False,
    pad_to_multiple_of=8
)
```

To:
```python
data_collator = DataCollatorForCausalLM(
    tokenizer=self.tokenizer,
    pad_to_multiple_of=8
)
```

#### C. Removed Unused Import
Removed `DataCollatorForLanguageModeling` from imports since we're using the custom collator.

## Why This Works

1. **Explicit Padding**: The custom collator explicitly pads each sequence to the max length in the batch
2. **Label Handling**: Labels are padded with `-100` which PyTorch's `CrossEntropyLoss` ignores during loss computation
3. **Attention Mask**: Proper attention masks ensure the model doesn't attend to padding tokens
4. **Tensor Consistency**: All tensors in a batch have the same shape, preventing dimension mismatch errors

## Expected Behavior After Fix

- ✅ Training should start without tensor dimension errors
- ✅ Batches will have consistent tensor shapes
- ✅ Padding tokens will be properly ignored in loss calculation
- ✅ Memory efficiency maintained with `pad_to_multiple_of=8`

## Testing

To verify the fix works, run your training command:
```bash
python train_qwen.py --domain technology --data_dir data/prepared
```

The training should now proceed past the 0% mark without the ValueError.
