"""
Test script to demonstrate the custom data collator behavior
"""
from typing import List, Dict

def demonstrate_collator_logic():
    """
    Demonstrates how the custom DataCollatorForCausalLM handles variable-length sequences
    """
    print("=" * 80)
    print("CUSTOM DATA COLLATOR DEMONSTRATION")
    print("=" * 80)
    
    # Example batch with variable-length sequences
    features = [
        {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, 4, 5]},
        {"input_ids": [10, 20, 30], "labels": [10, 20, 30]},
        {"input_ids": [100, 200, 300, 400, 500, 600, 700], "labels": [100, 200, 300, 400, 500, 600, 700]},
    ]
    
    pad_token_id = 0
    pad_to_multiple_of = 8
    
    print("\n📥 INPUT BATCH (before padding):")
    for i, feature in enumerate(features):
        print(f"  Sample {i+1}: length={len(feature['input_ids'])}, ids={feature['input_ids']}")
    
    # Extract input_ids and labels
    input_ids = [feature["input_ids"] for feature in features]
    labels = [feature["labels"] for feature in features]
    
    # Find max length
    max_length = max(len(ids) for ids in input_ids)
    print(f"\n📏 Max length in batch: {max_length}")
    
    # Pad to multiple
    if pad_to_multiple_of is not None:
        padded_max = ((max_length + pad_to_multiple_of - 1) 
                     // pad_to_multiple_of * pad_to_multiple_of)
        print(f"📐 Padded to multiple of {pad_to_multiple_of}: {padded_max}")
        max_length = padded_max
    
    # Pad sequences
    print(f"\n📤 OUTPUT BATCH (after padding to length {max_length}):")
    padded_input_ids = []
    padded_labels = []
    attention_mask = []
    
    for i, (ids, lbls) in enumerate(zip(input_ids, labels)):
        padding_length = max_length - len(ids)
        
        # Pad input_ids with pad_token_id (0)
        padded_ids = ids + [pad_token_id] * padding_length
        padded_input_ids.append(padded_ids)
        
        # Pad labels with -100 (ignore index)
        padded_lbls = lbls + [-100] * padding_length
        padded_labels.append(padded_lbls)
        
        # Create attention mask
        mask = [1] * len(ids) + [0] * padding_length
        attention_mask.append(mask)
        
        print(f"\n  Sample {i+1}:")
        print(f"    Original length: {len(ids)}")
        print(f"    Padding added:   {padding_length}")
        print(f"    Input IDs:       {padded_ids}")
        print(f"    Labels:          {padded_lbls}")
        print(f"    Attention Mask:  {mask}")
    
    print("\n" + "=" * 80)
    print("KEY POINTS:")
    print("=" * 80)
    print("✅ All sequences now have the same length")
    print("✅ Input IDs padded with 0 (pad_token_id)")
    print("✅ Labels padded with -100 (ignored in loss calculation)")
    print("✅ Attention mask shows which tokens are real (1) vs padding (0)")
    print("✅ Ready for batch processing without dimension errors!")
    print("=" * 80)

if __name__ == "__main__":
    demonstrate_collator_logic()
