# QWEN Fine-tuning Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QWEN FINE-TUNING PIPELINE                        │
│                 Domain-Specific English-Arabic Translation          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         PHASE 1: DATA PREPARATION                    │
└─────────────────────────────────────────────────────────────────────┘

    Input: Raw CSV Files
    ├── technology.csv         (English-Arabic pairs)
    ├── economic v1.csv        (English-Arabic pairs)
    └── Education_Research.csv (English-Arabic pairs)
                ↓
    [prepare_data.py]
                ↓
    ├── Load and clean data
    ├── Train/Val/Test split (80/10/10)
    ├── Format with QWEN chat template
    └── Save as JSONL
                ↓
    Output: Prepared Datasets
    └── data/prepared/
        ├── technology_train.jsonl
        ├── technology_val.jsonl
        ├── technology_test.jsonl
        ├── economic_train.jsonl
        ├── economic_val.jsonl
        ├── economic_test.jsonl
        ├── education_train.jsonl
        ├── education_val.jsonl
        └── education_test.jsonl


┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 2: MODEL TRAINING                         │
└─────────────────────────────────────────────────────────────────────┘

    Base Model: Qwen/Qwen2-1.5B-Instruct
                ↓
    [train_qwen.py] or [train_all.py]
                ↓
    ┌───────────────────┬───────────────────┬───────────────────┐
    │   Technology      │    Economic       │    Education      │
    │   Domain          │    Domain         │    Domain         │
    └───────────────────┴───────────────────┴───────────────────┘
            ↓                   ↓                   ↓
    ┌──────────────────────────────────────────────────────────┐
    │  Training Process (Per Domain)                           │
    │  ──────────────────────────────                          │
    │  1. Load base QWEN model                                 │
    │  2. Apply 4-bit quantization (QLoRA)                     │
    │  3. Add LoRA adapters                                    │
    │  4. Load domain-specific data                            │
    │  5. Train for N epochs                                   │
    │  6. Save LoRA adapters                                   │
    └──────────────────────────────────────────────────────────┘
                ↓
    Output: Fine-tuned Models
    └── models/
        ├── technology_model/
        │   ├── adapter_config.json
        │   ├── adapter_model.safetensors
        │   └── tokenizer files
        ├── economic_model/
        │   └── (same structure)
        └── education_model/
            └── (same structure)


┌─────────────────────────────────────────────────────────────────────┐
│                       PHASE 3: EVALUATION                            │
└─────────────────────────────────────────────────────────────────────┘

    Trained Models + Test Data
                ↓
    [evaluate.py]
                ↓
    ├── Load fine-tuned model
    ├── Translate test examples
    ├── Calculate BLEU scores
    ├── Calculate chrF scores
    └── Generate example outputs
                ↓
    Output: Evaluation Results
    └── results/
        ├── technology_results.json
        │   ├── BLEU: 35-45
        │   ├── chrF: 60-70
        │   └── Examples: 5 samples
        ├── economic_results.json
        └── education_results.json


┌─────────────────────────────────────────────────────────────────────┐
│                      PHASE 4: DEPLOYMENT & USE                       │
└─────────────────────────────────────────────────────────────────────┘

    [app.py - Streamlit UI]
                ↓
    ┌───────────────────────────────────────────────────┐
    │  Web Interface Features:                          │
    │  ─────────────────────────                        │
    │                                                    │
    │  📝 Translation Tab                               │
    │     - Input English text                          │
    │     - Select domain                               │
    │     - Get Arabic translation                      │
    │     - View metrics                                │
    │                                                    │
    │  🔍 Compare Domains                               │
    │     - Test across all models                      │
    │     - Side-by-side comparison                     │
    │                                                    │
    │  📊 Evaluation                                    │
    │     - View BLEU/chrF scores                       │
    │     - Interactive charts                          │
    │     - Example translations                        │
    │                                                    │
    │  📜 History                                       │
    │     - Past translations                           │
    │     - Usage statistics                            │
    └───────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────────┘

    User Input (English Text)
            ↓
    ┌──────────────────┐
    │  Domain Selector │ ← User chooses: Technology/Economic/Education
    └──────────────────┘
            ↓
    ┌──────────────────────────────────┐
    │  Load Domain-Specific Model      │
    │  ────────────────────────────    │
    │  - Base: QWEN 2 (1.5B)           │
    │  - LoRA adapters for domain      │
    │  - 4-bit quantization            │
    └──────────────────────────────────┘
            ↓
    ┌──────────────────────────────────┐
    │  Translation Generation          │
    │  ───────────────────────────     │
    │  - Apply chat template           │
    │  - Generate with domain context  │
    │  - Decode Arabic output          │
    └──────────────────────────────────┘
            ↓
    Arabic Translation Output


┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING CONFIGURATION                         │
└─────────────────────────────────────────────────────────────────────┘

    Model Architecture:
    ├── Base Model: QWEN 2 (1.5B parameters)
    ├── Quantization: 4-bit (QLoRA)
    ├── LoRA Rank: 16
    ├── LoRA Alpha: 32
    └── Target Modules: All attention + FFN layers

    Training Settings:
    ├── Learning Rate: 2e-4
    ├── Batch Size: 4 × 4 (effective 16)
    ├── Epochs: 3
    ├── Optimizer: Paged AdamW
    ├── Precision: BFloat16
    └── Gradient Checkpointing: Enabled

    Hardware Requirements:
    ├── GPU: 16GB+ VRAM (CUDA)
    ├── RAM: 16GB+
    ├── Storage: 50GB+
    └── Training Time: ~8-12 hours per domain


┌─────────────────────────────────────────────────────────────────────┐
│                         FILE ORGANIZATION                            │
└─────────────────────────────────────────────────────────────────────┘

    LLMs/QWEN/
    │
    ├── 📄 Core Scripts
    │   ├── prepare_data.py      → Data preprocessing
    │   ├── train_qwen.py        → Single domain training
    │   ├── train_all.py         → Batch training
    │   ├── evaluate.py          → Model evaluation
    │   └── app.py               → Streamlit UI
    │
    ├── 📚 Documentation
    │   ├── README.md            → Complete guide
    │   ├── QUICKSTART.md        → 5-min start guide
    │   ├── PROJECT_SUMMARY.md   → Project overview
    │   └── WORKFLOW.md          → This file!
    │
    ├── ⚙️ Configuration
    │   ├── requirements.txt     → Dependencies
    │   ├── config.ini           → Settings
    │   ├── .gitignore          → Git exclusions
    │   └── setup.fish/sh       → Setup scripts
    │
    ├── 📓 Interactive
    │   └── quick_start.ipynb   → Jupyter notebook
    │
    └── 📁 Generated (during use)
        ├── data/prepared/      → Processed datasets
        ├── models/            → Trained models
        └── results/           → Evaluation outputs


┌─────────────────────────────────────────────────────────────────────┐
│                      QUICK COMMAND REFERENCE                         │
└─────────────────────────────────────────────────────────────────────┘

    # Setup
    pip install -r requirements.txt

    # Prepare data
    python prepare_data.py

    # Train single domain
    python train_qwen.py --domain technology

    # Train all domains
    python train_all.py

    # Evaluate model
    python evaluate.py --model_path models/technology_model --domain technology

    # Quick translation
    python evaluate.py --model_path models/technology_model --domain technology \
        --translate_text "Your text here"

    # Launch UI
    streamlit run app.py

    # Monitor training
    tensorboard --logdir models/

    # Check GPU
    nvidia-smi


┌─────────────────────────────────────────────────────────────────────┐
│                         SUCCESS CRITERIA                             │
└─────────────────────────────────────────────────────────────────────┘

    ✅ Data Preparation Complete
       → All JSONL files created in data/prepared/

    ✅ Training Successful
       → Model directories created in models/
       → Training loss decreased
       → No OOM errors

    ✅ Evaluation Passed
       → BLEU scores: 30-45
       → chrF scores: 55-70
       → Results saved to results/

    ✅ UI Working
       → Streamlit launches successfully
       → Translations generate correctly
       → Charts display properly


┌─────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────┘

    CSV Files → DataPreparator → JSONL Files → QWENTrainer → Models
                                                                  ↓
    User Input → Streamlit UI ← QWENTranslator ← Evaluator ← Models


┌─────────────────────────────────────────────────────────────────────┐
│                         TIMELINE ESTIMATE                            │
└─────────────────────────────────────────────────────────────────────┘

    Phase 1: Setup & Data Prep     →  30 minutes
    Phase 2: Training              →  24-36 hours (all domains)
    Phase 3: Evaluation            →  1-2 hours
    Phase 4: Testing & Refinement  →  2-4 hours
    ────────────────────────────────────────────
    Total Project Time             →  28-43 hours


┌─────────────────────────────────────────────────────────────────────┐
│                       CUSTOMIZATION POINTS                           │
└─────────────────────────────────────────────────────────────────────┘

    ⚡ For Faster Training:
       - Reduce epochs: --num_train_epochs 2
       - Smaller batch: --per_device_train_batch_size 2
       - Shorter sequences: --max_seq_length 256

    🎯 For Better Quality:
       - More epochs: --num_train_epochs 5
       - Larger model: --model_name Qwen/Qwen2-7B-Instruct
       - Higher LoRA rank: --lora_r 32

    💾 For Less Memory:
       - Smaller batch: --per_device_train_batch_size 2
       - Shorter sequences: --max_seq_length 256
       - Gradient accumulation: --gradient_accumulation_steps 8


┌─────────────────────────────────────────────────────────────────────┐
│                     🎉 YOU'RE READY TO START! 🎉                    │
└─────────────────────────────────────────────────────────────────────┘

    Begin with: python prepare_data.py
    Then follow the workflow above!
```
