# Transformer Attention Head Importance and Pruning

This project explores **interpretability and efficiency in transformers** by analyzing and pruning attention heads in the **T5 model**.  
The focus is on understanding which attention heads matter most for different NLP tasks and how pruning affects performance.

## Features
- **Evaluation Metrics**
  - ROUGE-L (Longest Common Subsequence based)
  - BLEU score (up to 4-grams, with smoothing)
- **Attention Head Importance**
  - Gradient-based importance scoring for encoder, decoder, and cross-attention heads
  - Heatmap visualization of head importance
- **Head Pruning**
  - Pruning less important heads to introduce sparsity
  - Controlled pruning from encoder, decoder, or cross-attention separately
  - BLEU score vs sparsity plots to evaluate trade-offs
- **Tasks Evaluated**
  - Summarization (CNN/DailyMail)
  - Translation (WMT16 English–German)

## Technologies
- Python (3.9–3.11)  
- PyTorch  
- HuggingFace Transformers & Datasets  
- NumPy, Matplotlib, Seaborn  

## How to Run
```bash
# Install dependencies
pip install torch transformers datasets matplotlib seaborn

# Run the script
python assignment4_transformer_pruning.py
```

## Results
- Identified task-specific and universal heads across summarization and translation.
- Observed performance degradation with increasing sparsity.
- Showed that structured pruning outperforms random pruning in preserving BLEU scores.

## Author
Siddharth Jain
M.Tech  in Robotics and Autonomous Systems, IISc Bengaluru
