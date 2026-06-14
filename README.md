# VGU-DSAI_2026 self-guided project

## CTC Handwritten Mathematical Expression recognition

This notebook trains an LSTM + CTC model on the CROHME 2019 online handwriting dataset (InkML traces) for math expression / character recognition, with manual training loop, WandB logging, checkpointing, and early stopping.

`InkML strokes -> features -> Dataset/DataLoader -> BiLSTM -> CTC loss -> decoder -> WER metrics -> WandB training log`

Refer to this for basic information of what will be implementing in this notebook. [[link](https://www.youtube.com/watch?v=aD3X1ils4J4)]

This notebook focuses on the implementation contract, expected tensor shapes, formulas, checks, and grading requirements.

## Resources

- [Starter Notebook](https://www.kaggle.com/code/ntcuong2103/crohme-ctc-pytorch-project): starter notebook used in this assignment
- [CROHME 2019 Kaggle dataset](https://www.kaggle.com/datasets/ntcuong2103/crohme2019): dataset used in this assignment.
- [PyTorch Dataset and DataLoader](https://pytorch.org/docs/stable/data.html): reference for custom datasets, batching, and `collate_fn`.
- [PyTorch `pad_sequence`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html): pads variable-length sequence tensors.
- [PyTorch `CTCLoss`](https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html): exact input/target shape requirements for CTC training.
- [PyTorch training recipe](https://pytorch.org/tutorials/recipes/recipes/defining_a_neural_network.html): reference for manual model, optimizer, and loss usage.
- [WandB environment variables](https://docs.wandb.ai/guides/track/environment-variables/): configure `WANDB_API_KEY` securely in Kaggle.
- [Distill CTC guide](https://distill.pub/2017/ctc/): optional intuition for CTC alignments.
- [Torchaudio CTC decoder tutorial](https://docs.pytorch.org/audio/stable/tutorials/asr_inference_with_ctc_decoder_tutorial.html): optional reference for greedy and beam-search decoding.

## Assignment API Contract

| Symbol | Required behavior |
|---|---|
| `get_unique_tokens(annotation_paths)` | Return all non-empty label tokens from all annotation files. |
| `Vocab` | Build/save/load vocabulary, blank token `""` at index `0`, encode/decode token sequences. |
| `Segment`, `Inkml` | Provided InkML parser classes. Do not redefine them. |
| `feature_extraction(traces)` | Return `float32` array `(T, 4)` with `(dx/d, dy/d, d, pen_up)`. |
| `plot_inkml`, `plot_feature_channels`, `plot_emission_heatmap` | Return Matplotlib figures for static inspection. |
| `InkmlDataset` | Return `(features, targets, input_len, target_len)` for one sample. |
| `collate_fn(batch)` | Pad variable-length features/targets batch-first and return length tensors. |
| `create_dataloaders(...)` | Build train/validation/test `DataLoader` objects with the custom collate function. |
| `LSTMTemporalClassifier` | BiLSTM classifier returning log-probs `(B, T, C)`. |
| `GreedyCTCDecoder` | Greedy CTC decode: argmax, collapse repeats, remove blank. |
| `edit_distance`, `word_error_rate` | Token-level edit distance and WER. |
| `compute_ctc_loss` | Compute one CTC loss from a model, criterion, batch, and device. |
| `train_one_epoch` | Run one manual training epoch over a dataloader. |
| `evaluate_model` | Run manual evaluation and return loss/WER metrics. |
| `fit_with_early_stopping` | Run the manual training loop with WandB logging, checkpointing, and early stopping. |

## Local/ Colab Notebook
## 1. Requirements
 
- Google Colab (GPU runtime recommended: `Runtime > Change runtime type > GPU`)
- A Weights & Biases (WandB) account, if you want experiment logging
- The CROHME 2019 dataset (InkML files + annotation `.txt` files), as a `.zip` archive
## 2. Get the dataset into Colab
 
You have two options:
 
### 1 - Upload directly to the Colab session
- In the left sidebar, open the **Files** panel (folder icon).
- Upload your dataset archive (e.g. `archive.zip`) into `/content/`.
- Set `LOCAL_ZIP` in the setup cell to match that exact path, e.g.:
```python
   LOCAL_ZIP = Path("/content/archive.zip")
```
 
### 2 - Use Google Drive (persists across sessions)
- Mount your Drive at the top of the setup cell:
```python
   from google.colab import drive
   drive.mount('/content/drive')
```
- Place the dataset (extracted folder or zip) somewhere in your Drive, e.g.
   `MyDrive/crohme2019/`.
- Point `COLAB_DATA_ROOTS` (or `LOCAL_ZIP`) at that Drive path.
## 3. Run the setup cell
 
The setup cell will:
- Search `COLAB_DATA_ROOTS` and `LOCAL_DATA_ROOT` for an existing
  `crohme2019_train.txt`.
- If not found, extract `LOCAL_ZIP` into `WORK_DIR/crohme2019_dataset`.
- Resolve `INKML_ROOT` automatically (handles both flat and nested
  `crohme2019/crohme2019/...` archive layouts).
- Print the resolved `DATA_ROOT`, `INKML_ROOT`, and `WORK_DIR`.
If the printed `INKML_ROOT` doesn't contain a `test/` folder with `.inkml`
files, inspect the extracted structure and adjust the path-resolution logic
accordingly:
 
```python
!find /content/working/crohme2019_dataset -maxdepth 4 -type d
```
 
## 4. WandB logging (optional for grading)
 
To enable experiment tracking:
 
```python
import wandb
run = wandb.init(project="crohme-ctc")
```
 
You'll be prompted for an API key
 
To skip logging entirely, set:
 
```python
run = None
```
 
The training loop checks `if run is not None` before logging, so either
choice works without further code changes.
 
## 5. Training
 
Run the cells in order:
1. Vocabulary build (`loaded_vocab`)
2. Dataset / DataLoader definitions (`create_dataloaders`, `collate_fn`)
3. Model definition (`LSTMTemporalClassifier`)
4. Training helpers (`train_one_epoch`, `evaluate_model`, `GreedyCTCDecoder`)
5. The smoke test cell (1-batch sanity check) — **run this first** to catch
   shape/dtype bugs cheaply before a full run
6. The full training run (`fit_with_early_stopping` / `fit_with_early_stopping_wer`)
While debugging, keep `max_train_batches` / `max_val_batches` small (e.g. `5`)
in `config`. Set both to `None` for a real training run.
 
## 6. Checkpoints
 
Best checkpoints are saved under `checkpoint_dir`:
- `checkpoints/` — best validation **loss**
- `checkpoints_wer/` — best validation **WER** (if using
  `fit_with_early_stopping_wer`)
To reload a checkpoint for continued training or inference:
 
```python
checkpoint = torch.load(best_checkpoint_path, map_location=device)
train_model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```


## Kaggle Notebook

https://www.kaggle.com/code/nhanngt/crohme-ctc-pytorch-project/notebook