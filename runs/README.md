### 📁 Directory Structure

The `runs/` directory is automatically created during training and contains two main subdirectories:

- `Pretext_random_rot/`: Store model weights, training logs, and run logs for the pretext (pretraining) stage.
- `Downstream_random_rot/`: Contain evaluation results and logs for downstream few-shot identification tasks.

Both folders share a similar structure:
```
runs/
├── Pretext_random_rot/
│   ├── <experiment_name>/
│   │   ├── <timestamp>/                     # e.g., 0113_205528, stores complete logs and checkpoints
│   │   │   ├── LOG_INFO.log                 # Run logs
│   │   │   ├── checkpoint_step<1/2>.pth     # Temporary checkpoint at different stages
│   │   │   ├── best_model.pth               # Best checkpoint during training
│   │   │   └── final_model.pth              # Final model at the end of training
│   │   ├── best_model.pth                   # Copied from the timestamp folder, for quick access
│   │   └── final_model.pth                  # Copied from the timestamp folder
│
├── Downstream_random_rot/
│   └── <experiment_name>/
│       ├── <timestamp>/
│       │   ├── best_model.pth
│       │   └── final_model.pth
│       ├── best_model.pth
│       └── final_model.pth
├── Downstream_random_rot/
│   └── <experiment_name>/
│       ├── <xxx>Shot/                       # e.g., 5Shot
│       │   └── <timestamp>/
│       │       ├── LOG_INFO.log
│       │       └── test_result.xlsx         # Identification results 
│       └── <experiment_name>_<xxx>Shot.xlsx # Copied from the timestamp folder
```