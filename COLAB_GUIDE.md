# Google Colab Training Guide

## Problem: Tab Closure Terminates Training

Google Colab will terminate your runtime if:
- You close the browser tab
- The session times out (even with Pro+)
- Your internet connection drops

## Solutions

### Option 1: Keep Tab Open + Keep-Alive Script (Recommended)

**Step 1:** Run the keep-alive script first:
```python
# In a Colab cell
!python colab_keepalive.py
```

**Step 2:** Start training:
```bash
!./setup_and_run.sh all -y
```

**Step 3:** Keep the tab open throughout training

**Pros:**
- Simple, no code changes
- Works with existing setup

**Cons:**
- Must keep tab open
- Vulnerable to network issues

---

### Option 2: Periodic Auto-Save to GitHub (Best for Long Training)

The current setup already saves the best model during training. To add periodic commits:

**During Training:**
1. Training saves best model to `save/model.pth` automatically
2. After training completes, run:
   ```bash
   !./setup_and_run.sh save -y
   ```
3. This commits and pushes to GitHub

**Manual Checkpoint Save:**
If you need to save mid-training, interrupt the training (Runtime → Interrupt) and run:
```bash
!./setup_and_run.sh save -y
```
Then restart training if needed.

---

### Option 3: Use Colab Background Execution (Experimental)

⚠️ This feature is experimental and may not work reliably.

```python
# Try to enable background execution
from google.colab import runtime
runtime.unassign()
```

---

### Option 4: Split Training into Checkpoints

Modify training to save every N epochs:

**In `codes/misc/train.py`, change line 81:**
```python
# Save every 10 epochs (in addition to best model)
if epoch % 10 == 0:
    save_manager.save_model(epoch, self.model.state_dict(), val_metric)
    save_manager.write_log(f'Checkpoint saved at Epoch {epoch}', 4)
```

Then periodically commit:
```bash
!./setup_and_run.sh save -y
```

---

## Recommended Workflow for Long Training

```python
# Cell 1: Setup keep-alive
!python colab_keepalive.py

# Cell 2: Clone and setup (only needs to run once)
!git clone https://github.com/YOUR_USERNAME/keybot.git
%cd keybot
!./setup_and_run.sh setup -y

# Cell 3: Download and prepare data (only once)
# Manually download AASCE dataset to codes/preprocess_data/AASCE_rawdata/
!./setup_and_run.sh data -y

# Cell 4: Train (can restart from here if interrupted)
!./setup_and_run.sh train -y

# Cell 5: Evaluate
!./setup_and_run.sh eval -y

# Cell 6: Save and push to GitHub
!./setup_and_run.sh save -y
```

**Benefits:**
- Each step is separate
- Can restart from any cell if disconnected
- Models are saved after each major step

---

## Best Practices

1. **Use Colab Pro/Pro+** for longer runtimes (up to 24 hours)
2. **Train in chunks** if possible (early stopping, smaller epochs)
3. **Commit frequently** - run `./setup_and_run.sh save -y` periodically
4. **Monitor training** - check logs regularly
5. **Use GPU wisely** - Colab limits GPU hours per month

---

## If Session Disconnects

1. **Check if model was saved:**
   ```bash
   !ls -lh save/AASCE_interactive_keypoint_estimation/
   ```

2. **Commit last checkpoint:**
   ```bash
   !./setup_and_run.sh save -y
   ```

3. **To resume training from checkpoint:**
   The training script automatically loads the best model if it exists.
   Just run training again:
   ```bash
   !./setup_and_run.sh train -y
   ```

---

## Monitoring Training

**View logs in real-time:**
```bash
!tail -f save/AASCE_interactive_keypoint_estimation/log.txt
```

**Check current model:**
```bash
!./setup_and_run.sh results
```

**View in TensorBoard:**
```bash
%load_ext tensorboard
%tensorboard --logdir save/AASCE_interactive_keypoint_estimation/
```

