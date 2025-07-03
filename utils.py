import os
import pickle

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            state = pickle.load(f)
        print(f"Resumed from batch {state['last_batch_index']}")
        return state["results"], state["last_batch_index"]
    else:
        return {}, -1

def save_checkpoint(path, results, last_batch_index):
    with open(path, "wb") as f:
        pickle.dump({
            "results": results,
            "last_batch_index": last_batch_index
        }, f)
    print(f"[Checkpoint] Saved batch {last_batch_index}, total items: {len(results)}")