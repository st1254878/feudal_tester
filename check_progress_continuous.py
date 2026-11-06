import glob
import os
import time
from tensorboard.backend.event_processing import event_accumulator

def check_run(run_dir):
    run_name = os.path.basename(run_dir)
    event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
    if not event_files:
        return

    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    print(f"\n{'='*70}")
    print(f"Run: {run_name}")
    print(f"{'='*70}")

    if "eval/mean_reward" in tags:
        eval_events = ea.Scalars("eval/mean_reward")
        best_eval = max(e.value for e in eval_events)
        latest_eval = eval_events[-1].value
        latest_step = eval_events[-1].step
        last_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(eval_events[-1].wall_time))
        print(f"Eval Checks: {len(eval_events)} | Latest Step: {latest_step:,} ({last_time})")
        print(f"Latest Deterministic Eval: {latest_eval:.2f} | Peak Best: {best_eval:.2f}")

    if "train/lr_worker" in tags:
        lr_events = ea.Scalars("train/lr_worker")
        print(f"Current LR: {lr_events[-1].value:.6f}")

    if "loss/total" in tags:
        loss_events = ea.Scalars("loss/total")
        first_step = loss_events[0].step
        last_step = loss_events[-1].step
        dt = loss_events[-1].wall_time - loss_events[0].wall_time
        sps = (last_step - first_step) / dt if dt > 0 else 0
        print(f"Training Speed: {sps:.1f} steps/sec ({sps * 3600 / 1e6:.2f}M steps/hour)")

def main():
    runs = sorted(glob.glob("runs_continuous/*"))
    if not runs:
        print("No continuous runs found in runs_continuous/")
        return
    for r in runs:
        check_run(r)

if __name__ == "__main__":
    main()
