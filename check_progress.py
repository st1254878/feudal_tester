import glob, time
from tensorboard.backend.event_processing import event_accumulator

log_files = glob.glob('runs/20261015_002245_montezuma_800m/events.out.tfevents.*')
if log_files:
    ea = event_accumulator.EventAccumulator(log_files[0])
    ea.Reload()
    
    if 'episode/reward' in ea.Tags()['scalars']:
        rewards = ea.Scalars('episode/reward')
        print(f'Total episodes logged: {len(rewards)}')
        
        last_step = rewards[-1].step
        last_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rewards[-1].wall_time))
        first_step = rewards[0].step
        first_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(rewards[0].wall_time))
        
        dt_seconds = rewards[-1].wall_time - rewards[0].wall_time
        d_steps = last_step - first_step
        sps = d_steps / dt_seconds if dt_seconds > 0 else 0
        
        print(f'First log: Step {first_step} at {first_time}')
        print(f'Last log:  Step {last_step} at {last_time}')
        print(f'Average speed: {sps:.1f} steps/second ({sps * 3600 / 1e6:.2f}M steps/hour)')
        
        max_r = max(ev.value for ev in rewards)
        print(f'Max episode reward: {max_r}')
        non_zero = [ev for ev in rewards if ev.value > 0]
        print(f'Episodes with reward > 0: {len(non_zero)}')
        if non_zero:
            print('Sample non-zero episodes:')
            for ev in non_zero[-5:]:
                t_str = time.strftime('%H:%M:%S', time.localtime(ev.wall_time))
                print(f'  Step {ev.step:8d} | Reward: {ev.value:5.1f} | Time: {t_str}')
