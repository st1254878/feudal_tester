import logging
import os
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Handles TensorBoard metrics logging, model checkpoints directory management,
    and formatted command-line output.
    """
    def __init__(self, run_name: str, args, log_dir_base: str = "runs", checkpoint_dir_base: str = "checkpoints"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{timestamp}_{run_name}"
        self.log_dir = os.path.join(log_dir_base, self.run_name)
        self.checkpoint_dir = os.path.join(checkpoint_dir_base, self.run_name)

        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)
        self.start_time = time.time()
        self.num_episodes = 0

        # Console logging format
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        logging.info(f"Initialized Logger. Run: {self.run_name}")
        logging.info(f"TensorBoard log directory: {self.log_dir}")
        logging.info(f"Checkpoint directory: {self.checkpoint_dir}")

    def log_scalars(self, scalar_dict: dict, step: int) -> None:
        """Log training losses, intrinsic rewards, advantages, etc."""
        for key, val in scalar_dict.items():
            if val is not None:
                self.writer.add_scalar(key, float(val), step)

    def log_episode(self, info: dict, step: int) -> None:
        """
        Extracts episodic returns from environment info dictionary.
        Compatible with Gymnasium's RecordEpisodeStatistics.
        """
        if not info:
            return

        if "episode" in info:
            ep_data = info["episode"]
            if isinstance(ep_data, dict):
                rewards = ep_data.get("r")
                lengths = ep_data.get("l")
                masks = info.get("_episode")

                if rewards is not None and lengths is not None:
                    if not hasattr(rewards, "__iter__"):
                        rewards = [rewards]
                        lengths = [lengths]
                        masks = [True]

                    if masks is None:
                        masks = [True] * len(rewards)

                    for r, l, m in zip(rewards, lengths, masks):
                        if m:
                            self.num_episodes += 1
                            r_float = float(r)
                            l_int = int(l)
                            self.writer.add_scalar("episode/reward", r_float, step)
                            self.writer.add_scalar("episode/length", l_int, step)
                            elapsed_min = (time.time() - self.start_time) / 60.0

                            logging.info(
                                f"Ep #{self.num_episodes:4d} | Step: {step:8d} | "
                                f"Reward: {r_float:6.1f} | Length: {l_int:5d} | Time: {elapsed_min:5.1f}m"
                            )
            elif isinstance(ep_data, (list, tuple)):
                for item in ep_data:
                    if isinstance(item, dict):
                        r = item.get("r")
                        l = item.get("l")
                        if r is not None and l is not None:
                            if hasattr(r, "__iter__"):
                                for single_r, single_l in zip(r, l):
                                    self.num_episodes += 1
                                    r_float = float(single_r)
                                    l_int = int(single_l)
                                    self.writer.add_scalar("episode/reward", r_float, step)
                                    self.writer.add_scalar("episode/length", l_int, step)
                                    elapsed_min = (time.time() - self.start_time) / 60.0
                                    logging.info(
                                        f"Ep #{self.num_episodes:4d} | Step: {step:8d} | "
                                        f"Reward: {r_float:6.1f} | Length: {l_int:5d} | Time: {elapsed_min:5.1f}m"
                                    )
                            else:
                                self.num_episodes += 1
                                r_float = float(r)
                                l_int = int(l)
                                self.writer.add_scalar("episode/reward", r_float, step)
                                self.writer.add_scalar("episode/length", l_int, step)
                                elapsed_min = (time.time() - self.start_time) / 60.0
                                logging.info(
                                    f"Ep #{self.num_episodes:4d} | Step: {step:8d} | "
                                    f"Reward: {r_float:6.1f} | Length: {l_int:5d} | Time: {elapsed_min:5.1f}m"
                                )

    def close(self):
        self.writer.close()
