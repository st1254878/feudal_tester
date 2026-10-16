import torch


class Storage:
    """
    Rollout trajectory buffer for FeUdal Networks.
    Stores step transitions and calculates n-step discounted returns.
    """
    def __init__(self, size: int, keys: list[str] = None):
        self.size = size
        self.keys = keys if keys is not None else []
        self.reset()

    def add(self, data: dict[str, torch.Tensor]) -> None:
        for k, v in data.items():
            if k not in self.keys:
                self.keys.append(k)
                setattr(self, k, [])
            getattr(self, k).append(v)

    def placeholder(self, extra_keys: list[str] = None) -> None:
        if extra_keys:
            for k in extra_keys:
                if k not in self.keys:
                    self.keys.append(k)
        for k in self.keys:
            v = getattr(self, k, None)
            if v is None or len(v) == 0:
                setattr(self, k, [None] * self.size)

    def reset(self) -> None:
        for key in self.keys:
            setattr(self, key, [])

    def normalize(self, keys: list[str]) -> None:
        """Normalize advantages or returns across the rollout."""
        for key in keys:
            tensors = getattr(self, key)
            stacked = torch.stack(tensors)
            mean = stacked.mean()
            std = stacked.std() + 1e-8
            normalized = (stacked - mean) / std
            setattr(self, key, [elem for elem in normalized])

    def stack(self, keys: list[str]) -> list[torch.Tensor]:
        return [torch.stack(getattr(self, k)[:self.size], dim=0) for k in keys]
