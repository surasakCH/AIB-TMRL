# standard library imports
import os
import pickle
import zlib
from abc import ABC, abstractmethod
from pathlib import Path
from random import randint, random
import logging

# third-party imports
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler

# local imports
from tmrl.util import collate


__docformat__ = "google"


def check_samples_crc(original_po, original_a, original_o, original_r, original_d, original_t, rebuilt_po, rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t):
    assert original_po is None or str(original_po) == str(rebuilt_po), f"previous observations don't match:\noriginal:\n{original_po}\n!= rebuilt:\n{rebuilt_po}"
    assert str(original_a) == str(rebuilt_a), f"actions don't match:\noriginal:\n{original_a}\n!= rebuilt:\n{rebuilt_a}"
    assert str(original_o) == str(rebuilt_o), f"observations don't match:\noriginal:\n{original_o}\n!= rebuilt:\n{rebuilt_o}"
    assert str(original_r) == str(rebuilt_r), f"rewards don't match:\noriginal:\n{original_r}\n!= rebuilt:\n{rebuilt_r}"
    assert str(original_d) == str(rebuilt_d), f"terminated don't match:\noriginal:\n{original_d}\n!= rebuilt:\n{rebuilt_d}"
    assert str(original_t) == str(rebuilt_t), f"truncated don't match:\noriginal:\n{original_t}\n!= rebuilt:\n{rebuilt_t}"
    original_crc = zlib.crc32(str.encode(str((original_a, original_o, original_r, original_d, original_t))))
    crc = zlib.crc32(str.encode(str((rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d, rebuilt_t))))
    assert crc == original_crc, f"CRC failed: new crc:{crc} != old crc:{original_crc}.\nEither the custom pipeline is corrupted, or crc_debug is False in the rollout worker.\noriginal sample:\n{(original_a, original_o, original_r, original_d)}\n!= rebuilt sample:\n{(rebuilt_a, rebuilt_o, rebuilt_r, rebuilt_d)}"
    print("DEBUG: CRC check passed.")


class MemoryBatchSampler(Sampler):
    """
    Iterator over nb_steps randomly sampled batches of size batch_size
    """
    def __init__(self, data_source, nb_steps, batch_size):
        super().__init__(data_source)
        self._dataset = data_source
        self._nb_steps = nb_steps
        self._batch_size = batch_size

    def __len__(self):
        return self._nb_steps

    def __iter__(self):
        i = 0
        while i < self._nb_steps:
            i += 1
            yield (int(len(self._dataset) * random()) - 1 for _ in range(self._batch_size))  # faster than randint


class MemoryDataloading(ABC):  # FIXME: should be an instance of Dataset but partial doesn't work with Dataset
    """
    Interface for a simple replay buffer.

    This class supports sampling and collating simple batches of prev_obs, new_act, new_obs, rew, terminated, truncated.

    In case you need more advanced replay buffers, you can store whatever you need in the `info` dict and collate
    batches manually in your TrainingAgent.

    .. note::
       When overriding `__init__`, don't forget to call `super().__init__` in the subclass.
       Your `__init__` method needs to take at least all the arguments of the superclass.
    """
    def __init__(self,
                 device,
                 nb_steps,
                 sample_preprocessor: callable = None,
                 memory_size=1000000,
                 batch_size=256,
                 dataset_path="",
                 crc_debug=False,
                 use_dataloader=False,
                 num_workers=0,
                 pin_memory=False):
        """
        Args:
            device (str): output tensors will be collated to this device
            nb_steps (int): number of steps per round
            sample_preprocessor (callable): can be used for data augmentation
            memory_size (int): size of the circular buffer
            batch_size (int): batch size of the output tensors
            dataset_path (str): an offline dataset may be provided here to initialize the memory
            crc_debug (bool): False usually, True when using CRC debugging of the pipeline
            use_dataloader (bool): Not yet supported
            num_workers (int): Not yet supported
            pin_memory: Not yet supported
        """
        self.nb_steps = nb_steps
        self.use_dataloader = use_dataloader
        self.device = device
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.sample_preprocessor = sample_preprocessor
        self.crc_debug = crc_debug

        # These stats are here because they reach the trainer along with the buffer:
        self.stat_test_return = 0.0
        self.stat_train_return = 0.0
        self.stat_test_steps = 0
        self.stat_train_steps = 0

        # init memory
        self.path = Path(dataset_path)
        logging.debug(f"MemoryDataloading self.path:{self.path}")
        if os.path.isfile(self.path / 'data.pkl'):
            with open(self.path / 'data.pkl', 'rb') as f:
                self.data = list(pickle.load(f))
        else:
            logging.info("no data found, initializing empty replay memory")
            self.data = []

        if len(self) > self.memory_size:
            # TODO: crop to memory_size
            logging.warning(f"the dataset length ({len(self)}) is longer than memory_size ({self.memory_size})")

        # init dataloader
        self._batch_sampler = MemoryBatchSampler(data_source=self, nb_steps=nb_steps, batch_size=batch_size)
        self._dataloader = DataLoader(dataset=self, batch_sampler=self._batch_sampler, num_workers=num_workers, pin_memory=pin_memory)

    def __iter__(self):
        if not self.use_dataloader:
            for _ in range(self.nb_steps):
                yield self.sample()
        else:
            for batch in self._dataloader:
                yield batch  # TODO: move this to self.device !!!

    @abstractmethod
    def append_buffer(self, buffer):
        """
        Must append a Buffer object to the memory.

        Args:
            buffer (tmrl.networking.Buffer): the buffer of samples to append.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """
        Must return the length of the memory.

        Returns:
            length (int): the maximum `item` argument of `get_transition`

        """
        raise NotImplementedError

    @abstractmethod
    def get_transition(self, item):
        """
        Must return a transition.

        `info` is required in each sample for CRC debugging. The 'crc' key is what is important when using this feature.

        Args:
            item (int): the index where to sample

        Returns:
            sample (Tuple): (prev_obs, prev_act, rew, obs, terminated, truncated, info)
        """
        raise NotImplementedError

    def append(self, buffer):
        if len(buffer) > 0:
            self.stat_train_return = buffer.stat_train_return
            self.stat_test_return = buffer.stat_test_return
            self.stat_train_steps = buffer.stat_train_steps
            self.stat_test_steps = buffer.stat_test_steps
            self.append_buffer(buffer)

    def __getitem__(self, item):
        prev_obs, new_act, rew, new_obs, terminated, truncated, info = self.get_transition(item)
        if self.crc_debug:
            po, a, o, r, d, t = info['crc_sample']
            check_samples_crc(po, a, o, r, d, t, prev_obs, new_act, new_obs, rew, terminated, truncated)
        if self.sample_preprocessor is not None:
            prev_obs, new_act, rew, new_obs, terminated, truncated = self.sample_preprocessor(prev_obs, new_act, rew, new_obs, terminated, truncated)
        terminated = np.float32(terminated)  # we don't want bool tensors
        truncated = np.float32(truncated)  # we don't want bool tensors
        return prev_obs, new_act, rew, new_obs, terminated, truncated

    def sample_indices(self):
        return (randint(0, len(self) - 1) for _ in range(self.batch_size))

    def sample(self, indices=None):
        indices = self.sample_indices() if indices is None else indices
        batch = [self[idx] for idx in indices]
        batch = collate(batch, self.device)
        return batch


def load_and_print_pickle_file(path=r"C:\Users\Yann\Desktop\git\tmrl\data\data.pkl"):  # r"D:\data2020"
    import pickle
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"nb samples: {len(data[0])}")
    for i, d in enumerate(data):
        print(f"[{i}][0]: {d[0]}")
    print("full data:")
    for i, d in enumerate(data):
        print(f"[{i}]: {d}")


if __name__ == "__main__":
    load_and_print_pickle_file()
