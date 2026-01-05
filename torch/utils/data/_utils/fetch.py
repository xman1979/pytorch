# mypy: allow-untyped-defs
r"""Contains definitions of the methods used by the _BaseDataLoaderIter to fetch data from an iterable-style or map-style dataset.

This logic is shared in both single- and multi-processing data loading.
"""

import time


def _get_training_metrics_collector():
    """Get the training metrics collector if available and enabled."""
    try:
        from torch.training_metrics.collector import get_collector, is_enabled
        if is_enabled():
            return get_collector()
    except ImportError:
        pass
    return None


class _BaseDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        self.dataset = dataset
        self.auto_collation = auto_collation
        self.collate_fn = collate_fn
        self.drop_last = drop_last

        # Check if training metrics are enabled
        self._training_metrics_collector = _get_training_metrics_collector()

    def fetch(self, possibly_batched_index):
        raise NotImplementedError

    def _timed_collate(self, data):
        """Collate data with optional timing for training metrics."""
        if self._training_metrics_collector is not None:
            start_time = time.perf_counter()
            result = self.collate_fn(data)
            elapsed = time.perf_counter() - start_time
            self._training_metrics_collector.record_data_preprocessing_time(elapsed)
            return result
        else:
            return self.collate_fn(data)


class _IterableDatasetFetcher(_BaseDatasetFetcher):
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        super().__init__(dataset, auto_collation, collate_fn, drop_last)
        self.dataset_iter = iter(dataset)
        self.ended = False

    def fetch(self, possibly_batched_index):
        if self.ended:
            raise StopIteration

        if self.auto_collation:
            data = []
            for _ in possibly_batched_index:
                try:
                    data.append(next(self.dataset_iter))
                except StopIteration:
                    self.ended = True
                    break
            if len(data) == 0 or (
                self.drop_last and len(data) < len(possibly_batched_index)
            ):
                raise StopIteration
        else:
            data = next(self.dataset_iter)
        return self._timed_collate(data)


class _MapDatasetFetcher(_BaseDatasetFetcher):
    def fetch(self, possibly_batched_index):
        if self.auto_collation:
            if hasattr(self.dataset, "__getitems__") and self.dataset.__getitems__:
                data = self.dataset.__getitems__(possibly_batched_index)
            else:
                data = [self.dataset[idx] for idx in possibly_batched_index]
        else:
            data = self.dataset[possibly_batched_index]
        return self._timed_collate(data)
