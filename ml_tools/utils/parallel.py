from typing import Optional
from contextlib import contextmanager
import ray

@contextmanager
def ray_context(num_cpus: Optional[int] = None):
    """Context manager that initializes and shuts down Ray if needed.

    This ensures that Ray is only started if it hasn't already been initialized,
    and will shut it down afterward only if this context manager started it.

    Parameters
    ----------
    num_cpus : Optional[int]
        Number of CPUs to allocate when initializing Ray. If None, Ray will choose automatically.
    """
    ray_started = False
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
        ray_started = True
    try:
        yield
    finally:
        if ray_started:
            ray.shutdown()
