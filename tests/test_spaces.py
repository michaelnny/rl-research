import numpy as np

from rlh_bench.spaces import Box, Discrete, MultiDiscrete, clip_to_box, flatdim


def test_box_sample_and_contains():
    space = Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    rng = np.random.default_rng(0)
    sample = space.sample(rng)
    assert sample.shape == (3,)
    assert sample.dtype == np.float32
    assert space.contains(sample)
    assert not space.contains(np.array([2.0, 0.0, 0.0], dtype=np.float32))
    assert flatdim(space) == 3


def test_discrete_and_multidiscrete():
    d = Discrete(4)
    assert d.contains(0)
    assert d.contains(3)
    assert not d.contains(4)
    md = MultiDiscrete([2, 3, 4])
    assert md.contains(np.array([1, 2, 3]))
    assert not md.contains(np.array([1, 3, 3]))
    assert flatdim(md) == 9


def test_clip_to_box():
    space = Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
    clipped = clip_to_box(space, np.array([-1.0, 2.0]))
    assert np.allclose(clipped, np.array([0.0, 1.0], dtype=np.float32))
