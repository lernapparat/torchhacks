import pytest
import tempfile
import pathlib
import torchhacks
import torch


def test_lazy_load_basic():
    with tempfile.TemporaryDirectory() as tmpdirname:
        m = torch.nn.Linear(5, 3)
        path = pathlib.Path(tmpdirname)
        fn = str(path / "test.pt")
        torch.save(m.state_dict(), fn)
        sd_lazy = torchhacks.lazy_load(fn)
        assert "NotYetLoadedTensor" in str(next(iter(sd_lazy.values())))
        m2 = torch.nn.Linear(5, 3)
        m2.load_state_dict(sd_lazy)

        x = torch.randn(2, 5)
        actual = m2(x)
        expected = m(x)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    pytest.main([__file__])
