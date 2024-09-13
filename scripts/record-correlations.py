# standard library
from itertools import cycle as cycle_, islice
from logging import INFO, FileHandler, StreamHandler, basicConfig
from pathlib import Path
from time import sleep
from typing import Iterator, Sequence, TypeVar
from warnings import catch_warnings, simplefilter


# dependencies
from drs4_dsbs import download, measure, output, stop
from fire import Fire
from xarray import concat


# constants
LOG = Path(__file__).with_suffix(".log").name


# type hints
T = TypeVar("T")


def cycle(sequence: Sequence[T], n: int = 1, /) -> Iterator[T]:
    """itertools.cycle that has the maximum number of cycles."""
    return islice(cycle_(sequence), len(sequence) * n)


def main(
    signal_chans: list[int],
    output_zarr: Path,
    /,
    *,
    append: bool = False,
    repeat: int = 1,
) -> None:
    """Repeatedly record auto/cross-correlations in a Zarr file.

    Args:
        signal_chans: Channels where the CW signal enters.
        output_zarr: Path of the output Zarr file.
        append: Whether to append to an existing Zarr.
        repeat: Maximum number of the repetitions.

    Raises:
        FileExistsError: Raised if append is not allowed
            and the output Zarr file already exists.

    """
    if not append and Path(output_zarr).exists():
        raise FileExistsError(output_zarr)

    try:
        for signal_chan in cycle(signal_chans, repeat):
            # Measure the CW signal in USB
            signal_SB = "USB"

            output(
                signal_chan=signal_chan,
                signal_SB=signal_SB,
            )
            sleep(1)
            measure()
            ds_usb = download(
                signal_chan=signal_chan,
                signal_SB=signal_SB,
            )

            # Measure the CW signal in LSB
            signal_SB = "LSB"

            output(
                signal_chan=signal_chan,
                signal_SB=signal_SB,
            )
            sleep(1)
            measure()
            ds_lsb = download(
                signal_chan=signal_chan,
                signal_SB=signal_SB,
            )

            # Write/append the measurements to the output Zarr
            ds = concat([ds_usb, ds_lsb], "time")

            if not Path(output_zarr).exists():
                ds.to_zarr(output_zarr, mode="w")
            else:
                ds.to_zarr(output_zarr, mode="a", append_dim="time")
    except KeyboardInterrupt:
        pass
    finally:
        stop()


if __name__ == "__main__":
    basicConfig(
        level=INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=(StreamHandler(), FileHandler(LOG)),
    )

    with catch_warnings():
        simplefilter("ignore")
        Fire(main)
