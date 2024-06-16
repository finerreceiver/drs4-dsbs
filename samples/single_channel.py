# standard library
from logging import INFO, FileHandler, StreamHandler, basicConfig
from sys import argv
from time import sleep


# dependencies
import xarray as xr
from drs4_dsbs import download, measure, output, stop


def main():
    basicConfig(
        level=INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=(StreamHandler(), FileHandler("single_channel.log")),
    )

    try:
        signal_chan = int(argv[1])
        output_zarr = str(argv[2])

        # Measure CW signal in USB
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

        # Measure CW signal in LSB
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

        # Save measurements in netCDF
        ds = xr.concat([ds_usb, ds_lsb], "time")
        ds.to_zarr(output_zarr, mode="w")
    finally:
        stop()


if __name__ == "__main__":
    main()
