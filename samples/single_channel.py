# standard library
from sys import argv


# dependencies
import xarray as xr
from drs4_dsbs import download, measure, output


def main():
    signal_chan, nc_file, *_ = argv[1:]

    # Measure CW signal in USB
    signal_SB = "USB"

    output(
        signal_chan=signal_chan,
        signal_SB=signal_SB,
    )
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
    measure()
    ds_lsb = download(
        signal_chan=signal_chan,
        signal_SB=signal_SB,
    )

    # Save measurements in netCDF
    xr.concat([ds_usb, ds_lsb], "time").to_netcdf(nc_file)


if __name__ == "__main__":
    main()
