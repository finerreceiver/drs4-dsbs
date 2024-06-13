__all__ = ["download", "measure"]


# standard library
from dataclasses import dataclass
from datetime import UTC, datetime
from io import StringIO
from os import getenv
from subprocess import run
from typing import Literal as L, Optional


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof


# constants
DEFAULT_INPUT_NUM = 1
DEFAULT_INTEG_TIME = 1000
DEFAULT_SIGNAL_SB = "USB"
DEFAULT_TIMEOUT = 10.0


# data classes (dims)
@dataclass
class Time:
    data: Data[L["time"], L["M8[ns]"]]
    long_name: Attr[str] = "Measured time in UTC"


@dataclass
class Chan:
    data: Data[L["chan"], np.int64]
    long_name: Attr[str] = "Channel number"


# data classes (coords)
@dataclass
class SignalChan:
    data: Data[L["time"], np.int64]
    long_name: Attr[str] = "Signal channel number"


@dataclass
class SignalSB:
    data: Data[L["time"], L["U3"]]
    long_name: Attr[str] = "Signal sideband (USB|LSB)"


@dataclass
class Freq:
    data: Data[L["chan"], np.float64]
    long_name: Attr[str] = "Measured frequency"
    units: Attr[str] = "GHz"


# data classes (vars)
@dataclass
class AutoUSB:
    data: Data[tuple[L["time"], L["chan"]], np.float64]
    long_name: Attr[str] = "Auto-correlation of USB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class AutoLSB:
    data: Data[tuple[L["time"], L["chan"]], np.float64]
    long_name: Attr[str] = "Auto-correlation of LSB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class Cross2SB:
    data: Data[tuple[L["time"], L["chan"]], np.complex128]
    long_name: Attr[str] = "Cross-correlation between USB and LSB"
    units: Attr[str] = "Arbitrary unit"


# data class (dataset)
@dataclass
class DSBS(AsDataset):
    """Digital sideband measurement set."""

    # dims
    time: Coordof[Time]
    """Measured time in UTC."""

    chan: Coordof[Chan]
    """Channel number."""

    # coords
    signal_chan: Coordof[SignalChan]
    """Signal channel number."""

    signal_sb: Coordof[SignalSB]
    """Signal sideband (USB|LSB)."""

    freq: Coordof[Freq]
    """Measured frequency (GHz)."""

    # vars
    auto_USB: Dataof[AutoUSB]
    """Auto-correlation of USB."""

    auto_LSB: Dataof[AutoLSB]
    """Auto-correlation of LSB."""

    cross_2SB: Dataof[Cross2SB]
    """Cross-correlation between USB and LSB."""

    # attrs
    input_num: Attr[L[1, 2]]
    """Input (data module) number (1|2)."""

    integ_time: Attr[L[100, 200, 500, 1000]]
    """Integration time in ms (100|200|500|1000)."""


def download(
    *,
    # for connection
    user: Optional[str] = None,
    host: Optional[str] = None,
    password: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    # for measurement
    signal_chan: int = 0,
    signal_sb: L["USB", "LSB"] = DEFAULT_SIGNAL_SB,
    input_num: L[1, 2] = DEFAULT_INPUT_NUM,
    integ_time: L[100, 200, 500, 1000] = DEFAULT_INTEG_TIME,
) -> xr.Dataset:
    """Download the latest measurement of auto/cross-correlations.

    Args:
        user: User name used to login to the DRS4.
            If not specified, environment variable ``DRS4_USER`` will be used.
        host: Host name or IP address of the DRS4.
            If not specified, environment variable ``DRS4_HOST`` will be used.
        password: Password of the login user.
            If not specified, environment variable ``DRS4_PASSWORD`` will be used.
        timeout: Timeout of the connection process in seconds.
        signal_chan: Signal channel number.
        signal_sb: Signal sideband (USB|LSB).
        input_num: Input (data module) number (1|2).
        integ_time: Integration time in ms (100|200|500|1000).

    Returns:
        Dataset of the latest measurement of auto/cross-correlations.

    """
    user = user or getenv("DRS4_USER")
    host = host or getenv("DRS4_HOST")
    password = password or getenv("DRS4_PASSWORD")

    cmd_autos = f"cat /home/{user}/DRS4/mrdsppy/output/new_pow.csv"
    cmd_cross = f"cat /home/{user}/DRS4/mrdsppy/output/new_phase.csv"

    cp_autos = run(
        f"ssh {user}@{host} -p {password} '{cmd_autos}'",
        check=True,
        shell=True,
        text=True,
        timeout=timeout,
    )
    cp_cross = run(
        f"ssh {user}@{host} -p {password} '{cmd_cross}'",
        check=True,
        shell=True,
        text=True,
        timeout=timeout,
    )

    df_autos = pd.read_csv(StringIO(cp_autos.stdout))
    df_cross = pd.read_csv(StringIO(cp_cross.stdout))

    return DSBS.new(
        time=datetime.now(UTC),
        chan=np.arange(len(df_autos)),
        signal_chan=signal_chan,
        signal_sb=signal_sb,
        freq=df_autos["freq[GHz]"],
        auto_USB=df_autos["out0"],
        auto_LSB=df_autos["out1"],
        cross_2SB=df_cross["real"] + 1j * df_cross["imag"],
        input_num=input_num,
        integ_time=integ_time,
    )


def measure(
    *,
    # for connection
    user: Optional[str] = None,
    host: Optional[str] = None,
    password: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    # for measurement
    input_num: L[1, 2] = DEFAULT_INPUT_NUM,
    integ_time: L[100, 200, 500, 1000] = DEFAULT_INTEG_TIME,
) -> None:
    """Measure auto/cross-correlations.

    Args:
        user: User name used to login to the DRS4.
            If not specified, environment variable ``DRS4_USER`` will be used.
        host: Host name or IP address of the DRS4.
            If not specified, environment variable ``DRS4_HOST`` will be used.
        password: Password of the login user.
            If not specified, environment variable ``DRS4_PASSWORD`` will be used.
        timeout: Timeout of the connection process in seconds.
        input_num: Input (data module) number (1|2).
        integ_time: Integration time in ms (100|200|500|1000).

    """
    user = user or getenv("DRS4_USER")
    host = host or getenv("DRS4_HOST")
    password = password or getenv("DRS4_PASSWORD")

    if integ_time not in [100, 200, 500, 1000]:
        raise ValueError("Integration time must be either 100|200|500|1000.")

    cmd = ";".join(
        [
            f"cd /home/{user}/DRS4/cmd",
            f"./set_intg_time.py --In {input_num} --It {integ_time // 100}",
            f"./get_corr_rslt.py --In {input_num}",
        ]
    )

    run(
        f"ssh {user}@{host} -p {password} '{cmd}'",
        check=True,
        shell=True,
        text=True,
        timeout=timeout,
    )
