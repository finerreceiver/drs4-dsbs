__all__ = ["download", "measure", "output"]


# standard library
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from os import getenv
from subprocess import run
from typing import Literal as L, Optional


# dependencies
import numpy as np
import pandas as pd
import xarray as xr
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof
from .scpi import send_commands


# constants
FREQ_INTERVAL = 0.02  # GHz
DEFAULT_INPUT_NUM = 1
DEFAULT_INTEG_TIME = 1000
DEFAULT_LO_FREQ = 90.0  # GHz
DEFAULT_LO_MUX = 5
DEFAULT_SIGNAL_CHAN = 0
DEFAULT_SIGNAL_SB = "USB"
DEFAULT_TIMEOUT = 10.0  # s


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
    """Signal channel number (0-1023)."""

    signal_SB: Coordof[SignalSB]
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
    signal_SB: L["USB", "LSB"] = DEFAULT_SIGNAL_SB,
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
        signal_chan: Signal channel number (0-1023).
        signal_SB: Signal sideband (USB|LSB).
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
        time=datetime.now(timezone.utc),
        chan=np.arange(len(df_autos)),
        signal_chan=signal_chan,
        signal_SB=signal_SB,
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


def output(
    *,
    # for connection,
    host: Optional[str] = None,
    port: Optional[int] = None,
    timeout: float = DEFAULT_TIMEOUT,
    # for frequency
    signal_chan: int = DEFAULT_SIGNAL_CHAN,
    signal_SB: L["USB", "LSB"] = DEFAULT_SIGNAL_SB,
    LO_freq: float = DEFAULT_LO_FREQ,
    LO_mux: int = DEFAULT_LO_MUX,
) -> None:
    """Output CW signal by setting SG frequency and turning output on.

    Args:
        host: Host name or IP address of the SG (Keysight 8257D).
            If not specified, environment variable ``SG_HOST`` will be used.
        port: Port number of the SG (Keysight 8257D).
            If not specified, environment variable ``SG_PORT`` will be used.
        timeout: Timeout of the connection process in seconds.
        signal_chan: Signal channel number (0-1023).
        signal_SB: Signal sideband (USB|LSB).
        LO_freq: LO frequency in GHz.
        LO_mux: LO multiplication factor.

    """
    host = host or getenv("SG_HOST")
    port = port or getenv("SG_PORT")

    if signal_SB == "USB":
        SG_freq = (LO_freq + FREQ_INTERVAL * signal_chan) / LO_mux
    elif signal_SB == "LSB":
        SG_freq = (LO_freq - FREQ_INTERVAL * signal_chan) / LO_mux
    else:
        raise ValueError("Signal sideband must be either USB|LSB.")

    send_commands(
        [
            "OUTP OFF",
            "FREQ:MODE CW",
            f"FREQ:CW {SG_freq}GHZ",
            "OUTP ON",
        ],
        host=host,
        port=int(port),
        timeout=timeout,
    )
