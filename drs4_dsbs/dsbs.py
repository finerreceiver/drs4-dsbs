# standard library
from dataclasses import dataclass
from typing import Literal as L


# dependencies
from numpy import complex128, float64, int64
from xarray_dataclasses import AsDataset, Attr, Coordof, Data, Dataof


# data classes (dims)
@dataclass
class Time:
    data: Data[L["time"], L["M8[ns]"]]
    long_name: Attr[str] = "Measured time"


@dataclass
class Chan:
    data: Data[L["chan"], int64]
    long_name: Attr[str] = "Channel number"


# data classes (coords)
@dataclass
class SignalChan:
    data: Data[L["time"], int64]
    long_name: Attr[str] = "Signal channel number"


@dataclass
class SignalSB:
    data: Data[L["time"], L["U3"]]
    long_name: Attr[str] = "Signal sideband (LSB|USB)"


@dataclass
class SignalFreq:
    data: Data[L["chan"], float64]
    long_name: Attr[str] = "Signal frequency"
    units: Attr[str] = "GHz"


# data classes (vars)
@dataclass
class AutoLSB:
    data: Data[tuple[L["time"], L["chan"]], float64]
    long_name: Attr[str] = "Auto-correlation of LSB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class AutoUSB:
    data: Data[tuple[L["time"], L["chan"]], float64]
    long_name: Attr[str] = "Auto-correlation of USB"
    units: Attr[str] = "Arbitrary unit"


@dataclass
class Cross2SB:
    data: Data[tuple[L["time"], L["chan"]], complex128]
    long_name: Attr[str] = "Cross-correlation between LSB and USB"
    units: Attr[str] = "Arbitrary unit"


# data class (dataset)
@dataclass
class DSBS(AsDataset):
    """Digital sideband measurement set."""

    # dims
    time: Coordof[Time]
    chan: Coordof[Chan]
    # coords
    signal_chan: Coordof[SignalChan]
    signal_sb: Coordof[SignalSB]
    signal_freq: Coordof[SignalFreq]
    # vars
    auto_lsb: Dataof[AutoLSB]
    auto_usb: Dataof[AutoUSB]
    cross_2sb: Dataof[Cross2SB]
