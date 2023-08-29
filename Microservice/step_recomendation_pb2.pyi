from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Step1Request(_message.Message):
    __slots__ = ["houseArea", "siteArea", "floorCount", "region", "budgetFloor", "budgetCeil"]
    HOUSEAREA_FIELD_NUMBER: _ClassVar[int]
    SITEAREA_FIELD_NUMBER: _ClassVar[int]
    FLOORCOUNT_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    BUDGETFLOOR_FIELD_NUMBER: _ClassVar[int]
    BUDGETCEIL_FIELD_NUMBER: _ClassVar[int]
    houseArea: int
    siteArea: int
    floorCount: int
    region: str
    budgetFloor: int
    budgetCeil: int
    def __init__(self, houseArea: _Optional[int] = ..., siteArea: _Optional[int] = ..., floorCount: _Optional[int] = ..., region: _Optional[str] = ..., budgetFloor: _Optional[int] = ..., budgetCeil: _Optional[int] = ...) -> None: ...

class Step1Response(_message.Message):
    __slots__ = ["message"]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    message: str
    def __init__(self, message: _Optional[str] = ...) -> None: ...
