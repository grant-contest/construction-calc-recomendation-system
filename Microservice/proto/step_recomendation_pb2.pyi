from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Step0(_message.Message):
    __slots__ = ["houseArea", "siteArea", "floorCount", "region", "purpose", "budgetFloor", "budgetCeil", "houseAreaPrice", "siteAreaPrice", "floorCountPrice", "regionPrice", "purposePrice", "budgetFloorPrice", "budgetCeilPrice"]
    HOUSEAREA_FIELD_NUMBER: _ClassVar[int]
    SITEAREA_FIELD_NUMBER: _ClassVar[int]
    FLOORCOUNT_FIELD_NUMBER: _ClassVar[int]
    REGION_FIELD_NUMBER: _ClassVar[int]
    PURPOSE_FIELD_NUMBER: _ClassVar[int]
    BUDGETFLOOR_FIELD_NUMBER: _ClassVar[int]
    BUDGETCEIL_FIELD_NUMBER: _ClassVar[int]
    HOUSEAREAPRICE_FIELD_NUMBER: _ClassVar[int]
    SITEAREAPRICE_FIELD_NUMBER: _ClassVar[int]
    FLOORCOUNTPRICE_FIELD_NUMBER: _ClassVar[int]
    REGIONPRICE_FIELD_NUMBER: _ClassVar[int]
    PURPOSEPRICE_FIELD_NUMBER: _ClassVar[int]
    BUDGETFLOORPRICE_FIELD_NUMBER: _ClassVar[int]
    BUDGETCEILPRICE_FIELD_NUMBER: _ClassVar[int]
    houseArea: int
    siteArea: int
    floorCount: int
    region: str
    purpose: str
    budgetFloor: int
    budgetCeil: int
    houseAreaPrice: int
    siteAreaPrice: int
    floorCountPrice: int
    regionPrice: int
    purposePrice: int
    budgetFloorPrice: int
    budgetCeilPrice: int
    def __init__(self, houseArea: _Optional[int] = ..., siteArea: _Optional[int] = ..., floorCount: _Optional[int] = ..., region: _Optional[str] = ..., purpose: _Optional[str] = ..., budgetFloor: _Optional[int] = ..., budgetCeil: _Optional[int] = ..., houseAreaPrice: _Optional[int] = ..., siteAreaPrice: _Optional[int] = ..., floorCountPrice: _Optional[int] = ..., regionPrice: _Optional[int] = ..., purposePrice: _Optional[int] = ..., budgetFloorPrice: _Optional[int] = ..., budgetCeilPrice: _Optional[int] = ...) -> None: ...

class SitePreparation(_message.Message):
    __slots__ = ["siteChoosing", "geologicalWorks", "geodeticalWorks", "cuttingBushesAndSmallForests", "clearingTheSiteOfDebris"]
    SITECHOOSING_FIELD_NUMBER: _ClassVar[int]
    GEOLOGICALWORKS_FIELD_NUMBER: _ClassVar[int]
    GEODETICALWORKS_FIELD_NUMBER: _ClassVar[int]
    CUTTINGBUSHESANDSMALLFORESTS_FIELD_NUMBER: _ClassVar[int]
    CLEARINGTHESITEOFDEBRIS_FIELD_NUMBER: _ClassVar[int]
    siteChoosing: bool
    geologicalWorks: bool
    geodeticalWorks: bool
    cuttingBushesAndSmallForests: bool
    clearingTheSiteOfDebris: bool
    def __init__(self, siteChoosing: bool = ..., geologicalWorks: bool = ..., geodeticalWorks: bool = ..., cuttingBushesAndSmallForests: bool = ..., clearingTheSiteOfDebris: bool = ...) -> None: ...

class SiteWorks(_message.Message):
    __slots__ = ["cameras", "temporaryFence"]
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    TEMPORARYFENCE_FIELD_NUMBER: _ClassVar[int]
    cameras: bool
    temporaryFence: bool
    def __init__(self, cameras: bool = ..., temporaryFence: bool = ...) -> None: ...

class HouseDesignAndProject(_message.Message):
    __slots__ = ["homeProject", "designProject"]
    HOMEPROJECT_FIELD_NUMBER: _ClassVar[int]
    DESIGNPROJECT_FIELD_NUMBER: _ClassVar[int]
    homeProject: bool
    designProject: bool
    def __init__(self, homeProject: bool = ..., designProject: bool = ...) -> None: ...

class Step1(_message.Message):
    __slots__ = ["sitePreparation", "siteWorks", "houseDesignAndProject"]
    SITEPREPARATION_FIELD_NUMBER: _ClassVar[int]
    SITEWORKS_FIELD_NUMBER: _ClassVar[int]
    HOUSEDESIGNANDPROJECT_FIELD_NUMBER: _ClassVar[int]
    sitePreparation: SitePreparation
    siteWorks: SiteWorks
    houseDesignAndProject: HouseDesignAndProject
    def __init__(self, sitePreparation: _Optional[_Union[SitePreparation, _Mapping]] = ..., siteWorks: _Optional[_Union[SiteWorks, _Mapping]] = ..., houseDesignAndProject: _Optional[_Union[HouseDesignAndProject, _Mapping]] = ...) -> None: ...

class Step2(_message.Message):
    __slots__ = ["foundationType"]
    FOUNDATIONTYPE_FIELD_NUMBER: _ClassVar[int]
    foundationType: str
    def __init__(self, foundationType: _Optional[str] = ...) -> None: ...

class Step3(_message.Message):
    __slots__ = ["wallsMaterial"]
    WALLSMATERIAL_FIELD_NUMBER: _ClassVar[int]
    wallsMaterial: str
    def __init__(self, wallsMaterial: _Optional[str] = ...) -> None: ...

class Step4(_message.Message):
    __slots__ = ["slopesNumber", "roofType"]
    SLOPESNUMBER_FIELD_NUMBER: _ClassVar[int]
    ROOFTYPE_FIELD_NUMBER: _ClassVar[int]
    slopesNumber: int
    roofType: str
    def __init__(self, slopesNumber: _Optional[int] = ..., roofType: _Optional[str] = ...) -> None: ...

class Step5(_message.Message):
    __slots__ = ["facadeTechnology"]
    FACADETECHNOLOGY_FIELD_NUMBER: _ClassVar[int]
    facadeTechnology: str
    def __init__(self, facadeTechnology: _Optional[str] = ...) -> None: ...

class Step6(_message.Message):
    __slots__ = ["windowMaterial", "windowType", "doorMaterial"]
    WINDOWMATERIAL_FIELD_NUMBER: _ClassVar[int]
    WINDOWTYPE_FIELD_NUMBER: _ClassVar[int]
    DOORMATERIAL_FIELD_NUMBER: _ClassVar[int]
    windowMaterial: str
    windowType: str
    doorMaterial: str
    def __init__(self, windowMaterial: _Optional[str] = ..., windowType: _Optional[str] = ..., doorMaterial: _Optional[str] = ...) -> None: ...

class Electrician(_message.Message):
    __slots__ = ["plasticBoxesUpTo40mmWide", "layingAThreeToFive", "cableLaying", "installationOfTwoKey", "installationOfSingleKey", "recessedTypeSocketDevice", "installationOfPendant", "chandeliersAndPendants"]
    PLASTICBOXESUPTO40MMWIDE_FIELD_NUMBER: _ClassVar[int]
    LAYINGATHREETOFIVE_FIELD_NUMBER: _ClassVar[int]
    CABLELAYING_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFTWOKEY_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFSINGLEKEY_FIELD_NUMBER: _ClassVar[int]
    RECESSEDTYPESOCKETDEVICE_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFPENDANT_FIELD_NUMBER: _ClassVar[int]
    CHANDELIERSANDPENDANTS_FIELD_NUMBER: _ClassVar[int]
    plasticBoxesUpTo40mmWide: bool
    layingAThreeToFive: bool
    cableLaying: bool
    installationOfTwoKey: bool
    installationOfSingleKey: bool
    recessedTypeSocketDevice: bool
    installationOfPendant: bool
    chandeliersAndPendants: bool
    def __init__(self, plasticBoxesUpTo40mmWide: bool = ..., layingAThreeToFive: bool = ..., cableLaying: bool = ..., installationOfTwoKey: bool = ..., installationOfSingleKey: bool = ..., recessedTypeSocketDevice: bool = ..., installationOfPendant: bool = ..., chandeliersAndPendants: bool = ...) -> None: ...

class WaterSupply(_message.Message):
    __slots__ = ["layingOfInternalWaterSupplyPipelines", "installationOfBathtubs", "installationOfSingle", "installationOfMixers"]
    LAYINGOFINTERNALWATERSUPPLYPIPELINES_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFBATHTUBS_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFSINGLE_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFMIXERS_FIELD_NUMBER: _ClassVar[int]
    layingOfInternalWaterSupplyPipelines: bool
    installationOfBathtubs: bool
    installationOfSingle: bool
    installationOfMixers: bool
    def __init__(self, layingOfInternalWaterSupplyPipelines: bool = ..., installationOfBathtubs: bool = ..., installationOfSingle: bool = ..., installationOfMixers: bool = ...) -> None: ...

class Sewerage(_message.Message):
    __slots__ = ["installationOfToilet", "layingOfSewerage50mm", "layingOfSewerage110mm"]
    INSTALLATIONOFTOILET_FIELD_NUMBER: _ClassVar[int]
    LAYINGOFSEWERAGE50MM_FIELD_NUMBER: _ClassVar[int]
    LAYINGOFSEWERAGE110MM_FIELD_NUMBER: _ClassVar[int]
    installationOfToilet: bool
    layingOfSewerage50mm: bool
    layingOfSewerage110mm: bool
    def __init__(self, installationOfToilet: bool = ..., layingOfSewerage50mm: bool = ..., layingOfSewerage110mm: bool = ...) -> None: ...

class Heating(_message.Message):
    __slots__ = ["assemblyOfAWaterSupply", "layingOfInternalHeatingPipelines", "installationOfWindowFixtures"]
    ASSEMBLYOFAWATERSUPPLY_FIELD_NUMBER: _ClassVar[int]
    LAYINGOFINTERNALHEATINGPIPELINES_FIELD_NUMBER: _ClassVar[int]
    INSTALLATIONOFWINDOWFIXTURES_FIELD_NUMBER: _ClassVar[int]
    assemblyOfAWaterSupply: bool
    layingOfInternalHeatingPipelines: bool
    installationOfWindowFixtures: bool
    def __init__(self, assemblyOfAWaterSupply: bool = ..., layingOfInternalHeatingPipelines: bool = ..., installationOfWindowFixtures: bool = ...) -> None: ...

class Ventilation(_message.Message):
    __slots__ = ["installationOfSplitSystems", "cablingOnABrickWall"]
    INSTALLATIONOFSPLITSYSTEMS_FIELD_NUMBER: _ClassVar[int]
    CABLINGONABRICKWALL_FIELD_NUMBER: _ClassVar[int]
    installationOfSplitSystems: bool
    cablingOnABrickWall: bool
    def __init__(self, installationOfSplitSystems: bool = ..., cablingOnABrickWall: bool = ...) -> None: ...

class Step7(_message.Message):
    __slots__ = ["electrician", "waterSupply", "sewerage", "heating", "ventilation"]
    ELECTRICIAN_FIELD_NUMBER: _ClassVar[int]
    WATERSUPPLY_FIELD_NUMBER: _ClassVar[int]
    SEWERAGE_FIELD_NUMBER: _ClassVar[int]
    HEATING_FIELD_NUMBER: _ClassVar[int]
    VENTILATION_FIELD_NUMBER: _ClassVar[int]
    electrician: Electrician
    waterSupply: WaterSupply
    sewerage: Sewerage
    heating: Heating
    ventilation: Ventilation
    def __init__(self, electrician: _Optional[_Union[Electrician, _Mapping]] = ..., waterSupply: _Optional[_Union[WaterSupply, _Mapping]] = ..., sewerage: _Optional[_Union[Sewerage, _Mapping]] = ..., heating: _Optional[_Union[Heating, _Mapping]] = ..., ventilation: _Optional[_Union[Ventilation, _Mapping]] = ...) -> None: ...

class Step8(_message.Message):
    __slots__ = ["warmFloor", "ladderMaterial"]
    WARMFLOOR_FIELD_NUMBER: _ClassVar[int]
    LADDERMATERIAL_FIELD_NUMBER: _ClassVar[int]
    warmFloor: bool
    ladderMaterial: str
    def __init__(self, warmFloor: bool = ..., ladderMaterial: _Optional[str] = ...) -> None: ...

class Step9(_message.Message):
    __slots__ = ["wallDecoration", "floorCovering", "ceilCovering"]
    WALLDECORATION_FIELD_NUMBER: _ClassVar[int]
    FLOORCOVERING_FIELD_NUMBER: _ClassVar[int]
    CEILCOVERING_FIELD_NUMBER: _ClassVar[int]
    wallDecoration: str
    floorCovering: str
    ceilCovering: str
    def __init__(self, wallDecoration: _Optional[str] = ..., floorCovering: _Optional[str] = ..., ceilCovering: _Optional[str] = ...) -> None: ...

class Step1Request(_message.Message):
    __slots__ = ["step0"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ...) -> None: ...

class Step2Request(_message.Message):
    __slots__ = ["step0", "step1"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ...) -> None: ...

class Step3Request(_message.Message):
    __slots__ = ["step0", "step1", "step2"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ...) -> None: ...

class Step4Request(_message.Message):
    __slots__ = ["step0", "step1", "step2", "step3"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    step3: Step3
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ..., step3: _Optional[_Union[Step3, _Mapping]] = ...) -> None: ...

class Step5Request(_message.Message):
    __slots__ = ["step0", "step1", "step2", "step3", "step4"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    STEP4_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    step3: Step3
    step4: Step4
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ..., step3: _Optional[_Union[Step3, _Mapping]] = ..., step4: _Optional[_Union[Step4, _Mapping]] = ...) -> None: ...

class Step6Request(_message.Message):
    __slots__ = ["step0", "step1", "step2", "step3", "step4", "step5"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    STEP4_FIELD_NUMBER: _ClassVar[int]
    STEP5_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    step3: Step3
    step4: Step4
    step5: Step5
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ..., step3: _Optional[_Union[Step3, _Mapping]] = ..., step4: _Optional[_Union[Step4, _Mapping]] = ..., step5: _Optional[_Union[Step5, _Mapping]] = ...) -> None: ...

class Step7Request(_message.Message):
    __slots__ = ["step0", "step1", "step2", "step3", "step4", "step5", "step6"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    STEP4_FIELD_NUMBER: _ClassVar[int]
    STEP5_FIELD_NUMBER: _ClassVar[int]
    STEP6_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    step3: Step3
    step4: Step4
    step5: Step5
    step6: Step6
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ..., step3: _Optional[_Union[Step3, _Mapping]] = ..., step4: _Optional[_Union[Step4, _Mapping]] = ..., step5: _Optional[_Union[Step5, _Mapping]] = ..., step6: _Optional[_Union[Step6, _Mapping]] = ...) -> None: ...

class Step8Request(_message.Message):
    __slots__ = ["step0", "step1", "step2", "step3", "step4", "step5", "step6", "step7"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    STEP4_FIELD_NUMBER: _ClassVar[int]
    STEP5_FIELD_NUMBER: _ClassVar[int]
    STEP6_FIELD_NUMBER: _ClassVar[int]
    STEP7_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    step3: Step3
    step4: Step4
    step5: Step5
    step6: Step6
    step7: Step7
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ..., step3: _Optional[_Union[Step3, _Mapping]] = ..., step4: _Optional[_Union[Step4, _Mapping]] = ..., step5: _Optional[_Union[Step5, _Mapping]] = ..., step6: _Optional[_Union[Step6, _Mapping]] = ..., step7: _Optional[_Union[Step7, _Mapping]] = ...) -> None: ...

class Step9Request(_message.Message):
    __slots__ = ["step0", "step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8"]
    STEP0_FIELD_NUMBER: _ClassVar[int]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    STEP4_FIELD_NUMBER: _ClassVar[int]
    STEP5_FIELD_NUMBER: _ClassVar[int]
    STEP6_FIELD_NUMBER: _ClassVar[int]
    STEP7_FIELD_NUMBER: _ClassVar[int]
    STEP8_FIELD_NUMBER: _ClassVar[int]
    step0: Step0
    step1: Step1
    step2: Step2
    step3: Step3
    step4: Step4
    step5: Step5
    step6: Step6
    step7: Step7
    step8: Step8
    def __init__(self, step0: _Optional[_Union[Step0, _Mapping]] = ..., step1: _Optional[_Union[Step1, _Mapping]] = ..., step2: _Optional[_Union[Step2, _Mapping]] = ..., step3: _Optional[_Union[Step3, _Mapping]] = ..., step4: _Optional[_Union[Step4, _Mapping]] = ..., step5: _Optional[_Union[Step5, _Mapping]] = ..., step6: _Optional[_Union[Step6, _Mapping]] = ..., step7: _Optional[_Union[Step7, _Mapping]] = ..., step8: _Optional[_Union[Step8, _Mapping]] = ...) -> None: ...

class Step1Response(_message.Message):
    __slots__ = ["step1"]
    STEP1_FIELD_NUMBER: _ClassVar[int]
    step1: Step1
    def __init__(self, step1: _Optional[_Union[Step1, _Mapping]] = ...) -> None: ...

class Step2Response(_message.Message):
    __slots__ = ["step2"]
    STEP2_FIELD_NUMBER: _ClassVar[int]
    step2: Step2
    def __init__(self, step2: _Optional[_Union[Step2, _Mapping]] = ...) -> None: ...

class Step3Response(_message.Message):
    __slots__ = ["step3"]
    STEP3_FIELD_NUMBER: _ClassVar[int]
    step3: Step3
    def __init__(self, step3: _Optional[_Union[Step3, _Mapping]] = ...) -> None: ...

class Step4Response(_message.Message):
    __slots__ = ["step4"]
    STEP4_FIELD_NUMBER: _ClassVar[int]
    step4: Step4
    def __init__(self, step4: _Optional[_Union[Step4, _Mapping]] = ...) -> None: ...

class Step5Response(_message.Message):
    __slots__ = ["step5"]
    STEP5_FIELD_NUMBER: _ClassVar[int]
    step5: Step5
    def __init__(self, step5: _Optional[_Union[Step5, _Mapping]] = ...) -> None: ...

class Step6Response(_message.Message):
    __slots__ = ["step6"]
    STEP6_FIELD_NUMBER: _ClassVar[int]
    step6: Step6
    def __init__(self, step6: _Optional[_Union[Step6, _Mapping]] = ...) -> None: ...

class Step7Response(_message.Message):
    __slots__ = ["step7"]
    STEP7_FIELD_NUMBER: _ClassVar[int]
    step7: Step7
    def __init__(self, step7: _Optional[_Union[Step7, _Mapping]] = ...) -> None: ...

class Step8Response(_message.Message):
    __slots__ = ["step8"]
    STEP8_FIELD_NUMBER: _ClassVar[int]
    step8: Step8
    def __init__(self, step8: _Optional[_Union[Step8, _Mapping]] = ...) -> None: ...

class Step9Response(_message.Message):
    __slots__ = ["step9"]
    STEP9_FIELD_NUMBER: _ClassVar[int]
    step9: Step9
    def __init__(self, step9: _Optional[_Union[Step9, _Mapping]] = ...) -> None: ...
