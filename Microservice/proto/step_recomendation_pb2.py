# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: step_recomendation.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x18step_recomendation.proto\"\xd6\x03\n\x05Step0\x12\x11\n\thouseArea\x18\x01 \x01(\r\x12\x10\n\x08siteArea\x18\x02 \x01(\r\x12\x12\n\nfloorCount\x18\x03 \x01(\r\x12\x0e\n\x06region\x18\x04 \x01(\t\x12\x0f\n\x07purpose\x18\x05 \x01(\t\x12\x13\n\x0b\x62udgetFloor\x18\x06 \x01(\r\x12\x12\n\nbudgetCeil\x18\x07 \x01(\r\x12\x1b\n\x0ehouseAreaPrice\x18\x08 \x01(\rH\x00\x88\x01\x01\x12\x1a\n\rsiteAreaPrice\x18\t \x01(\rH\x01\x88\x01\x01\x12\x1c\n\x0f\x66loorCountPrice\x18\n \x01(\rH\x02\x88\x01\x01\x12\x18\n\x0bregionPrice\x18\x0b \x01(\rH\x03\x88\x01\x01\x12\x19\n\x0cpurposePrice\x18\x0c \x01(\rH\x04\x88\x01\x01\x12\x1d\n\x10\x62udgetFloorPrice\x18\r \x01(\rH\x05\x88\x01\x01\x12\x1c\n\x0f\x62udgetCeilPrice\x18\x0e \x01(\rH\x06\x88\x01\x01\x42\x11\n\x0f_houseAreaPriceB\x10\n\x0e_siteAreaPriceB\x12\n\x10_floorCountPriceB\x0e\n\x0c_regionPriceB\x0f\n\r_purposePriceB\x13\n\x11_budgetFloorPriceB\x12\n\x10_budgetCeilPrice\"\xf0\x03\n\x0fSitePreparation\x12\x14\n\x0csiteChoosing\x18\x01 \x01(\x08\x12\x17\n\x0fgeologicalWorks\x18\x02 \x01(\x08\x12\x17\n\x0fgeodeticalWorks\x18\x03 \x01(\x08\x12$\n\x1c\x63uttingBushesAndSmallForests\x18\x04 \x01(\x08\x12\x1f\n\x17\x63learingTheSiteOfDebris\x18\x05 \x01(\x08\x12\x1e\n\x11siteChoosingPrice\x18\x06 \x01(\rH\x00\x88\x01\x01\x12!\n\x14geologicalWorksPrice\x18\x07 \x01(\rH\x01\x88\x01\x01\x12!\n\x14geodeticalWorksPrice\x18\x08 \x01(\rH\x02\x88\x01\x01\x12.\n!cuttingBushesAndSmallForestsPrice\x18\t \x01(\rH\x03\x88\x01\x01\x12)\n\x1c\x63learingTheSiteOfDebrisPrice\x18\n \x01(\rH\x04\x88\x01\x01\x42\x14\n\x12_siteChoosingPriceB\x17\n\x15_geologicalWorksPriceB\x17\n\x15_geodeticalWorksPriceB$\n\"_cuttingBushesAndSmallForestsPriceB\x1f\n\x1d_clearingTheSiteOfDebrisPrice\"\x9a\x01\n\tSiteWorks\x12\x0f\n\x07\x63\x61meras\x18\x01 \x01(\x08\x12\x16\n\x0etemporaryFence\x18\x02 \x01(\x08\x12\x19\n\x0c\x63\x61merasPrice\x18\x03 \x01(\rH\x00\x88\x01\x01\x12 \n\x13temporaryFencePrice\x18\x04 \x01(\rH\x01\x88\x01\x01\x42\x0f\n\r_camerasPriceB\x16\n\x14_temporaryFencePrice\"\xaf\x01\n\x15HouseDesignAndProject\x12\x13\n\x0bhomeProject\x18\x01 \x01(\x08\x12\x15\n\rdesignProject\x18\x02 \x01(\x08\x12\x1d\n\x10homeProjectPrice\x18\x03 \x01(\rH\x00\x88\x01\x01\x12\x1f\n\x12\x64\x65signProjectPrice\x18\x04 \x01(\rH\x01\x88\x01\x01\x42\x13\n\x11_homeProjectPriceB\x15\n\x13_designProjectPrice\"\x88\x01\n\x05Step1\x12)\n\x0fsitePreparation\x18\x01 \x01(\x0b\x32\x10.SitePreparation\x12\x1d\n\tsiteWorks\x18\x02 \x01(\x0b\x32\n.SiteWorks\x12\x35\n\x15houseDesignAndProject\x18\x03 \x01(\x0b\x32\x16.HouseDesignAndProject\"Y\n\x05Step2\x12\x16\n\x0e\x66oundationType\x18\x01 \x01(\t\x12 \n\x13\x66oundationTypePrice\x18\x02 \x01(\rH\x00\x88\x01\x01\x42\x16\n\x14_foundationTypePrice\"V\n\x05Step3\x12\x15\n\rwallsMaterial\x18\x01 \x01(\t\x12\x1f\n\x12wallsMaterialPrice\x18\x02 \x01(\rH\x00\x88\x01\x01\x42\x15\n\x13_wallsMaterialPrice\"/\n\x05Step4\x12\x14\n\x0cslopesNumber\x18\x01 \x01(\r\x12\x10\n\x08roofType\x18\x02 \x01(\t\"!\n\x05Step5\x12\x18\n\x10\x66\x61\x63\x61\x64\x65Technology\x18\x01 \x01(\t\"I\n\x05Step6\x12\x16\n\x0ewindowMaterial\x18\x01 \x01(\t\x12\x12\n\nwindowType\x18\x02 \x01(\t\x12\x14\n\x0c\x64oorMaterial\x18\x03 \x01(\t\"\x80\x02\n\x0b\x45lectrician\x12 \n\x18plasticBoxesUpTo40mmWide\x18\x01 \x01(\x08\x12\x1a\n\x12layingAThreeToFive\x18\x02 \x01(\x08\x12\x13\n\x0b\x63\x61\x62leLaying\x18\x03 \x01(\x08\x12\x1c\n\x14installationOfTwoKey\x18\x04 \x01(\x08\x12\x1f\n\x17installationOfSingleKey\x18\x05 \x01(\x08\x12 \n\x18recessedTypeSocketDevice\x18\x06 \x01(\x08\x12\x1d\n\x15installationOfPendant\x18\x07 \x01(\x08\x12\x1e\n\x16\x63handeliersAndPendants\x18\x08 \x01(\x08\"\x97\x01\n\x0bWaterSupply\x12,\n$layingOfInternalWaterSupplyPipelines\x18\x01 \x01(\x08\x12\x1e\n\x16installationOfBathtubs\x18\x02 \x01(\x08\x12\x1c\n\x14installationOfSingle\x18\x03 \x01(\x08\x12\x1c\n\x14installationOfMixers\x18\x04 \x01(\x08\"e\n\x08Sewerage\x12\x1c\n\x14installationOfToilet\x18\x01 \x01(\x08\x12\x1c\n\x14layingOfSewerage50mm\x18\x02 \x01(\x08\x12\x1d\n\x15layingOfSewerage110mm\x18\x03 \x01(\x08\"y\n\x07Heating\x12\x1e\n\x16\x61ssemblyOfAWaterSupply\x18\x01 \x01(\x08\x12(\n layingOfInternalHeatingPipelines\x18\x02 \x01(\x08\x12$\n\x1cinstallationOfWindowFixtures\x18\x03 \x01(\x08\"N\n\x0bVentilation\x12\"\n\x1ainstallationOfSplitSystems\x18\x01 \x01(\x08\x12\x1b\n\x13\x63\x61\x62lingOnABrickWall\x18\x02 \x01(\x08\"\xa8\x01\n\x05Step7\x12!\n\x0b\x65lectrician\x18\x01 \x01(\x0b\x32\x0c.Electrician\x12!\n\x0bwaterSupply\x18\x02 \x01(\x0b\x32\x0c.WaterSupply\x12\x1b\n\x08sewerage\x18\x03 \x01(\x0b\x32\t.Sewerage\x12\x19\n\x07heating\x18\x04 \x01(\x0b\x32\x08.Heating\x12!\n\x0bventilation\x18\x05 \x01(\x0b\x32\x0c.Ventilation\"2\n\x05Step8\x12\x11\n\twarmFloor\x18\x01 \x01(\x08\x12\x16\n\x0eladderMaterial\x18\x02 \x01(\t\"L\n\x05Step9\x12\x16\n\x0ewallDecoration\x18\x01 \x01(\t\x12\x15\n\rfloorCovering\x18\x02 \x01(\t\x12\x14\n\x0c\x63\x65ilCovering\x18\x03 \x01(\t\"(\n\tStepFinal\x12\x1b\n\x13\x61\x64\x64itionalBuildings\x18\x01 \x01(\t\"%\n\x0cStep1Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\"<\n\x0cStep2Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\"S\n\x0cStep3Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\"j\n\x0cStep4Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\"\x81\x01\n\x0cStep5Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\x12\x15\n\x05step4\x18\x05 \x01(\x0b\x32\x06.Step4\"\x98\x01\n\x0cStep6Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\x12\x15\n\x05step4\x18\x05 \x01(\x0b\x32\x06.Step4\x12\x15\n\x05step5\x18\x06 \x01(\x0b\x32\x06.Step5\"\xaf\x01\n\x0cStep7Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\x12\x15\n\x05step4\x18\x05 \x01(\x0b\x32\x06.Step4\x12\x15\n\x05step5\x18\x06 \x01(\x0b\x32\x06.Step5\x12\x15\n\x05step6\x18\x07 \x01(\x0b\x32\x06.Step6\"\xc6\x01\n\x0cStep8Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\x12\x15\n\x05step4\x18\x05 \x01(\x0b\x32\x06.Step4\x12\x15\n\x05step5\x18\x06 \x01(\x0b\x32\x06.Step5\x12\x15\n\x05step6\x18\x07 \x01(\x0b\x32\x06.Step6\x12\x15\n\x05step7\x18\x08 \x01(\x0b\x32\x06.Step7\"\xdd\x01\n\x0cStep9Request\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\x12\x15\n\x05step4\x18\x05 \x01(\x0b\x32\x06.Step4\x12\x15\n\x05step5\x18\x06 \x01(\x0b\x32\x06.Step5\x12\x15\n\x05step6\x18\x07 \x01(\x0b\x32\x06.Step6\x12\x15\n\x05step7\x18\x08 \x01(\x0b\x32\x06.Step7\x12\x15\n\x05step8\x18\t \x01(\x0b\x32\x06.Step8\"\xf4\x01\n\x0c\x46inalRequest\x12\x15\n\x05step0\x18\x01 \x01(\x0b\x32\x06.Step0\x12\x15\n\x05step1\x18\x02 \x01(\x0b\x32\x06.Step1\x12\x15\n\x05step2\x18\x03 \x01(\x0b\x32\x06.Step2\x12\x15\n\x05step3\x18\x04 \x01(\x0b\x32\x06.Step3\x12\x15\n\x05step4\x18\x05 \x01(\x0b\x32\x06.Step4\x12\x15\n\x05step5\x18\x06 \x01(\x0b\x32\x06.Step5\x12\x15\n\x05step6\x18\x07 \x01(\x0b\x32\x06.Step6\x12\x15\n\x05step7\x18\x08 \x01(\x0b\x32\x06.Step7\x12\x15\n\x05step8\x18\t \x01(\x0b\x32\x06.Step8\x12\x15\n\x05step9\x18\n \x01(\x0b\x32\x06.Step9\"&\n\rStep1Response\x12\x15\n\x05step1\x18\x01 \x01(\x0b\x32\x06.Step1\"&\n\rStep2Response\x12\x15\n\x05step2\x18\x01 \x01(\x0b\x32\x06.Step2\"&\n\rStep3Response\x12\x15\n\x05step3\x18\x01 \x01(\x0b\x32\x06.Step3\"&\n\rStep4Response\x12\x15\n\x05step4\x18\x01 \x01(\x0b\x32\x06.Step4\"&\n\rStep5Response\x12\x15\n\x05step5\x18\x01 \x01(\x0b\x32\x06.Step5\"&\n\rStep6Response\x12\x15\n\x05step6\x18\x01 \x01(\x0b\x32\x06.Step6\"&\n\rStep7Response\x12\x15\n\x05step7\x18\x01 \x01(\x0b\x32\x06.Step7\"&\n\rStep8Response\x12\x15\n\x05step8\x18\x01 \x01(\x0b\x32\x06.Step8\"&\n\rStep9Response\x12\x15\n\x05step9\x18\x01 \x01(\x0b\x32\x06.Step9\".\n\rFinalResponse\x12\x1d\n\tstepFinal\x18\x01 \x01(\x0b\x32\n.StepFinal2\x94\x04\n\x14Recomendation_system\x12\x31\n\x0erecomend_step1\x12\r.Step1Request\x1a\x0e.Step1Response\"\x00\x12\x31\n\x0erecomend_step2\x12\r.Step2Request\x1a\x0e.Step2Response\"\x00\x12\x31\n\x0erecomend_step3\x12\r.Step3Request\x1a\x0e.Step3Response\"\x00\x12\x31\n\x0erecomend_step4\x12\r.Step4Request\x1a\x0e.Step4Response\"\x00\x12\x31\n\x0erecomend_step5\x12\r.Step5Request\x1a\x0e.Step5Response\"\x00\x12\x31\n\x0erecomend_step6\x12\r.Step6Request\x1a\x0e.Step6Response\"\x00\x12\x31\n\x0erecomend_step7\x12\r.Step7Request\x1a\x0e.Step7Response\"\x00\x12\x31\n\x0erecomend_step8\x12\r.Step8Request\x1a\x0e.Step8Response\"\x00\x12\x31\n\x0erecomend_step9\x12\r.Step9Request\x1a\x0e.Step9Response\"\x00\x12\x31\n\x0erecomend_final\x12\r.FinalRequest\x1a\x0e.FinalResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'step_recomendation_pb2', _globals)
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  _globals['_STEP0']._serialized_start=29
  _globals['_STEP0']._serialized_end=499
  _globals['_SITEPREPARATION']._serialized_start=502
  _globals['_SITEPREPARATION']._serialized_end=998
  _globals['_SITEWORKS']._serialized_start=1001
  _globals['_SITEWORKS']._serialized_end=1155
  _globals['_HOUSEDESIGNANDPROJECT']._serialized_start=1158
  _globals['_HOUSEDESIGNANDPROJECT']._serialized_end=1333
  _globals['_STEP1']._serialized_start=1336
  _globals['_STEP1']._serialized_end=1472
  _globals['_STEP2']._serialized_start=1474
  _globals['_STEP2']._serialized_end=1563
  _globals['_STEP3']._serialized_start=1565
  _globals['_STEP3']._serialized_end=1651
  _globals['_STEP4']._serialized_start=1653
  _globals['_STEP4']._serialized_end=1700
  _globals['_STEP5']._serialized_start=1702
  _globals['_STEP5']._serialized_end=1735
  _globals['_STEP6']._serialized_start=1737
  _globals['_STEP6']._serialized_end=1810
  _globals['_ELECTRICIAN']._serialized_start=1813
  _globals['_ELECTRICIAN']._serialized_end=2069
  _globals['_WATERSUPPLY']._serialized_start=2072
  _globals['_WATERSUPPLY']._serialized_end=2223
  _globals['_SEWERAGE']._serialized_start=2225
  _globals['_SEWERAGE']._serialized_end=2326
  _globals['_HEATING']._serialized_start=2328
  _globals['_HEATING']._serialized_end=2449
  _globals['_VENTILATION']._serialized_start=2451
  _globals['_VENTILATION']._serialized_end=2529
  _globals['_STEP7']._serialized_start=2532
  _globals['_STEP7']._serialized_end=2700
  _globals['_STEP8']._serialized_start=2702
  _globals['_STEP8']._serialized_end=2752
  _globals['_STEP9']._serialized_start=2754
  _globals['_STEP9']._serialized_end=2830
  _globals['_STEPFINAL']._serialized_start=2832
  _globals['_STEPFINAL']._serialized_end=2872
  _globals['_STEP1REQUEST']._serialized_start=2874
  _globals['_STEP1REQUEST']._serialized_end=2911
  _globals['_STEP2REQUEST']._serialized_start=2913
  _globals['_STEP2REQUEST']._serialized_end=2973
  _globals['_STEP3REQUEST']._serialized_start=2975
  _globals['_STEP3REQUEST']._serialized_end=3058
  _globals['_STEP4REQUEST']._serialized_start=3060
  _globals['_STEP4REQUEST']._serialized_end=3166
  _globals['_STEP5REQUEST']._serialized_start=3169
  _globals['_STEP5REQUEST']._serialized_end=3298
  _globals['_STEP6REQUEST']._serialized_start=3301
  _globals['_STEP6REQUEST']._serialized_end=3453
  _globals['_STEP7REQUEST']._serialized_start=3456
  _globals['_STEP7REQUEST']._serialized_end=3631
  _globals['_STEP8REQUEST']._serialized_start=3634
  _globals['_STEP8REQUEST']._serialized_end=3832
  _globals['_STEP9REQUEST']._serialized_start=3835
  _globals['_STEP9REQUEST']._serialized_end=4056
  _globals['_FINALREQUEST']._serialized_start=4059
  _globals['_FINALREQUEST']._serialized_end=4303
  _globals['_STEP1RESPONSE']._serialized_start=4305
  _globals['_STEP1RESPONSE']._serialized_end=4343
  _globals['_STEP2RESPONSE']._serialized_start=4345
  _globals['_STEP2RESPONSE']._serialized_end=4383
  _globals['_STEP3RESPONSE']._serialized_start=4385
  _globals['_STEP3RESPONSE']._serialized_end=4423
  _globals['_STEP4RESPONSE']._serialized_start=4425
  _globals['_STEP4RESPONSE']._serialized_end=4463
  _globals['_STEP5RESPONSE']._serialized_start=4465
  _globals['_STEP5RESPONSE']._serialized_end=4503
  _globals['_STEP6RESPONSE']._serialized_start=4505
  _globals['_STEP6RESPONSE']._serialized_end=4543
  _globals['_STEP7RESPONSE']._serialized_start=4545
  _globals['_STEP7RESPONSE']._serialized_end=4583
  _globals['_STEP8RESPONSE']._serialized_start=4585
  _globals['_STEP8RESPONSE']._serialized_end=4623
  _globals['_STEP9RESPONSE']._serialized_start=4625
  _globals['_STEP9RESPONSE']._serialized_end=4663
  _globals['_FINALRESPONSE']._serialized_start=4665
  _globals['_FINALRESPONSE']._serialized_end=4711
  _globals['_RECOMENDATION_SYSTEM']._serialized_start=4714
  _globals['_RECOMENDATION_SYSTEM']._serialized_end=5246
# @@protoc_insertion_point(module_scope)
