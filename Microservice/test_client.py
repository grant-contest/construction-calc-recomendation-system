from __future__ import print_function

import logging

import grpc

import sys
sys.path.append('proto/')

import proto.step_recomendation_pb2 as sr_pb2
import proto.step_recomendation_pb2_grpc as sr_pb2_grpc

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = sr_pb2_grpc.Recomendation_systemStub(channel)

        step0 = sr_pb2.Step0(houseArea = 1, siteArea = 1, floorCount = 2, region = 'Адыгея', budgetFloor = 100_000, budgetCeil = 1_000_000)
        step1 = sr_pb2.Step1(sitePreparation = sr_pb2.SitePreparation(siteChoosing = True, geologicalWorks = True, 
                                                                      geodeticalWorks = True, cuttingBushesAndSmallForests = True,
                                                                      clearingTheSiteOfDebris = True),
                            siteWorks = sr_pb2.SiteWorks(cameras = True, temporaryFence = True),
                            houseDesignAndProject = sr_pb2.HouseDesignAndProject(homeProject = True, designProject = True))
        step2 = sr_pb2.Step2(foundationType = "Плитный")
        step3 = sr_pb2.Step3(wallsMaterial = "Кирпич")
        step4 = sr_pb2.Step4(slopesNumber = 1, roofType = "Металлочерепица")
        step5 = sr_pb2.Step5(facadeTechnology = "Без отделки")

        step6 = sr_pb2.Step6(windowMaterial = "Деревянные", windowType = "Однокамерные", doorMaterial = "Деревянные")
        step7 = sr_pb2.Step7(electrician = sr_pb2.Electrician(plasticBoxesUpTo40mmWide = True, layingAThreeToFive = True,
                                                              cableLaying = True, installationOfTwoKey = True,
                                                              installationOfSingleKey = True, recessedTypeSocketDevice = True,
                                                              installationOfPendant = True, chandeliersAndPendants = True),
                            waterSupply = sr_pb2.WaterSupply(layingOfInternalWaterSupplyPipelines = True, installationOfBathtubs = True,
                                                             installationOfSingle = True, installationOfMixers = True),
                            sewerage = sr_pb2.Sewerage(installationOfToilet = True, layingOfSewerage50mm = True, layingOfSewerage110mm = True),
                            heating = sr_pb2.Heating(assemblyOfAWaterSupply = True, layingOfInternalHeatingPipelines = True, installationOfWindowFixtures = True),
                            ventilation = sr_pb2.Ventilation(installationOfSplitSystems = True, cablingOnABrickWall = True)
                            )
        step8 = sr_pb2.Step8(warmFloor = True, ladderMaterial = "Дерево")
        step9 = sr_pb2.Step9(wallDecoration = "Декоративная штукатурка", floorCovering = "Ламинат", ceilCovering = "Натяжной потолок")
        # step7 = sr_pb2.Step7(windowMaterial = "Деревянные" 


        responses = [None] * 9
        responses[0] = stub.recomend_step1(sr_pb2.Step1Request(step0 = step0))
        responses[1] = stub.recomend_step2(sr_pb2.Step2Request(step0 = step0, step1 = step1))
        responses[2] = stub.recomend_step3(sr_pb2.Step3Request(step0 = step0, step1 = step1, step2 = step2))
        responses[3] = stub.recomend_step4(sr_pb2.Step4Request(step0 = step0, step1 = step1, step2 = step2, step3 = step3))
        responses[4] = stub.recomend_step5(sr_pb2.Step5Request(step0 = step0, step1 = step1, step2 = step2, step3 = step3, step4 = step4))
        responses[5] = stub.recomend_step6(sr_pb2.Step6Request(step0 = step0, step1 = step1, step2 = step2, step3 = step3, step4 = step4, step5 = step5))
        responses[6] = stub.recomend_step7(sr_pb2.Step7Request(step0 = step0, step1 = step1, step2 = step2, step3 = step3, step4 = step4, step5 = step5, step6 = step6))
        responses[7] = stub.recomend_step8(sr_pb2.Step8Request(step0 = step0, step1 = step1, step2 = step2, step3 = step3, step4 = step4, step5 = step5, step6 = step6, step7 = step7))
        responses[8] = stub.recomend_step9(sr_pb2.Step9Request(step0 = step0, step1 = step1, step2 = step2, step3 = step3, step4 = step4, step5 = step5, step6 = step6, step7 = step7, step8 = step8))

        print("Greeter client received: ")

        # if responses[0].step1.sitePreparation.cuttingBushesAndSmallForests:
        #     print("cuttingBushesAndSmallForests is true")
        # else:
        #     print("cuttingBushesAndSmallForests i false")
        
        for i in range(len(responses)):
            if responses[i] != None:
                print( responses[i])


if __name__ == "__main__":
    logging.basicConfig()
    run()
