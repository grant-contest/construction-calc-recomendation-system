from __future__ import print_function

import logging

import grpc
import step_recomendation_pb2 as sr_pb2
import step_recomendation_pb2_grpc as sr_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = sr_pb2_grpc.Recomendation_systemStub(channel)

        step0 = sr_pb2.Step0(houseArea = 1, siteArea = 2, floorCount = 3, region = 'Адыгея', budgetFloor = 5, budgetCeil = 6)
        step1 = sr_pb2.Step1(sitePreparation = sr_pb2.SitePreparation(siteChoosing = True, geologicalWorks = True, 
                                                                      geodeticalWorks = True, cuttingBushesAndSmallForests = True,
                                                                      clearingTheSiteOfDebris = True),
                            siteWorks = sr_pb2.SiteWorks(cameras = True, temporaryFence = True),
                            houseDesignAndProject = sr_pb2.HouseDesignAndProject(homeProject = True, designProject = True))
        # step2 = sr_pb2.Step2()
        responses = [None]*9
        responses[0] = stub.recomend_step1(sr_pb2.Step1Request(step0 = step0))
        responses[1] = stub.recomend_step2(sr_pb2.Step2Request(step0 = step0, step1 = step1))
        # responses[2] = stub.recomend_step2(sr_pb2.Step2Request(step0 = step0, step1 = step1))

        print("Greeter client received: ")

        # if responses[0].step1.sitePreparation.cuttingBushesAndSmallForests:
        #     print("cuttingBushesAndSmallForests is true")
        # else:
        #     print("cuttingBushesAndSmallForests i false")
        
        for i in range(len(responses)):
            if responses[i] != None:
                print(f"step %d:" % (i), responses[i])


if __name__ == "__main__":
    logging.basicConfig()
    run()
