from __future__ import print_function

import logging

import grpc
import step_recomendation_pb2
import step_recomendation_pb2_grpc


def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    print("Will try to greet world ...")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = step_recomendation_pb2_grpc.Recomendation_systemStub(channel)

        # response = stub.recomend_step1(step_recomendation_pb2.Step1Request(houseArea = 1, siteArea = 2, floorCount = 3, region = 4, budgetFloor = 5, budgetCeil = 6)
        response = stub.recomend_step1(step_recomendation_pb2.Step1Request(step0 = step_recomendation_pb2.Step0(houseArea = 1, siteArea = 2, floorCount = 3, region = 'Адыгея', budgetFloor = 5, budgetCeil = 6)))
    print("Greeter client received: ")


if __name__ == "__main__":
    logging.basicConfig()
    run()
