# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import step_recomendation_pb2 as step__recomendation__pb2


class Recomendation_systemStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.recomend_step1 = channel.unary_unary(
                '/Recomendation_system/recomend_step1',
                request_serializer=step__recomendation__pb2.Step1Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step1Response.FromString,
                )
        self.recomend_step2 = channel.unary_unary(
                '/Recomendation_system/recomend_step2',
                request_serializer=step__recomendation__pb2.Step2Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step2Response.FromString,
                )
        self.recomend_step3 = channel.unary_unary(
                '/Recomendation_system/recomend_step3',
                request_serializer=step__recomendation__pb2.Step3Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step3Response.FromString,
                )
        self.recomend_step4 = channel.unary_unary(
                '/Recomendation_system/recomend_step4',
                request_serializer=step__recomendation__pb2.Step4Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step4Response.FromString,
                )
        self.recomend_step5 = channel.unary_unary(
                '/Recomendation_system/recomend_step5',
                request_serializer=step__recomendation__pb2.Step5Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step5Response.FromString,
                )
        self.recomend_step6 = channel.unary_unary(
                '/Recomendation_system/recomend_step6',
                request_serializer=step__recomendation__pb2.Step6Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step6Response.FromString,
                )
        self.recomend_step7 = channel.unary_unary(
                '/Recomendation_system/recomend_step7',
                request_serializer=step__recomendation__pb2.Step7Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step7Response.FromString,
                )
        self.recomend_step8 = channel.unary_unary(
                '/Recomendation_system/recomend_step8',
                request_serializer=step__recomendation__pb2.Step8Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step8Response.FromString,
                )
        self.recomend_step9 = channel.unary_unary(
                '/Recomendation_system/recomend_step9',
                request_serializer=step__recomendation__pb2.Step9Request.SerializeToString,
                response_deserializer=step__recomendation__pb2.Step9Response.FromString,
                )


class Recomendation_systemServicer(object):
    """Missing associated documentation comment in .proto file."""

    def recomend_step1(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step2(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step3(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step4(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step5(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step6(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step7(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step8(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def recomend_step9(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_Recomendation_systemServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'recomend_step1': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step1,
                    request_deserializer=step__recomendation__pb2.Step1Request.FromString,
                    response_serializer=step__recomendation__pb2.Step1Response.SerializeToString,
            ),
            'recomend_step2': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step2,
                    request_deserializer=step__recomendation__pb2.Step2Request.FromString,
                    response_serializer=step__recomendation__pb2.Step2Response.SerializeToString,
            ),
            'recomend_step3': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step3,
                    request_deserializer=step__recomendation__pb2.Step3Request.FromString,
                    response_serializer=step__recomendation__pb2.Step3Response.SerializeToString,
            ),
            'recomend_step4': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step4,
                    request_deserializer=step__recomendation__pb2.Step4Request.FromString,
                    response_serializer=step__recomendation__pb2.Step4Response.SerializeToString,
            ),
            'recomend_step5': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step5,
                    request_deserializer=step__recomendation__pb2.Step5Request.FromString,
                    response_serializer=step__recomendation__pb2.Step5Response.SerializeToString,
            ),
            'recomend_step6': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step6,
                    request_deserializer=step__recomendation__pb2.Step6Request.FromString,
                    response_serializer=step__recomendation__pb2.Step6Response.SerializeToString,
            ),
            'recomend_step7': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step7,
                    request_deserializer=step__recomendation__pb2.Step7Request.FromString,
                    response_serializer=step__recomendation__pb2.Step7Response.SerializeToString,
            ),
            'recomend_step8': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step8,
                    request_deserializer=step__recomendation__pb2.Step8Request.FromString,
                    response_serializer=step__recomendation__pb2.Step8Response.SerializeToString,
            ),
            'recomend_step9': grpc.unary_unary_rpc_method_handler(
                    servicer.recomend_step9,
                    request_deserializer=step__recomendation__pb2.Step9Request.FromString,
                    response_serializer=step__recomendation__pb2.Step9Response.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'Recomendation_system', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Recomendation_system(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def recomend_step1(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step1',
            step__recomendation__pb2.Step1Request.SerializeToString,
            step__recomendation__pb2.Step1Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step2(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step2',
            step__recomendation__pb2.Step2Request.SerializeToString,
            step__recomendation__pb2.Step2Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step3(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step3',
            step__recomendation__pb2.Step3Request.SerializeToString,
            step__recomendation__pb2.Step3Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step4(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step4',
            step__recomendation__pb2.Step4Request.SerializeToString,
            step__recomendation__pb2.Step4Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step5(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step5',
            step__recomendation__pb2.Step5Request.SerializeToString,
            step__recomendation__pb2.Step5Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step6(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step6',
            step__recomendation__pb2.Step6Request.SerializeToString,
            step__recomendation__pb2.Step6Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step7(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step7',
            step__recomendation__pb2.Step7Request.SerializeToString,
            step__recomendation__pb2.Step7Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step8(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step8',
            step__recomendation__pb2.Step8Request.SerializeToString,
            step__recomendation__pb2.Step8Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def recomend_step9(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/Recomendation_system/recomend_step9',
            step__recomendation__pb2.Step9Request.SerializeToString,
            step__recomendation__pb2.Step9Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
