import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch.ao.quantization.quantizer import X86InductorQuantizer
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e
import copy
import torch._dynamo as torchdynamo
import torch

def get_inductor_int8_convert_model(model, example_inputs):
    # x = example_inputs[0]
    # example_inputs = (x.contiguous(memory_format=torch.channels_last), )
    with torch.no_grad():
        exported_model, guards = torchdynamo.export(
            model,
            *copy.deepcopy(example_inputs),
            aten_graph=True,
        )

        quantizer = X86InductorQuantizer()
        operator_spec = xiq.get_default_x86_inductor_quantization_config()
        quantizer.set_global(operator_spec)
        prepared_model = prepare_pt2e(exported_model, quantizer)
        # calibration
        prepared_model(*example_inputs)
        converted_model = convert_pt2e(prepared_model).eval()
        print("converted_model is: {}".format(converted_model), flush=True)

        # optimized_model = torch.compile(converted_model)

        # import time
        # for i in range(200):
        #     start = time.time()
        #     inductor_result = optimized_model(*example_inputs)
        #     print(time.time()- start, flush=True)

        return converted_model
