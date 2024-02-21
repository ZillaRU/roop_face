# from tpu_perf.infer import SGInfer
from ..npuengine import EngineOV
# import numpy as np 
# import time 
# import torch
# import os
# import sophon.sail as sail 


# class EngineOV:
    
#     def __init__(self, model_path="",output_names="", device_id=0) :
#         # 如果环境变量中没有设置device_id，则使用默认值
#         if "DEVICE_ID" in os.environ:
#             device_id = int(os.environ["DEVICE_ID"])
#             print(">>>> device_id is in os.environ. and device_id = ",device_id)
#         self.model_path = model_path
#         self.device_id = device_id
#         try:
#             self.model = sail.Engine(model_path, device_id, sail.IOMode.SYSIO)
#         except Exception as e:
#             print("load model error; please check model path and device status;")
#             print(">>>> model_path: ",model_path)
#             print(">>>> device_id: ",device_id)
#             print(">>>> sail.Engine error: ",e)
#             raise e
#         sail.set_print_flag(True)
#         self.graph_name = self.model.get_graph_names()[0]
#         self.input_name = self.model.get_input_names(self.graph_name)
#         self.output_name= self.model.get_output_names(self.graph_name)

#     def __str__(self):
#         return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
#     def __call__(self, args):
#         if isinstance(args, list):
#             values = args
#         elif isinstance(args, dict):
#             values = list(args.values())
#         else:
#             raise TypeError("args is not list or dict")
#         args = {}
#         for i in range(len(values)):
#             args[self.input_name[i]] = values[i]
#         output = self.model.process(self.graph_name, args)
#         res = []

#         for name in self.output_name:
#             res.append(output[name])
#         return res