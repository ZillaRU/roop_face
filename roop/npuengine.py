from tpu_perf.infer import SGInfer
import numpy as np 
import time 
import torch
import os
class EngineOV:
    
    def __init__(self, model_path="", batch=1,device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        
    def __str__(self):
        return "EngineOV: model_path={}, device_id={}".format(self.model_path,self.device_id)
    
        
    def __call__(self, args):
        start = time.time()
        if isinstance(args, list):
            values = args
        elif isinstance(args, dict):
            values = list(args.values())
        else:
            raise TypeError("args is not list or dict")
            # print(values)
        print(time.time() - start)
        start = time.time()
        task_id = self.model.put(*values)
        print("put time : ",time.time() - start)
        task_id, results, valid = self.model.get()
        return results