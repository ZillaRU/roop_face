from tpu_perf.infer import SGInfer
import numpy as np 
import time 
import torch
import os

class EngineOV:
    
    def __init__(self, model_path="", batch=1, device_id=0) :
        # 如果环境变量中没有设置device_id，则使用默认值
        if "DEVICE_ID" in os.environ:
            device_id = int(os.environ["DEVICE_ID"])
            print(">>>> device_id is in os.environ. and device_id = ",device_id)
        self.model_path = model_path
        self.model = SGInfer(model_path , batch=batch, devices=[device_id])
        self.device_id = device_id
        
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
        
if __name__ == '__main__':
        import sys
        import numpy as np
        import onnxruntime as ort
        
        def generate_random_inputs(input_specs):
            """根据输入规格生成随机测试数据"""
            inputs = {}
            for input_spec in input_specs:
                input_name = input_spec.name
                input_shape = [i if isinstance(i, int) else 1 for i in input_spec.shape]
                input_type = input_spec.type
                print(input_name, input_shape,input_type)
                # 假设所有输入都是浮点类型
                inputs[input_name] = np.random.rand(*input_shape).astype(np.float32 if input_type=="tensor(float)" else np.int64)
            return inputs
        
        def load_onnx_model(onnx_model_path):
            return ort.InferenceSession(onnx_model_path)

        def compare_results(onnx_results, bmodel_results):
            for i, (onnx_result, bmodel_result) in enumerate(zip(onnx_results, bmodel_results)):
                mean_abs_diff = np.mean(np.abs(onnx_result - bmodel_result))
                max_diff = np.max(np.abs(onnx_result - bmodel_result))
                print(f"Output {i}: Mean absolute difference: {mean_abs_diff}, Max difference: {max_diff}")
        
        if len(sys.argv) != 3:
            print("Usage: python npuengine.py <onnx_model_path> <bmodel_path>")
            sys.exit(1)
    
        onnx_model_path, bmodel_path = sys.argv[1], sys.argv[2]
    
        # 1. 加载ONNX模型
        onnx_session = load_onnx_model(onnx_model_path)
        input_specs = onnx_session.get_inputs()
    
        # 2. 生成测试数据
        input_data = generate_random_inputs(input_specs)
        np.savez("input_data.npz", input_data)
    
        # 3. ONNX模型推理
        onnx_results = onnx_session.run(None, input_data)
        # np.savez("onnx_inference_results.npz", onnx_results)
        
        # 4. 加载BModel
        bmodel = EngineOV(bmodel_path)
        #input_data['processed_lens'] = input_data['processed_lens'].astype(np.int32)
        # 5. BModel推理
        bmodel_results = bmodel(input_data)
        import pdb;pdb.set_trace()
        # 6. 比较结果
        compare_results(onnx_results, bmodel_results)
