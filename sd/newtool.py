
from .untool import *
import numpy as np
import logging
from collections import defaultdict
from tqdm import tqdm

mylog = logging.getLogger(__name__)
MLIR_START   = 4294967296
# print better 
np.set_printoptions(precision=6, suppress=True)
def cpu(self, force=True, record=True):
    self.npy
    copy_tensor_to_host(self, force)

def malloc_device_addr(self):
    if self.bm_handle is None:
        self.get_bm_handle()
    malloc_device(self)

def npu(self, force=True, record=True):
    copy_tensor_to_device(self, force)

def malloc_host_addr(self):
    assert self.dims > 0
    shape = list(self.shape)[:self.dims]
    dtype = type_map[self.dtype]
    if self.is_malloc_host is False:
        self.npy__ = np.zeros(shape, dtype=dtype)
        self.data  = make_np2c(self.npy__)
        self.is_from_np = True
        return 
    dtype = type_map[dtype]
    self.npy__ = make_c2np(self.data, shape, dtype)

@property
def npy(self):
    if self.is_from_np is False:
        if self.is_malloc_host is True:
            shape = list(self.shape)[:self.dims]
            data_type = type_map[self.dtype]
            self.npy__ = make_c2np(self.data, shape, data_type)
            self.is_from_np = True
        else:
            if self.size == 0:
                size = self.max_size // np.dtype(type_map[self.dtype]).itemsize
                self.npy__ = np.zeros(size, dtype=type_map[self.dtype])
                self.is_from_np = True
                self.data  = make_np2c(self.npy__)
            else:
                shape = list(self.shape)[:self.dims]
                data_type = type_map[self.dtype]
                self.npy__ = np.zeros(shape, dtype=data_type)
                self.data  = make_np2c(self.npy__)
                self.is_from_np = True
        return self.npy__
    else:
        return self.npy__

@classmethod
def create_instance(cls):
    return cls()
        
@staticmethod
def build_from_np(np_array, device_id=-1):
    res = UntensorS.create_instance()
    res.data_type = data_type[np_array.dtype]
    res.dims      = len(np_array.shape)
    for i in range(res.dims):
        res.shape[i] = np_array.shape[i]
    res.size      = np_array.size * np_array.dtype.itemsize
    res.max_size  = res.size
    res.device_id = device_id
    setattr(res, 'npy__', np_array)
    res.is_from_np = True
    res.data = make_np2c(np_array)
    res.get_bm_handle()
    return res

def get_bm_handle(self):
    if self.bm_handle is not None:
        return self.bm_handle
    else:
        if self.device_id == -1:
            return None
        else:
            self.bm_handle = get_handle(self.device_id)
            self.malloc_bm_handle = True
            return self.bm_handle

def update_extra(self, ref_tensor):
    if ref_tensor:
        self.father_tensor = ref_tensor.father_tensor

@property
def cur_status(self):
    status = {}
    status['name']        = self.name
    status['dtype']       = self.dtype
    status['dims']        = self.dims
    status['shape']       = list(self.shape)[:self.dims]
    status['size']        = self.size
    status['max_size']    = self.max_size
    status['device_id']   = self.device_id
    status['is_from_np']  = self.is_from_np
    status['is_malloc_host'] = self.is_malloc_host
    status['is_malloc_device'] = self.is_malloc_device
    status['is_in_device'] = self.is_in_device
    status['data']        = self.data
    status['is_from_np']  = self.is_from_np
    status['is_copy']     = self.is_copy
    status['device_mem_addr'] = self.device_mem_addr
    status['device_mem_size'] = self.device_mem_size
    status['dmabuf_fd']       = self.dmabuf_fd
    status['reserved']        = self.reserved
    status['bm_handle']       = self.bm_handle
    status['copy_tensor']     = self.copy_tensor
    return status

def set_numpy(self, npy, do_necessary=True):
    self.is_from_np = True
    self.npy__      = npy
    if do_necessary is True:
        if npy.dtype != type_map[self.dtype]:
            npy = npy.astype(type_map[self.dtype])
    assert self.dtype == data_type[npy.dtype]
    assert self.max_size >= npy.size * npy.dtype.itemsize
    if self.max_size  == npy.size * npy.dtype.itemsize:
        assert list(self.shape)[:self.dims] == list(npy.shape)
    else:
        for i in range(self.dims):
            assert self.shape[i] == npy.shape[i]
    self.data       = make_np2c(npy)

def set_dtype_shape(self, dtype, shape, regen_npy=True):
    self.dtype = dtype
    self.dims  = len(shape)
    for i in range(self.dims):
        self.shape[i] = shape[i]
    self.size = np.prod(shape) * np.dtype(type_map[dtype]).itemsize
    if self.is_from_np and regen_npy is True:
        self.npy__ = np.zeros(shape, dtype=type_map[dtype])
        self.data  = make_np2c(self.npy__)

def find_father(self) -> UntensorS:
    if self.is_copy:
        return self.father_tensor.find_father()
        # cur_tensor = ctypes.cast(self.copy_tensor, ctypes.POINTER(UntensorS)).contents
        # return cur_tensor.find_father()
    else:
        return self

def set_copy_tensor(self, copy_tensor:UntensorS):
    self.is_copy     = True
    self.copy_tensor = ctypes.addressof(copy_tensor)
    self.father_tensor = copy_tensor

def diff_set_with_flag(self, npdata, flag=0, do_necessary=True):
    # 0: force copy, 1: simple check copy , 2: force not copy 
    tensor = self.find_father()
    if npdata.dtype != type_map[self.dtype] and do_necessary:
        npdata = npdata.astype(type_map[self.dtype])
    if flag == 0:
        tensor.data  = make_np2c(npdata)
        tensor.npy__ = npdata
        tensor.is_from_np = True
        tensor.size       = npdata.size * npdata.dtype.itemsize
        tensor.npu()
    else:
        ValueError("not support")

def copy_tensor_data(self, tensor:UntensorS, flag=0):
    tensor = tensor.find_father()
    if flag == 0:
        copy_data_from_tensor_with_flag(self, tensor, 0)
    else:
        ValueError("not support")

def __del__(self):
    pass

UntensorS.cpu                = cpu
UntensorS.npu                = npu
UntensorS.malloc_device_addr = malloc_device_addr
UntensorS.malloc_host_addr   = malloc_host_addr
UntensorS.npy                = npy
UntensorS.create_instance    = create_instance
UntensorS.build_from_np      = build_from_np
UntensorS.get_bm_handle      = get_bm_handle
UntensorS.cur_status         = cur_status
UntensorS.set_copy_tensor    = set_copy_tensor
UntensorS.set_numpy          = set_numpy
UntensorS.set_dtype_shape    = set_dtype_shape
UntensorS.diff_set_with_flag = diff_set_with_flag
UntensorS.copy_tensor_data   = copy_tensor_data
UntensorS.find_father        = find_father
UntensorS.father_tensor      = None
UntensorS.update_extra       = update_extra
UntensorS.__del__            = __del__

def convert_tensor_c_into_dict(tensor_c):
    res = {}
    res['name']        = tensor_c.name.decode('utf-8') 
    res['data_type']   = tensor_c.data_type
    # res['gmem_stmode'] = tensor_c.gmem_stmode
    res['size']        = tensor_c.size
    res['data_shape']  = []
    for i in list(tensor_c.shape):
        if i!=0:
            res['data_shape'].append(i)
    # res['mem_type']    = tensor_c.mem_type
    # res['scale']       = tensor_c.scale
    # res['cpu_addr']    = tensor_c.cpu_addr
    # res['pad_h']       = tensor_c.pad_h
    # res['zero_point']  = tensor_c.zero_point
    return res

def convert_device_mem_into_dict(device_mem_c):
    res = {}
    res['addr'] = device_mem_c.addr
    res['size'] = device_mem_c.size
    res['dmabuf_fd'] = device_mem_c.dmabuf_fd
    res['reserve']   = device_mem_c.reserve
    return res

def convert_cmd_group_c_into_dict(cmd_group_c):
    res = {}
    return res

def convert_coeff_size_map_c_into_dict(coeff_size_map_c):
    res = {}
    res['coeff_start_addr'] = coeff_size_map_c.coeff_start_addr
    res['coeff_size']       = coeff_size_map_c.coeff_size
    res['neuron_size']      = coeff_size_map_c.neuron_size
    res['neuron_device']    = coeff_size_map_c.neuron_device
    return res

def convert_stage_info_c_into_dict(stage_info_c):
    res = {}
    res['input_tensor']  = []
    res['output_tensor'] = []
    for i in range(stage_info_c.input_num):
        temp_tensor = stage_info_c.input_tensor[i]
        res['input_tensor'].append(convert_tensor_c_into_dict(temp_tensor))
    for i in range(stage_info_c.output_num):
        temp_tensor = stage_info_c.output_tensor[i]
        res['output_tensor'].append(convert_tensor_c_into_dict(temp_tensor))
    res['cmd_group_num'] = stage_info_c.cmd_group_num
    res['cmd_group']     = []
    for i in range(res['cmd_group_num']):
        temp_cmd_group = stage_info_c.cmd_group[i]
        res['cmd_group'].append(convert_cmd_group_c_into_dict(temp_cmd_group))
    res['device_offset']   = stage_info_c.device_offset
    res['bdc_cmd_device']  = convert_device_mem_into_dict(stage_info_c.bdc_cmd_device)
    res['gdma_cmd_device'] = convert_device_mem_into_dict(stage_info_c.gdma_cmd_device)
    res['neuron_device']   = convert_device_mem_into_dict(stage_info_c.neuron_device)
    res['coeff']           = {"start": stage_info_c.coeff.start, "size": stage_info_c.coeff.size}
    res['coeff_addr']      = stage_info_c.coeff_addr
    res['sum_ctx_size']    = stage_info_c.sum_ctx_size
    return res

def convert_model_info_c_into_dict(model_info_c):
    res = {}
    res['model_ctx']  = model_info_c.model_ctx
    res['input_num']  = model_info_c.input_num
    res['output_num'] = model_info_c.output_num
    res['max_input']  = []
    res['max_output'] = []
    res['max_input_addr']  = []
    res['max_output_addr'] = []
    for i in range(res['input_num']):
        temp_tensor = model_info_c.input_tensor[i]
        temp_addr   = model_info_c.input_tensor_addr[i]
        res['max_input'].append(convert_tensor_c_into_dict(temp_tensor))
        res['max_input_addr'].append(convert_device_mem_into_dict(temp_addr))
        
    for i in range(res['output_num']):
        temp_tensor = model_info_c.output_tensor[i]
        temp_addr   = model_info_c.output_tensor_addr[i]
        res['max_output'].append(convert_tensor_c_into_dict(temp_tensor))
        res['max_output_addr'].append(convert_device_mem_into_dict(temp_addr))
    
    res['stage_num'] = model_info_c.stage_num
    res['stage_info'] = []
    for i in range(res['stage_num']):
        temp_stage = model_info_c.stage_info[i]
        res['stage_info'].append(convert_stage_info_c_into_dict(temp_stage))
    
    res['net_num']   = model_info_c.net_num
    res['device_id'] = model_info_c.device_id
    res['cur_net']   = model_info_c.cur_net
    res['coeff_set_num'] = model_info_c.coeff_set_num
    res['coeff_set']     = []
    for i in range(res['coeff_set_num']):
        temp_coeff_set = model_info_c.coeff_set[i]
        res['coeff_set'].append(convert_coeff_size_map_c_into_dict(temp_coeff_set))
    res['pre_alloc_io'] = model_info_c.pre_alloc_io
    res['default_map']  = model_info_c.default_map
    res['default_input_map'] = model_info_c.default_input_map
    res['default_output_map'] = model_info_c.default_output_map
    return res

dtype_map = {
    "f16": np.float16,
    "f32": np.float32
}

def filter_reorder(filter, dtype_type=2):
    shape = filter.shape
    filter = filter.reshape(-1)
    oc, ic, kh, kw = shape
    IC_PARALLEL = 64 // dtype_type
    kernel_hw = kh * kw
    new_ic = (ic+IC_PARALLEL-1) // IC_PARALLEL
    new_hw = kernel_hw * IC_PARALLEL
    filter_new = np.zeros(oc*new_ic*new_hw, dtype=filter.dtype)
    for oc_idx in range(oc):
        for ic_idx in range(new_ic):
            for k_idx in range(kernel_hw):
                for inner in range(IC_PARALLEL):
                    if ic_idx * IC_PARALLEL + inner >= ic:
                        break
                    orig_offset = (oc_idx * ic * kh * kw +
                                   (ic_idx * IC_PARALLEL + inner) * kernel_hw + k_idx)
                    trans_offset = (oc_idx * new_ic * new_hw +
                                    ic_idx * new_hw + k_idx * IC_PARALLEL + inner)
                    filter_new[trans_offset] = filter[orig_offset]
    if new_ic * new_hw > 65535 and len(shape) == 4:
        shape = [1, oc, new_ic, new_hw]
    else:
        shape = [1, oc, 1, new_ic * new_hw]
    filter_new = filter_new.reshape(shape)
    return filter_new


def model_info_into_coeff_info(model_info):
    coeff_info = {}
    coeff_info['model_ctx'] = model_info['model_ctx']
    coeff_info['coeff_num'] = model_info['coeff_set_num']
    coeff_info['coeff_map'] = {}
    coeff_info['stage'] = []
    for stage_idx in range(model_info['stage_num']):
        stage = model_info['stage_info'][stage_idx]
        coeff_info['stage'].append({"start": stage['coeff']['start'], "size": stage['coeff']['size']})
    
    for set_idx in range(model_info['coeff_set_num']):
        coeff_set = model_info['coeff_set'][set_idx]
        coeff_start = coeff_set['coeff_start_addr']
        coeff_info['coeff_map'][coeff_start] = coeff_set
    return coeff_info

class ModelBinary:
    
    def __init__(self, model_coeff_info, stage_idx=0):
        self.model_coeff_info   = model_coeff_info
        self.model_ctx          = model_coeff_info['model_ctx']
        self.bmodel_coeff_start = None
        self.all_coeff          = None
        self.coeff_size         = None
        self.read_bmodel_with_mem(stage_idx)
        
    def read_bmodel_with_mem(self, stage_idx=0):
        self.all_coeff          = get_all_coeff(self.model_ctx, stage_idx)
        self.bmodel_coeff_start = self.model_coeff_info['stage'][stage_idx]['start']
        self.coeff_size         = self.model_coeff_info['stage'][stage_idx]['size']
    
    def read_tensor_by_address(self, address, shape, dtype=np.float16, from_json=True):
        if from_json:
            address = address - MLIR_START + self.bmodel_coeff_start
        size = np.prod(shape) * np.dtype(dtype).itemsize
        data = np.ones(shape, dtype=dtype)
        read_binary_by_address( self.model_ctx, address, size, make_np2c(data))
        return data 
    
    def read_tensor_by_address_with_mem(self, address, shape, dtype=np.float16, from_json=False):
        if from_json:
            address = address - MLIR_START + self.bmodel_coeff_start
        else:
            address = address - MLIR_START
        size = np.prod(shape) * np.dtype(dtype).itemsize
        data = np.ones(shape, dtype=dtype)
        read_binary_by_address_with_mem( self.all_coeff, address, size, make_np2c(data))
        return data
    
    def write_tensor_by_address_with_mem(self, address, size, data):
        write_binary_by_address_with_mem(self.all_coeff, address, size, data)

class LoRAPlugin:
    
    def __init__(self, match_csv, model_coeff_info, stage_idx=0):
        self.model_coeff_info = model_coeff_info
        self.match_csv = match_csv
        self.match()
        self.model_binary = ModelBinary(model_coeff_info, stage_idx)
        self.start     = self.model_binary.bmodel_coeff_start
        
    def match(self):
        self.match_dict = defaultdict(list)
        with open(self.match_csv, 'r') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                line = line.split(",")
                name = line[0].strip().replace(".npy", "")
                address = int(line[1].strip())
                dtype   = dtype_map[line[2].strip()]
                shape   = [int(i) for i in line[3].strip().split("x")]
                self.match_dict[name].append(address)
                self.match_dict[name].append(dtype)
                self.match_dict[name].append(shape)

    def lora_convert_bmodel_layout(self, temp_weight, layer_name, target_shape=None, ready_reorder=False):
        if "conv" in layer_name:
            if not ready_reorder:
                temp_weight = filter_reorder(temp_weight)
            return temp_weight
        if "proj" in layer_name and "ff_net" not in layer_name and "time" not in layer_name:
            return temp_weight.reshape(target_shape)
        temp_weight = temp_weight.T
        if len(target_shape) == 4:
            temp_weight = temp_weight[np.newaxis, :, :, np.newaxis]
        if len(target_shape) == 3:
            temp_weight = temp_weight[np.newaxis, :, :]
        return temp_weight
    
    def lora_weight_save_unet_opt(self, layer, temp_weight, alpha, alpha2, ready_reorder=False):
        item = self.match_dict[layer]
        address, dtype, shape = item
        tensor = self.model_binary.read_tensor_by_address_with_mem(address, shape, dtype=dtype)
        if temp_weight.dtype != dtype:
            temp_weight = temp_weight.astype(dtype, copy=False)
        
        temp_weight = self.lora_convert_bmodel_layout(temp_weight, layer, shape, ready_reorder)
        tensor += temp_weight*alpha
        self.model_binary.write_tensor_by_address_with_mem(address - MLIR_START, np.prod(shape) * np.dtype(dtype).itemsize, make_np2c(tensor))
    
    def load_lora_weight_npz(self, lora_path, alpha2=1, ready_reorder=False):
        updates = np.load(lora_path)
        all_keys = list(i.split(".")[0] for i in updates.files)
        all_keys = list(set(all_keys))
        all_keys = list(i for i in all_keys if i in self.match_dict)
        for layer in tqdm(all_keys):
            if layer in self.match_dict:
                weight = updates[layer+".weight"]
                alpha  = 1/updates[layer+".alpha"]
                self.lora_weight_save_unet_opt(layer, weight, alpha, alpha2, ready_reorder)

class UntoolEngineOV:
    def __init__(self, model_path="", device_id=0, pre_malloc=True, default_io_mem=False, output_list=None, sg=False) :
        self.model_path = model_path
        self.sg         = sg
        if self.sg:
            self.runtime = load_bmodel_sg(model_path, pre_malloc,default_io_mem, device_id)
        else:
            self.runtime = load_bmodel(model_path, pre_malloc,default_io_mem, device_id)
        self.basic_info = self.model_info
        self.input_num  = self.basic_info['input_num']
        self.output_num = self.basic_info['output_num']
        self.inputs  = self.input_num  * [None]
        self.outputs = self.output_num * [None]
        self.cur_stage = -1
        self.init_io()
        if pre_malloc and not self.sg:
            self.check_and_move_to_device()
        self.build_input_stage_map()
        self.build_output_stage_map()
        self.output_list = output_list
        
    def __str__(self):
        return self.model_path
    
    def fill_io_max(self):
        if not self.sg:
            fill_maxio(self.runtime)
        else:
            fill_maxio_sg(self.runtime)
        self.init_io()
    
    def build_input_stage_map(self):
        input_stage_map = {}
        stage_num       = self.basic_info['stage_num']
        for input_idx in range(self.input_num):
            input_stage_map[input_idx] = {}
        for stage_idx in range(stage_num):
            for input_idx in range(self.input_num):
                shape = tuple(self.basic_info['stage_info'][stage_idx]['input_tensor'][input_idx]['data_shape']) 
                input_stage_map[input_idx][shape] = stage_idx
        self.input_stage_map = input_stage_map
    
    def build_output_stage_map(self):
        output_stage_map = {}
        stage_num = self.basic_info['stage_num']
        for output_idx in range(self.output_num):
            output_stage_map[output_idx] = {}
        for stage_idx in range(stage_num):
            for output_idx in range(self.output_num):
                shape = tuple(self.basic_info['stage_info'][stage_idx]['output_tensor'][output_idx]['data_shape']) 
                output_stage_map[output_idx][stage_idx] = shape
        self.output_stage_map = output_stage_map
        return self
    
    def default_input(self, default_input_map: dict={}, default_output_map: dict={}):
        # default_input_map: 0/1: a single value
        # if user want to set npy, please set manually
        for idx, each_input in enumerate(self.inputs):
            if each_input.is_copy:
                continue
            size               = each_input.max_size // np.dtype(type_map[each_input.dtype]).itemsize
            each_input.npy__   = np.zeros(size, dtype=type_map[each_input.dtype])
            each_input.is_from_np = True
            each_input.size    = each_input.max_size
            if idx in default_input_map:
                value = default_input_map[idx]
                each_input.npy__ += value
            each_input.data    = make_np2c(each_input.npy__)
            each_input.npu()
        
        for idx, each_output in enumerate(self.outputs):
            if each_output.is_copy:
                continue
            each_output.size = each_output.max_size // np.dtype(type_map[each_output.dtype]).itemsize
            each_output.npy__ = np.zeros(each_output.size, dtype=type_map[each_output.dtype])
            each_output.is_from_np = True
            if idx in default_output_map:
                value = default_output_map[idx]
                each_output.npy__ += value
            each_output.data = make_np2c(each_output.npy__)
            each_output.npu()
        return self

    def init_io(self):
        for i in range(self.input_num):
            tmp = self.inputs[i]
            if not self.sg:
                self.inputs[i]  = get_input_tensor(self.runtime, i).contents
            else:
                self.inputs[i]  = get_input_tensor_sg(self.runtime, i).contents
            self.inputs[i].update_extra(tmp)
        for i in range(self.output_num):
            tmp = self.outputs[i]
            if not self.sg:
                self.outputs[i] = get_output_tensor(self.runtime, i).contents
            else:
                self.outputs[i] = get_output_tensor_sg(self.runtime, i).contents
            self.outputs[i].update_extra(tmp)

    def init_output_with_np(self):
        assert self.cur_stage >= 0
        # set output shape 
        stage_info = self.basic_info['stage_info'][self.cur_stage]
        output     = stage_info['output_tensor']
        for i in range(self.output_num):
            self.outputs[i].set_dtype_shape(output[i]['data_type'], output[i]['data_shape'])
            self.outputs[i].malloc_host_addr()
    
    def set_input_with_np(self, np_array):
        """check and set stage and set shape
        link input with np
        init output with np
        """
        assert len(np_array) == self.input_num
        # check all shape is match
        shape_list = [list(i.shape) for i in np_array]
        self.check_and_set_stage(shape_list)
        for i in range(self.input_num):
            self.inputs[i].set_numpy(np_array[i])
        self.init_output_with_np()
        
    def check_and_move_to_device(self):
        check_move_to_device_fill_api(self.runtime, True)
        return self
    
    def check_and_set_stage(self, shape_lists):
        cur_info  = self.model_info
        stage     = cur_info['stage_info']
        stage_num = cur_info['stage_num']
        for i in range(stage_num):
            flag = False
            for idx in range(cur_info['input_num']):
                # check shape is match
                t_shape = stage[i]['input_tensor'][idx]['data_shape']
                s_shape = shape_lists[idx]
                if t_shape != s_shape:
                    flag = False
                    break
                else:
                    flag = True
            if flag is True:
                self.cur_stage = i
                break
        if self.cur_stage == -1:
            raise ValueError("can't find stage")
        self.set_stage(self.cur_stage)
        # set input and output shape 
        for i in range(self.input_num):
            self.inputs[i].set_dtype_shape(stage[self.cur_stage]['input_tensor'][i]['data_type'], stage[self.cur_stage]['input_tensor'][i]['data_shape'])

        for i in range(self.output_num):
            self.outputs[i].set_dtype_shape(stage[self.cur_stage]['output_tensor'][i]['data_type'], stage[self.cur_stage]['output_tensor'][i]['data_shape'])
    
    def move_input_into_device(self, force=True, copy_force=False):
        # TODO: copy_force is not used
        for i in range(self.input_num):
            temp = self.inputs[i]
            if temp.is_copy:
                pass
            else:
                temp.npu(force)

    def move_output_into_host(self, force=True):
        for i in range(self.output_num):
            temp = self.outputs[i]
            if temp.is_copy:
                pass
            else:
                temp.cpu(force)

    def load_lora_file(self, lora_npz_path, match_csv, alpha=1, stage_idx=0):
        model_coeff_info = model_info_into_coeff_info(self.basic_info)
        lora_plugin      = LoRAPlugin(match_csv, model_coeff_info, stage_idx)
        lora_plugin.load_lora_weight_npz(lora_npz_path, alpha, True)
        data_pointer     = lora_plugin.model_binary.all_coeff
        size             = lora_plugin.model_binary.coeff_size
        self.replace_coeff(data_pointer, size, stage_idx)

    @property
    def model_info(self):
        model_info_c = ModelInfoSC()
        if not self.sg:
            convert_model_info_into_c(get_model_info(self.runtime), model_info_c)
        else:
            convert_sg_model_info_into_c(get_model_sg_info(self.runtime), model_info_c)
        res = convert_model_info_c_into_dict(model_info_c)
        return res
    
    def run_copy(self, args,stage_diff_idx=0, output_list:list=[]):
        if output_list is None:
            output_list = []
        if self.output_list is not None and len(output_list) == 0:
            output_list = self.output_list

        value = []
        if isinstance(args, list):
            value = args
        elif isinstance(args, dict):
            value = list(args.values())
        self.get_stage_by_shape(value[stage_diff_idx].shape, stage_diff_idx)
        input_map = {}
        for idx in range(self.input_num):
            input_map[idx] = {"data":value[idx], "flag":0}
        res = self.run_with_np(input_map, output_list)
        return res
    
    def fake_stage_run(self, stage_idx=0):
        stage = self.basic_info['stage_info'][stage_idx]
        value = []
        for i in range(self.input_num):
            temp = stage['input_tensor'][i]
            temp_np = np.zeros(temp['data_shape']).astype(type_map[temp['data_type']])
            value.append(temp_np)
        res = self.run_copy(value, stage_diff_idx=0)
        return res
    
    def replace_coeff(self, coeff, coeff_size, stage_idx=0):
        replace_model_weight(self.runtime, coeff, coeff_size, stage_idx)
    
    def replace_model(self, bmodel_file):
        
        pass
    
    def set_stage(self, stage):
        if self.sg:
            set_stage_sg(self.runtime, stage)
        else:
            set_stage(self.runtime, stage)
        self.cur_stage = stage
    
    def get_stage_by_shape(self, shape, idx):
        if isinstance(shape, list):
            shape = tuple(shape)
        if shape not in self.input_stage_map[idx]:
            ValueError("can't find stage")
        stage = self.input_stage_map[idx][shape]
        self.set_stage(stage)
        return self
    
    def run_with_np(self, input_map_dict:dict={}, output_list:list=[], check_stage=True, stage_diff_idx=0, stage=-1) -> list:
        # input_map_dict: {idx: { "data":np, "flag":0 } }
        # output_map_dict: 
        # 这个函数假设所有的输入都已经在device上了
        # 0: force copy, 1: simple check copy , 2: force not copy 
        # get stage 
        if output_list is None:
            output_list = []
        if self.output_list is not None and len(output_list) == 0:
            output_list = self.output_list
        
        if not check_stage and stage >= 0:
            self.set_stage(stage)
        
        for k,v in input_map_dict.items():
            input_tensor = self.inputs[k]
            input_tensor.diff_set_with_flag(v['data'], v['flag'], True)
            if check_stage and k == stage_diff_idx:
                self.get_stage_by_shape(v['data'].shape, k)
        if not self.sg:
            run(self.runtime)
        else:
            run_sg(self.runtime)
        res = []
        for idx in output_list:
            self.outputs[idx].set_dtype_shape(self.basic_info['stage_info'][self.cur_stage]['output_tensor'][idx]['data_type'], self.basic_info['stage_info'][self.cur_stage]['output_tensor'][idx]['data_shape'])
            self.outputs[idx].cpu()
            res.append(self.outputs[idx].npy)
        return res

    def free_runtime(self):
        free_runtime(self.runtime)

    def __call__(self, 
                 args=None,
                 stage_diff_idx=0,
                 input_map_dict_tensor:dict=None,
                 input_map_dict_np:dict=None,
                 output_list:list=None,
                 check_stage:bool=True):
        if args is not None:
            return self.run_copy(args, stage_diff_idx=stage_diff_idx)
        if input_map_dict_np is not None:
            return self.run_with_np(input_map_dict_np, output_list=output_list, check_stage=check_stage, stage_diff_idx=stage_diff_idx)
        if input_map_dict_tensor is not None:
            return self.run_with_tensor(input_map_dict_tensor, output_list=output_list, check_stage=check_stage, stage_diff_idx=stage_diff_idx)
        return None

    def run_with_tensor(self, input_map_dict_tensor:dict={}, output_list:list=[], check_stage=True, stage_diff_idx=0):
        if output_list is None:
            output_list = []
        if self.output_list is not None and len(output_list) == 0:
            output_list = self.output_list
        
        for k,v in input_map_dict_tensor.items():
            input_tensor = self.inputs[k]
            # check address is same and same pass
            # check address is not same and copy d2d
        pass

    # def __del__(self):
    #     self.active_free()
        
    def active_free(self):
        if self.runtime:
            free_runtime(self.runtime)
        self.runtime = None
        self.inputs  = None
        self.outputs = None
        self.basic_info = None

def link_bmodel(modelsrc: UntoolEngineOV, modeldst: UntoolEngineOV, link_map:dict):
    """link two models with map (map is (i/o, idx) : (i/o, idx) )
    Args:
        modelsrc (EngineOV): source model
        modeldst (EngineOV): dest model
        link_map (dict): map is (i/o, idx) : (i/o, idx) i=0,o=1
    """
    for key, value in link_map.items():
        src_is_input = key[0] == 0
        src_idx  = key[1]
        dst_is_input = value[0] == 0
        dst_idx  = value[1]
        if src_is_input:
            if dst_is_input:
                modeldst.inputs[dst_idx].set_copy_tensor(modelsrc.inputs[src_idx])
            else:
                modeldst.outputs[dst_idx].set_copy_tensor(modelsrc.inputs[src_idx])
        else:
            if dst_is_input:
                modeldst.inputs[dst_idx].set_copy_tensor(modelsrc.outputs[src_idx])
            else:
                modeldst.outputs[dst_idx].set_copy_tensor(modelsrc.outputs[src_idx])

