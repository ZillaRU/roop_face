import ctypes 
import numpy as np 
import time 
import os
import logging

MAX_DIMS          = 8
MAX_OP_TENSOR_NUM = 10
MAX_CHAR_NUM      = 256
MAX_SHAPE_NUM     = 8
MAX_TENSOR_NUM    = 64
MAX_CMD_GROUP_NUM = 64
MAX_STAGE_NUM     = 64

#so_path = ctypes.CDLL('/usr/local/untool/lib/libuntpu.so')
# judge current os is arm or x86
so_path = None
if os.uname().machine == 'aarch64':
    so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libuntpu_arm.so'))
elif os.uname().machine == 'x86_64':
    so_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'libuntpu.so'))
lib     = ctypes.CDLL(so_path)
int_point = ctypes.POINTER(ctypes.c_int)
int_      = ctypes.c_int
ulonglong = ctypes.c_ulonglong
cpoint    = ctypes.c_void_p
vpoint    = ctypes.c_void_p
spoint    = ctypes.c_char_p
bool_     = ctypes.c_bool
null_ptr  = ctypes.c_void_p(None)
ref       = lambda x: ctypes.byref(x)

# # 将整数值转换为 void* 类型
# data_ptr = ctypes.cast(tensor.data, ctypes.c_void_p)

def make2_c_uint64_list(my_list):
    return (ctypes.c_uint64 * len(my_list))(*my_list)

def make2_c_int_list(my_list):
    return (ctypes.c_int * len(my_list))(*my_list)

def char_point_2_str(char_point):
    return ctypes.string_at(char_point).decode('utf-8')

def make_np2c(np_array):
    if np_array.flags['CONTIGUOUS'] == False:
        # info users
        np_array = np.ascontiguousarray(np_array)
    return np_array.ctypes.data_as(ctypes.c_void_p)

def str2char_point(string):
    return ctypes.c_char_p(string.encode('utf-8'))

data_type = {
    np.float32:0,
    np.float16:1,
    np.int16:4,
    np.int32:6,
    np.dtype(np.float32):0,
    np.dtype(np.float16):1,
    np.dtype(np.int16):4,
    np.dtype(np.int32):6,
    np.int8:2,
    np.dtype(np.int8):2,
    np.uint8:3,
    np.dtype(np.uint8):3,
}

type_map = {
    0: np.float32,
    1: np.float16,
    4: np.int16,
    6: np.int32,
    2: np.int8,
    3: np.uint8,
}

dtype_ctype_map = {
    np.float32: ctypes.c_float,
    np.float16: ctypes.c_uint16,
    np.int8:   ctypes.c_int8,
}

def make_c2np(data_ptr, shape, dtype):
    num = np.prod(shape)
    array_type = ctypes.cast(data_ptr, ctypes.POINTER(dtype_ctype_map[dtype]))
    np_array = np.ctypeslib.as_array(array_type, shape=(num,))
    return np_array.view(dtype=dtype).reshape(shape)
class UntensorS(ctypes.Structure):
    pass
UntensorS._fields_ = [
        ('name', ctypes.c_char * MAX_CHAR_NUM),
        ('max_size', ctypes.c_size_t),
        ('dtype', ctypes.c_int),
        ('dims', ctypes.c_size_t),
        ('shape', ctypes.c_uint64 * MAX_DIMS),
        ('size', ctypes.c_size_t),
        ('data', ctypes.c_void_p),
        ('device_id', ctypes.c_int),
        ('is_in_device', ctypes.c_bool),
        ('is_malloc_device', ctypes.c_bool),
        ('is_malloc_host', ctypes.c_bool),
        ('is_have_data', ctypes.c_bool),
        ('is_from_np', ctypes.c_bool),
        ('is_copy', ctypes.c_bool),
        ('copy_tensor', ctypes.c_void_p),
        ('device_mem_addr', ctypes.c_uint64),
        ('device_mem_size', ctypes.c_uint64),
        ('bm_handle', ctypes.c_void_p),
        ('malloc_bm_handle', ctypes.c_bool),
        ('dmabuf_fd', ctypes.c_int),
        ('reserved', ctypes.c_int),
    ]
UntensorS._fields_[14] = ('copy_tensor', ctypes.POINTER(UntensorS))
class DeviceMemSC(ctypes.Structure):
    _fields_ = [
        ('addr', ctypes.c_uint64),
        ('size', ctypes.c_size_t),
        ('dmabuf_fd', ctypes.c_int),
        ('reserve', ctypes.c_int),
    ]

class TensorSC(ctypes.Structure):
    _fields_ = [
        ('name', ctypes.c_char * MAX_CHAR_NUM),
        ('data_type', ctypes.c_uint32),
        ('gmem_stmode', ctypes.c_int32),
        ('device_addr', ctypes.c_uint64),
        ('size', ctypes.c_uint64),
        ('shape', ctypes.c_uint64 * MAX_SHAPE_NUM),
        ('mem_type', ctypes.c_uint32),
        ('scale', ctypes.c_float),
        ('cpu_addr', ctypes.c_uint32),
        ('pad_h', ctypes.c_uint32),
        ('zero_point', ctypes.c_int32),
    ]

class UserTensorMapC(ctypes.Structure):
    _fields_ = [
        ('user_compiler_addr', ctypes.c_uint64),
        ('user_device_addr', ctypes.c_uint64),
    ]

class CoeffSizeMapC(ctypes.Structure):
    _fields_ = [
        ('coeff_start_addr', ctypes.c_uint64),
        ('coeff_size', ctypes.c_uint64),
        ('neuron_size', ctypes.c_uint64),
        ('neuron_device', ctypes.c_uint64),
    ]

class BinarySC(ctypes.Structure):
    _fields_ = [
        ('start', ctypes.c_uint64),
        ('size', ctypes.c_uint64),
    ]

class CmdGroupSC(ctypes.Structure):
    _fields_ = [
        ('bdc_num', ctypes.c_uint32),
        ('gdma_num', ctypes.c_uint32),
        ('bdc_cmd', BinarySC),
        ('gdma_cmd', BinarySC),
        ('bdc_byte', ctypes.c_uint32),
        ('gdma_byte', ctypes.c_uint32),
    ]

class StageInfoSC(ctypes.Structure):
    _fields_ = [
        ('input_num', ctypes.c_size_t),
        ('output_num', ctypes.c_size_t),
        ('input_tensor', TensorSC * MAX_TENSOR_NUM),
        ('output_tensor', TensorSC * MAX_TENSOR_NUM),
        ('cmd_group_num', ctypes.c_size_t),
        ('cmd_group', CmdGroupSC * MAX_CMD_GROUP_NUM),
        ('device_offset', ctypes.c_uint64),
        ('bdc_cmd_device', DeviceMemSC),
        ('gdma_cmd_device', DeviceMemSC),
        ('neuron_device', DeviceMemSC),
        ('coeff', BinarySC),
        ('coeff_addr', ctypes.c_uint64),
        ('sum_ctx_size', ctypes.c_uint64),
    ]

class ModelInfoSC(ctypes.Structure):
    _fields_ = [
        ('model_ctx', ctypes.c_void_p),
        ('input_num', ctypes.c_size_t),
        ('output_num', ctypes.c_size_t),
        ('input_tensor', TensorSC * MAX_TENSOR_NUM),
        ('output_tensor', TensorSC * MAX_TENSOR_NUM),
        ('input_tensor_addr', DeviceMemSC * MAX_TENSOR_NUM),
        ('output_tensor_addr', DeviceMemSC * MAX_TENSOR_NUM),
        ('stage_num', ctypes.c_size_t),
        ('stage_info', StageInfoSC * MAX_STAGE_NUM),
        ('net_num', ctypes.c_size_t),
        ('device_id', ctypes.c_int),
        ('cur_net', ctypes.c_size_t),
        ('coeff_set_num', ctypes.c_size_t),
        ('coeff_set', CoeffSizeMapC * MAX_STAGE_NUM),
        ('pre_alloc_io', ctypes.c_bool),
        ('default_map', ctypes.c_bool),
        ('default_input_map', ctypes.c_bool),
        ('default_output_map', ctypes.c_bool),
    ]



lib.find_optimal.restype  = ctypes.POINTER(UntensorS)
lib.find_optimal.argtypes = [ctypes.POINTER(UntensorS)]
def find_optimal(tensor) -> ctypes.POINTER(UntensorS):
    """
    untensor find_optimal(untensor tensor);
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.find_optimal(ctypes.byref(tensor))

lib.copy_tensor.restype  = None
lib.copy_tensor.argtypes = [ctypes.POINTER(UntensorS), ctypes.POINTER(UntensorS)]
def copy_tensor(src, dst) -> None:
    """
    void copy_tensor (untensor src, untensor dst);
    :param src: 	ctypes.POINTER(UntensorS)
    :param dst: 	ctypes.POINTER(UntensorS)
    """
    return lib.copy_tensor(ctypes.byref(src), ctypes.byref(dst))

lib.copy_tensor_to_device.restype  = None
lib.copy_tensor_to_device.argtypes = [ctypes.POINTER(UntensorS), ctypes.c_bool]
def copy_tensor_to_device(tensor, force) -> None:
    """
    void copy_tensor_to_device(untensor tensor, bool force);
    :param tensor: 	ctypes.POINTER(UntensorS)
    :param force: 	ctypes.c_bool
    """
    return lib.copy_tensor_to_device(ctypes.byref(tensor), force)

lib.set_data.restype  = None
lib.set_data.argtypes = [ctypes.POINTER(UntensorS), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_bool]
def set_data(tensor, data, size, is_copy) -> None:
    """
    void set_data(untensor tensor, void* data, size_t size, bool is_copy);
    :param tensor: 	ctypes.POINTER(UntensorS)
    :param data: 	ctypes.c_void_p
    :param size: 	ctypes.c_size_t
    :param is_copy: 	ctypes.c_bool
    """
    return lib.set_data(ctypes.byref(tensor), data, ctypes.c_size_t(size), is_copy)

lib.copy_tensor_to_host.restype  = None
lib.copy_tensor_to_host.argtypes = [ctypes.POINTER(UntensorS), ctypes.c_bool]
def copy_tensor_to_host(tensor, force) -> None:
    """
    void copy_tensor_to_host(untensor tensor, bool force);
    :param tensor: 	ctypes.POINTER(UntensorS)
    :param force: 	ctypes.c_bool
    """
    return lib.copy_tensor_to_host(ctypes.byref(tensor), force)

lib.show_device_data.restype  = None
lib.show_device_data.argtypes = [ctypes.POINTER(UntensorS), ctypes.c_int, ctypes.c_int]
def show_device_data(tensor, start, len) -> None:
    """
    void show_device_data(untensor tensor, int start, int len);
    :param tensor: 	ctypes.POINTER(UntensorS)
    :param start: 	ctypes.c_int
    :param len: 	ctypes.c_int
    """
    return lib.show_device_data(ctypes.byref(tensor), ctypes.c_int(start), ctypes.c_int(len))

lib.tensor_shape_is_same.restype  = ctypes.c_bool
lib.tensor_shape_is_same.argtypes = [ctypes.POINTER(UntensorS), ctypes.POINTER(UntensorS)]
def tensor_shape_is_same(tensor1, tensor2) -> ctypes.c_bool:
    """
    bool tensor_shape_is_same(untensor tensor1, untensor tensor2);
    :param tensor1: 	ctypes.POINTER(UntensorS)
    :param tensor2: 	ctypes.POINTER(UntensorS)
    """
    return lib.tensor_shape_is_same(ctypes.byref(tensor1), ctypes.byref(tensor2))

lib.create_tensor.restype  = ctypes.POINTER(UntensorS)
lib.create_tensor.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint64), ctypes.c_int, ctypes.c_bool, ctypes.c_int]
def create_tensor(dims, shape, dtype, is_malloc, device_id) -> ctypes.POINTER(UntensorS):
    """
    untensor create_tensor(int dims, uint64_t * shape, int dtype, bool is_malloc, int device_id);
    :param dims: 	ctypes.c_int
    :param shape: 	ctypes.POINTER(ctypes.c_uint64)
    :param dtype: 	ctypes.c_int
    :param is_malloc: 	ctypes.c_bool
    :param device_id: 	ctypes.c_int
    """
    return lib.create_tensor(ctypes.c_int(dims), make2_c_uint64_list(shape), ctypes.c_int(dtype), is_malloc, ctypes.c_int(device_id))

lib.free_data.restype  = None
lib.free_data.argtypes = [ctypes.POINTER(UntensorS)]
def free_data(tensor) -> None:
    """
    void free_data(untensor tensor);
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.free_data(ctypes.byref(tensor))

lib.get_tensor_device_address.restype  = ctypes.c_uint64
lib.get_tensor_device_address.argtypes = [ctypes.POINTER(UntensorS)]
def get_tensor_device_address(tensor) -> ctypes.c_uint64:
    """
    u64 get_tensor_device_address(untensor tensor);
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.get_tensor_device_address(ctypes.byref(tensor))

lib.malloc_host_data.restype  = None
lib.malloc_host_data.argtypes = [ctypes.POINTER(UntensorS)]
def malloc_host_data(tensor) -> None:
    """
    void malloc_host_data(untensor tensor);
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.malloc_host_data(ctypes.byref(tensor))

lib.show_device_mem_data.restype  = None
lib.show_device_mem_data.argtypes = [ctypes.c_int, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def show_device_mem_data(device_id, address, size, source_dtype, start, len) -> None:
    """
    void show_device_mem_data(int device_id, u64 address, size_t size, int source_dtype, int start, int len);
    :param device_id: 	ctypes.c_int
    :param address: 	ctypes.c_uint64
    :param size: 	ctypes.c_size_t
    :param source_dtype: 	ctypes.c_int
    :param start: 	ctypes.c_int
    :param len: 	ctypes.c_int
    """
    return lib.show_device_mem_data(ctypes.c_int(device_id), ctypes.c_uint64(address), ctypes.c_size_t(size), ctypes.c_int(source_dtype), ctypes.c_int(start), ctypes.c_int(len))

lib.malloc_device.restype  = None
lib.malloc_device.argtypes = [ctypes.POINTER(UntensorS)]
def malloc_device(tensor) -> None:
    """
    void malloc_device(untensor tensor);
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.malloc_device(ctypes.byref(tensor))

lib.copy_data_from_tensor_with_flag.restype  = None
lib.copy_data_from_tensor_with_flag.argtypes = [ctypes.POINTER(UntensorS), ctypes.POINTER(UntensorS), ctypes.c_int]
def copy_data_from_tensor_with_flag(src, dst, flag) -> None:
    """
    void copy_data_from_tensor_with_flag(untensor src, untensor dst, int flag);
    :param src: 	ctypes.POINTER(UntensorS)
    :param dst: 	ctypes.POINTER(UntensorS)
    :param flag: 	ctypes.c_int
    """
    return lib.copy_data_from_tensor_with_flag(ctypes.byref(src), ctypes.byref(dst), ctypes.c_int(flag))


lib.load_bmodel.restype  = ctypes.c_void_p
lib.load_bmodel.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_int]
def load_bmodel(bmodel_path, pre_malloc, default_io_mem, device_id) -> ctypes.c_void_p:
    """
    struct un_runtime_s * load_bmodel(const char* bmodel_path, bool pre_malloc, bool default_io_mem, int device_id);
    :param bmodel_path: 	ctypes.c_char_p
    :param pre_malloc: 	ctypes.c_bool
    :param default_io_mem: 	ctypes.c_bool
    :param device_id: 	ctypes.c_int
    """
    return lib.load_bmodel(str2char_point(bmodel_path), pre_malloc, default_io_mem, ctypes.c_int(device_id))

lib.set_input_tensor.restype  = None
lib.set_input_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(UntensorS)]
def set_input_tensor(runtime, index, tensor) -> None:
    """
    void set_input_tensor (struct un_runtime_s * runtime, int index, untensor tensor);
    :param runtime: 	ctypes.c_void_p
    :param index: 	ctypes.c_int
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.set_input_tensor(runtime, ctypes.c_int(index), ctypes.byref(tensor))

lib.set_output_tensor.restype  = None
lib.set_output_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(UntensorS)]
def set_output_tensor(runtime, index, tensor) -> None:
    """
    void set_output_tensor(struct un_runtime_s * runtime, int index, untensor tensor);
    :param runtime: 	ctypes.c_void_p
    :param index: 	ctypes.c_int
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.set_output_tensor(runtime, ctypes.c_int(index), ctypes.byref(tensor))


lib.init_io_tensor.restype  = None
lib.init_io_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
def init_io_tensor(runtime, stage_dix) -> None:
    """
    void init_io_tensor (struct un_runtime_s * runtime, int stage_dix);
    :param runtime: 	ctypes.c_void_p
    :param stage_dix: 	ctypes.c_int
    """
    return lib.init_io_tensor(runtime, ctypes.c_int(stage_dix))


lib.fill_io_tensor.restype  = None
lib.fill_io_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
def fill_io_tensor(runtime, stage_idx) -> None:
    """
    void fill_io_tensor (struct un_runtime_s * runtime, int stage_idx);
    :param runtime: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_int
    """
    return lib.fill_io_tensor(runtime, ctypes.c_int(stage_idx))


lib.fill_maxio.restype  = None
lib.fill_maxio.argtypes = [ctypes.c_void_p]
def fill_maxio(runtime) -> None:
    """
    void fill_maxio(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.fill_maxio(runtime)


lib.check_all_data.restype  = ctypes.c_bool
lib.check_all_data.argtypes = [ctypes.c_void_p]
def check_all_data(runtime) -> ctypes.c_bool:
    """
    bool check_all_data(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.check_all_data(runtime)

lib.copy_input_to_device.restype  = None
lib.copy_input_to_device.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def copy_input_to_device(runtime, force) -> None:
    """
    void copy_input_to_device(struct un_runtime_s * runtime, bool force);
    :param runtime: 	ctypes.c_void_p
    :param force: 	ctypes.c_bool
    """
    return lib.copy_input_to_device(runtime, force)

lib.copy_output_to_host.restype  = None
lib.copy_output_to_host.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def copy_output_to_host(runtime, force) -> None:
    """
    void copy_output_to_host(struct un_runtime_s * runtime, bool force);
    :param runtime: 	ctypes.c_void_p
    :param force: 	ctypes.c_bool
    """
    return lib.copy_output_to_host(runtime, force)

lib.check_move_to_device_fill_api.restype  = None
lib.check_move_to_device_fill_api.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def check_move_to_device_fill_api(runtime, default_move_coeff) -> None:
    """
    void check_move_to_device_fill_api(struct un_runtime_s * runtime, bool default_move_coeff);
    :param runtime: 	ctypes.c_void_p
    :param default_move_coeff: 	ctypes.c_bool
    """
    return lib.check_move_to_device_fill_api(runtime, default_move_coeff)

lib.run.restype  = None
lib.run.argtypes = [ctypes.c_void_p]
def run(runtime) -> None:
    """
    void run(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.run(runtime)

lib.stage_match_io.restype  = ctypes.c_bool
lib.stage_match_io.argtypes = [ctypes.c_void_p, ctypes.c_int]
def stage_match_io(runtime, stage_index) -> ctypes.c_bool:
    """
    bool stage_match_io(struct un_runtime_s * runtime, int stage_index);
    :param runtime: 	ctypes.c_void_p
    :param stage_index: 	ctypes.c_int
    """
    return lib.stage_match_io(runtime, ctypes.c_int(stage_index))

lib.free_runtime.restype  = None
lib.free_runtime.argtypes = [ctypes.c_void_p]
def free_runtime(runtime) -> None:
    """
    void free_runtime(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.free_runtime(runtime)

lib.delete_runtime.restype  = None
lib.delete_runtime.argtypes = [ctypes.c_void_p]
def delete_runtime(runtime) -> None:
    """
    void delete_runtime(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.delete_runtime(runtime)

lib.load_bmodel_sg.restype  = ctypes.c_void_p
lib.load_bmodel_sg.argtypes = [ctypes.c_char_p, ctypes.c_bool, ctypes.c_bool, ctypes.c_int]
def load_bmodel_sg(bmodel_path, pre_malloc, default_io_mem, device_id) -> ctypes.c_void_p:
    """
    un_runtime_sg_s* load_bmodel_sg(const char* bmodel_path, bool pre_malloc, bool default_io_mem, int device_id);
    :param bmodel_path: 	ctypes.c_char_p
    :param pre_malloc: 	ctypes.c_bool
    :param default_io_mem: 	ctypes.c_bool
    :param device_id: 	ctypes.c_int
    """
    return lib.load_bmodel_sg(str2char_point(bmodel_path), pre_malloc, default_io_mem, ctypes.c_int(device_id))

lib.run_sg.restype  = None
lib.run_sg.argtypes = [ctypes.c_void_p]
def run_sg(runtime) -> None:
    """
    void run_sg(struct un_runtime_sg_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.run_sg(runtime)

lib.fill_maxio_sg.restype  = None
lib.fill_maxio_sg.argtypes = [ctypes.c_void_p]
def fill_maxio_sg(runtime) -> None:
    """
    void fill_maxio_sg(struct un_runtime_sg_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.fill_maxio_sg(runtime)

lib.set_stage_sg.restype  = None
lib.set_stage_sg.argtypes = [ctypes.c_void_p, ctypes.c_int]
def set_stage_sg(runtime, stage_idx) -> None:
    """
    void set_stage_sg(struct un_runtime_sg_s * runtime, int stage_idx);
    :param runtime: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_int
    """
    return lib.set_stage_sg(runtime, ctypes.c_int(stage_idx))

lib.free_runtime_sg.restype  = None
lib.free_runtime_sg.argtypes = [ctypes.c_void_p]
def free_runtime_sg(runtime) -> None:
    """
    void free_runtime_sg(struct un_runtime_sg_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.free_runtime_sg(runtime)

lib.init_tensor_sg.restype  = None
lib.init_tensor_sg.argtypes = [ctypes.c_void_p, ctypes.c_int]
def init_tensor_sg(runtime, stage_idx) -> None:
    """
    void init_tensor_sg(struct un_runtime_sg_s * runtime, int stage_idx);
    :param runtime: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_int
    """
    return lib.init_tensor_sg(runtime, ctypes.c_int(stage_idx))

lib.get_model_sg_info.restype  = ctypes.c_void_p
lib.get_model_sg_info.argtypes = [ctypes.c_void_p]
def get_model_sg_info(runtime) -> ctypes.c_void_p:
    """
    model_info_sg* get_model_sg_info(struct un_runtime_sg_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.get_model_sg_info(runtime)

lib.get_input_tensor_sg.restype  = ctypes.POINTER(UntensorS)
lib.get_input_tensor_sg.argtypes = [ctypes.c_void_p, ctypes.c_int]
def get_input_tensor_sg(runtime, index) -> ctypes.POINTER(UntensorS):
    """
    untensor get_input_tensor_sg(struct un_runtime_sg_s * runtime, int index);
    :param runtime: 	ctypes.c_void_p
    :param index: 	ctypes.c_int
    """
    return lib.get_input_tensor_sg(runtime, ctypes.c_int(index))

lib.get_output_tensor_sg.restype  = ctypes.POINTER(UntensorS)
lib.get_output_tensor_sg.argtypes = [ctypes.c_void_p, ctypes.c_int]
def get_output_tensor_sg(runtime, index) -> ctypes.POINTER(UntensorS):
    """
    untensor get_output_tensor_sg(struct un_runtime_sg_s * runtime, int index);
    :param runtime: 	ctypes.c_void_p
    :param index: 	ctypes.c_int
    """
    return lib.get_output_tensor_sg(runtime, ctypes.c_int(index))


lib.replace_model_weight.restype  = None
lib.replace_model_weight.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
def replace_model_weight(runtime, new_coeff_data, size, stage_index) -> None:
    """
    void replace_model_weight(struct un_runtime_s * runtime, void* new_coeff_data, size_t size, int stage_index);
    :param runtime: 	ctypes.c_void_p
    :param new_coeff_data: 	ctypes.c_void_p
    :param size: 	ctypes.c_size_t
    :param stage_index: 	ctypes.c_int
    """
    return lib.replace_model_weight(runtime, new_coeff_data, ctypes.c_size_t(size), ctypes.c_int(stage_index))

lib.get_model_info.restype  = ctypes.c_void_p
lib.get_model_info.argtypes = [ctypes.c_void_p]
def get_model_info(runtime) -> ctypes.c_void_p:
    """
    struct model_info_s * get_model_info(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.get_model_info(runtime)

lib.get_input_tensor.restype  = ctypes.POINTER(UntensorS)
lib.get_input_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
def get_input_tensor(runtime, index) -> ctypes.POINTER(UntensorS):
    """
    untensor get_input_tensor(struct un_runtime_s * runtime, int index);
    :param runtime: 	ctypes.c_void_p
    :param index: 	ctypes.c_int
    """
    return lib.get_input_tensor(runtime, ctypes.c_int(index))

lib.get_output_tensor.restype  = ctypes.POINTER(UntensorS)
lib.get_output_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
def get_output_tensor(runtime, index) -> ctypes.POINTER(UntensorS):
    """
    untensor get_output_tensor(struct un_runtime_s * runtime, int index);
    :param runtime: 	ctypes.c_void_p
    :param index: 	ctypes.c_int
    """
    return lib.get_output_tensor(runtime, ctypes.c_int(index))

lib.set_stage.restype  = None
lib.set_stage.argtypes = [ctypes.c_void_p, ctypes.c_int]
def set_stage(runtime, stage_idx) -> None:
    """
    void set_stage(struct un_runtime_s * runtime, int stage_idx);
    :param runtime: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_int
    """
    return lib.set_stage(runtime, ctypes.c_int(stage_idx))

lib.free_handle.restype  = None
lib.free_handle.argtypes = [ctypes.c_void_p]
def free_handle(bm_handle) -> None:
    """
    void free_handle(bm_handle_t bm_handle)
    :param bm_handle: 	ctypes.c_void_p
    """
    return lib.free_handle(bm_handle)

lib.get_device_status.restype  = ctypes.c_int
lib.get_device_status.argtypes = [ctypes.c_int]
def get_device_status(device_id) -> ctypes.c_int:
    """
    int get_device_status(int device_id)
    :param device_id: 	ctypes.c_int
    """
    return lib.get_device_status(ctypes.c_int(device_id))

lib.get_handle.restype  = ctypes.c_void_p
lib.get_handle.argtypes = [ctypes.c_int]
def get_handle(device_id) -> ctypes.c_void_p:
    """
    bm_handle_t get_handle(int device_id)
    :param device_id: 	ctypes.c_int
    """
    return lib.get_handle(ctypes.c_int(device_id))

lib.read_bmodel_ctx.restype  = ctypes.c_void_p
lib.read_bmodel_ctx.argtypes = [ctypes.c_char_p]
def read_bmodel_ctx(filename) -> ctypes.c_void_p:
    """
    ModelCtx* read_bmodel_ctx(const char* filename);
    :param filename: 	ctypes.c_char_p
    """
    return lib.read_bmodel_ctx(str2char_point(filename))

lib.free_bmodel_ctx.restype  = None
lib.free_bmodel_ctx.argtypes = [ctypes.c_void_p]
def free_bmodel_ctx(model_ctx) -> None:
    """
    void free_bmodel_ctx(ModelCtx * model_ctx);
    :param model_ctx: 	ctypes.c_void_p
    """
    return lib.free_bmodel_ctx(model_ctx)

lib.read_binary_by_address.restype  = None
lib.read_binary_by_address.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
def read_binary_by_address(model_ctx, address, size, data) -> None:
    """
    void read_binary_by_address(ModelCtx * model_ctx, uint64_t address, size_t size, void* data);
    :param model_ctx: 	ctypes.c_void_p
    :param address: 	ctypes.c_uint64
    :param size: 	ctypes.c_size_t
    :param data: 	ctypes.c_void_p
    """
    return lib.read_binary_by_address(model_ctx, ctypes.c_uint64(address), ctypes.c_size_t(size), data)

lib.read_coeff_binary_address.restype  = ctypes.c_uint64
lib.read_coeff_binary_address.argtypes = [ctypes.c_void_p]
def read_coeff_binary_address(model_ctx) -> ctypes.c_uint64:
    """
    uint64_t read_coeff_binary_address(ModelCtx * model_ctx);
    :param model_ctx: 	ctypes.c_void_p
    """
    return lib.read_coeff_binary_address(model_ctx)

lib.get_all_coeff.restype  = ctypes.c_void_p
lib.get_all_coeff.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
def get_all_coeff(model_ctx, stage_id) -> ctypes.c_void_p:
    """
    void * get_all_coeff(ModelCtx * model_ctx, size_t stage_id);
    :param model_ctx: 	ctypes.c_void_p
    :param stage_id: 	ctypes.c_size_t
    """
    return lib.get_all_coeff(model_ctx, ctypes.c_size_t(stage_id))

lib.read_binary_by_address_with_mem.restype  = None
lib.read_binary_by_address_with_mem.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
def read_binary_by_address_with_mem(data, address, size, data2) -> None:
    """
    void read_binary_by_address_with_mem(void* data, uint64_t address, size_t size, void* data2);
    :param data: 	ctypes.c_void_p
    :param address: 	ctypes.c_uint64
    :param size: 	ctypes.c_size_t
    :param data2: 	ctypes.c_void_p
    """
    return lib.read_binary_by_address_with_mem(data, ctypes.c_uint64(address), ctypes.c_size_t(size), data2)

lib.write_binary_by_address_with_mem.restype  = None
lib.write_binary_by_address_with_mem.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
def write_binary_by_address_with_mem(dest_data, address, size, src_data) -> None:
    """
    void write_binary_by_address_with_mem(void* dest_data, uint64_t address, size_t size, void* src_data);
    :param dest_data: 	ctypes.c_void_p
    :param address: 	ctypes.c_uint64
    :param size: 	ctypes.c_size_t
    :param src_data: 	ctypes.c_void_p
    """
    return lib.write_binary_by_address_with_mem(dest_data, ctypes.c_uint64(address), ctypes.c_size_t(size), src_data)

lib.convert_model_info_into_c.restype  = None
lib.convert_model_info_into_c.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModelInfoSC)]
def convert_model_info_into_c(model_info, model_info_c) -> None:
    """
    void convert_model_info_into_c(struct model_info_s* model_info, struct model_info_s_c * model_info_c);
    :param model_info: 	ctypes.c_void_p
    :param model_info_c: 	ctypes.POINTER(ModelInfoSC)
    """
    return lib.convert_model_info_into_c(model_info, ctypes.byref(model_info_c))

lib.convert_sg_model_info_into_c.restype  = None
lib.convert_sg_model_info_into_c.argtypes = [ctypes.c_void_p, ctypes.POINTER(ModelInfoSC)]
def convert_sg_model_info_into_c(model_info, model_info_c) -> None:
    """
    void convert_sg_model_info_into_c(model_info_sg* model_info, struct model_info_s_c * model_info_c);
    :param model_info: 	ctypes.c_void_p
    :param model_info_c: 	ctypes.POINTER(ModelInfoSC)
    """
    return lib.convert_sg_model_info_into_c(model_info, ctypes.byref(model_info_c))

lib.free_tensor.restype  = None
lib.free_tensor.argtypes = [ctypes.POINTER(UntensorS)]
def free_tensor(tensor) -> None:
    """
    void free_tensor(untensor tensor);
    :param tensor: 	ctypes.POINTER(UntensorS)
    """
    return lib.free_tensor(ctypes.byref(tensor))

lib.get_model_info.restype  = ctypes.c_void_p
lib.get_model_info.argtypes = [ctypes.c_void_p]
def get_model_info(runtime) -> ctypes.c_void_p:
    """
    struct model_info_s * get_model_info(struct un_runtime_s * runtime);
    :param runtime: 	ctypes.c_void_p
    """
    return lib.get_model_info(runtime)

lib.print_data_by_fp32.restype  = None
lib.print_data_by_fp32.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
def print_data_by_fp32(data, size, dtype, start, len) -> None:
    """
    void print_data_by_fp32(void* data, int size, int dtype, int start, int len);
    :param data: 	ctypes.c_void_p
    :param size: 	ctypes.c_int
    :param dtype: 	ctypes.c_int
    :param start: 	ctypes.c_int
    :param len: 	ctypes.c_int
    """
    return lib.print_data_by_fp32(data, ctypes.c_int(size), ctypes.c_int(dtype), ctypes.c_int(start), ctypes.c_int(len))

lib.data_convert_to_fp32.restype  = None
lib.data_convert_to_fp32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
def data_convert_to_fp32(src, target, dtype, size) -> None:
    """
    void data_convert_to_fp32(void* src, void* target, int dtype, size_t size);
    :param src: 	ctypes.c_void_p
    :param target: 	ctypes.c_void_p
    :param dtype: 	ctypes.c_int
    :param size: 	ctypes.c_size_t
    """
    return lib.data_convert_to_fp32(src, target, ctypes.c_int(dtype), ctypes.c_size_t(size))

lib.convert_to_fp32.restype  = ctypes.c_float
lib.convert_to_fp32.argtypes = [ctypes.c_void_p, ctypes.c_int]
def convert_to_fp32(source, dtype) -> ctypes.c_float:
    """
    float convert_to_fp32(void* source, int dtype);
    :param source: 	ctypes.c_void_p
    :param dtype: 	ctypes.c_int
    """
    return lib.convert_to_fp32(source, ctypes.c_int(dtype))


lib.premalloc_input_output_addr.restype  = None
lib.premalloc_input_output_addr.argtypes = [ctypes.c_void_p]
def premalloc_input_output_addr(model_info_p) -> None:
    """
    void premalloc_input_output_addr(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.premalloc_input_output_addr(model_info_p)

lib.set_model_info_user_map.restype  = None
lib.set_model_info_user_map.argtypes = [ctypes.c_void_p]
def set_model_info_user_map(model_info_p) -> None:
    """
    void set_model_info_user_map(struct model_info_s * model_info_p);// 设置 input output addr
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.set_model_info_user_map(model_info_p)

lib.set_max_input_output.restype  = None
lib.set_max_input_output.argtypes = [ctypes.c_void_p]
def set_max_input_output(model_info_p) -> None:
    """
    void set_max_input_output(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.set_max_input_output(model_info_p)


lib.move_to_device.restype  = None
lib.move_to_device.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def move_to_device(model_info_p, default_move_coeff) -> None:
    """
    void move_to_device(struct model_info_s * model_info_p, bool default_move_coeff);
    :param model_info_p: 	ctypes.c_void_p
    :param default_move_coeff: 	ctypes.c_bool
    """
    return lib.move_to_device(model_info_p, default_move_coeff)

lib.convert_bdc_cmd.restype  = None
lib.convert_bdc_cmd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_bool]
def convert_bdc_cmd(model_info_p, cmd, is_last) -> None:
    """
    void convert_bdc_cmd(struct model_info_s * model_info_p, u32* cmd, bool is_last);// 转换 bdc cmd
    :param model_info_p: 	ctypes.c_void_p
    :param cmd: 	ctypes.POINTER(ctypes.c_uint32)
    :param is_last: 	ctypes.c_bool
    """
    return lib.convert_bdc_cmd(model_info_p, cmd, is_last)

lib.get_bdc_cmd_len.restype  = ctypes.c_uint32
lib.get_bdc_cmd_len.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool]
def get_bdc_cmd_len(model_ctx, binary_cmd, start_offset, last_cmd) -> ctypes.c_uint32:
    """
    uint32_t get_bdc_cmd_len(ModelCtx* model_ctx, const Binary* binary_cmd, u64 start_offset, bool last_cmd);
    :param model_ctx: 	ctypes.c_void_p
    :param binary_cmd: 	ctypes.c_void_p
    :param start_offset: 	ctypes.c_uint64
    :param last_cmd: 	ctypes.c_bool
    """
    return lib.get_bdc_cmd_len(model_ctx, binary_cmd, ctypes.c_uint64(start_offset), last_cmd)

lib.get_gdma_cmd_len.restype  = ctypes.c_uint32
lib.get_gdma_cmd_len.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_bool]
def get_gdma_cmd_len(model_ctx, binary_cmd, start_offset, last_cmd) -> ctypes.c_uint32:
    """
    uint32_t get_gdma_cmd_len(ModelCtx* model_ctx, const Binary* binary_cmd, u64 start_offset, bool last_cmd);
    :param model_ctx: 	ctypes.c_void_p
    :param binary_cmd: 	ctypes.c_void_p
    :param start_offset: 	ctypes.c_uint64
    :param last_cmd: 	ctypes.c_bool
    """
    return lib.get_gdma_cmd_len(model_ctx, binary_cmd, ctypes.c_uint64(start_offset), last_cmd)

lib.convert_gdma_cmd.restype  = None
lib.convert_gdma_cmd.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.c_bool, ctypes.c_size_t]
def convert_gdma_cmd(model_info_p, cmd, is_last, stage_idx) -> None:
    """
    void convert_gdma_cmd(struct model_info_s * model_info_p, u32* cmd, bool is_last, size_t stage_idx);
    :param model_info_p: 	ctypes.c_void_p
    :param cmd: 	ctypes.POINTER(ctypes.c_uint32)
    :param is_last: 	ctypes.c_bool
    :param stage_idx: 	ctypes.c_size_t
    """
    return lib.convert_gdma_cmd(model_info_p, cmd, is_last, ctypes.c_size_t(stage_idx))

lib.bdc_gdma_move_to_device.restype  = None
lib.bdc_gdma_move_to_device.argtypes = [ctypes.c_void_p]
def bdc_gdma_move_to_device(model_info_p) -> None:
    """
    void bdc_gdma_move_to_device(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.bdc_gdma_move_to_device(model_info_p)

lib.stage_bdc_gdma_move_to_device.restype  = None
lib.stage_bdc_gdma_move_to_device.argtypes = [ctypes.c_void_p, ctypes.c_int]
def stage_bdc_gdma_move_to_device(model_info_p, stage_idx) -> None:
    """
    void stage_bdc_gdma_move_to_device(struct model_info_s * model_info_p, int stage_idx);
    :param model_info_p: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_int
    """
    return lib.stage_bdc_gdma_move_to_device(model_info_p, ctypes.c_int(stage_idx))


lib.malloc_coeff_neuron_on_device_set_offset.restype  = None
lib.malloc_coeff_neuron_on_device_set_offset.argtypes = [ctypes.c_void_p]
def malloc_coeff_neuron_on_device_set_offset(model_info_p) -> None:
    """
    void malloc_coeff_neuron_on_device_set_offset(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.malloc_coeff_neuron_on_device_set_offset(model_info_p)

lib.set_coeff_neuron_map.restype  = None
lib.set_coeff_neuron_map.argtypes = [ctypes.c_void_p]
def set_coeff_neuron_map(model_info_p) -> None:
    """
    void set_coeff_neuron_map(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.set_coeff_neuron_map(model_info_p)

lib.coeff_to_device.restype  = None
lib.coeff_to_device.argtypes = [ctypes.c_void_p]
def coeff_to_device(model_info_p) -> None:
    """
    void coeff_to_device(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.coeff_to_device(model_info_p)


lib.load_kernel_module.restype  = None
lib.load_kernel_module.argtypes = [ctypes.c_void_p]
def load_kernel_module(model_info_p) -> None:
    """
    void load_kernel_module(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.load_kernel_module(model_info_p)


lib.bm1684x_fill_api_info.restype  = None
lib.bm1684x_fill_api_info.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
def bm1684x_fill_api_info(model_info_p, api_info_p, stage_idx) -> None:
    """
    void bm1684x_fill_api_info(struct model_info_s * model_info_p, struct api_info_t * api_info_p, int stage_idx);
    :param model_info_p: 	ctypes.c_void_p
    :param api_info_p: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_int
    """
    return lib.bm1684x_fill_api_info(model_info_p, api_info_p, ctypes.c_int(stage_idx))

lib.fill_api_info.restype  = None
lib.fill_api_info.argtypes = [ctypes.c_void_p]
def fill_api_info(model_info_p) -> None:
    """
    void fill_api_info(struct model_info_s * model_info_p);
    :param model_info_p: 	ctypes.c_void_p
    """
    return lib.fill_api_info(model_info_p)

lib.run_model.restype  = None
lib.run_model.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
def run_model(model_info_p, stage_idx) -> None:
    """
    void run_model(struct model_info_s * model_info_p, size_t stage_idx);
    :param model_info_p: 	ctypes.c_void_p
    :param stage_idx: 	ctypes.c_size_t
    """
    return lib.run_model(model_info_p, ctypes.c_size_t(stage_idx))

lib.free_model.restype  = None
lib.free_model.argtypes = [ctypes.c_void_p, ctypes.c_bool]
def free_model(model_info_p, free_input_output) -> None:
    """
    void free_model(struct model_info_s * model_info_p, bool free_input_output);
    :param model_info_p: 	ctypes.c_void_p
    :param free_input_output: 	ctypes.c_bool
    """
    return lib.free_model(model_info_p, free_input_output)
version='2023-12-21-14-32-11'
