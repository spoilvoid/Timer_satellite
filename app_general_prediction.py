import random
import numpy as np
import streamlit as st
import torch
import os
import argparse
from datetime import datetime
import multiprocessing as mp
import atexit
import os, subprocess, socket
import streamlit.components.v1 as components

from exp.exp_forecast import Exp_Forecast
from utils.tools import HiddenPrints


def get_available_devices():
    """检测并返回可用的PyTorch设备列表"""
    devices = ["cpu"]
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(i)
    return devices

@st.cache_resource
def launch_tensorboard(logdir):
    """
    启动一个独立的 TensorBoard 进程，并确保它在 Streamlit 退出时被清理。
    使用 @st.cache_resource 确保此函数在整个应用生命周期中只运行一次。
    """
    with socket.socket() as s:
        s.bind(('', 0))
        port = s.getsockname()[1]
    
    print(f"正在端口 {port} 上启动 TensorBoard...")
    command = [
        "tensorboard",
        "--logdir", logdir,
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    p = subprocess.Popen(command)
    
    atexit.register(p.terminate)
    
    return port

def preiction_process_wrapper(params: dict):
    """
    接收一个包含所有超参数的字典。
    """
    try:
        args = argparse.Namespace(
            # ===== basic config =====
            task_name="forecast",
            model_id="Timer_multivariate_forecast",
            model="Timer_multivariate",
            seed=42,

            # ===== data loader =====
            data="multivariate",
            root_path="./dataset/xw/elec",
            data_path="ETTh1.csv",
            features="M",
            target="OT",
            freq="h",
            checkpoints="./checkpoints/",
            inverse=False,

            # ===== model define =====
            d_model=1024,
            n_heads=8,
            e_layers=8,
            d_layers=1,
            d_ff=2048,
            factor=3,
            distil=True,
            dropout=0.1,
            embed="timeF",
            activation="gelu",
            output_attention=False,
            use_norm=True,
            max_len=10000,
            mask_flag=True,
            binary_bias=False,
            covariate=True,
            n_pred_vars=15,
            freeze_layer=False,

            # ===== optimization =====
            num_workers=4,
            itr=1,
            train_epochs=10,
            batch_size=32,
            patience=3,
            learning_rate=3e-5,
            des="Exp",
            loss="MSE",
            lradj="type1",
            use_amp=False,

            # ===== GPU =====
            use_gpu=True,
            gpu=0,
            use_multi_gpu=False,
            devices="0,1,2,3",

            # ===== misc =====
            stride=1,
            ckpt_path="checkpoints/Timer_forecast_1.0.ckpt",
            finetune_epochs=10,
            finetune_rate=0.1,
            local_rank=0,

            patch_len=96,
            subset_rand_ratio=1.0,
            data_type="custom",

            decay_fac=0.75,

            # ===== cosine decay =====
            cos_warm_up_steps=100,
            cos_max_decay_steps=60000,
            cos_max_decay_epoch=10,
            cos_max=1e-4,
            cos_min=2e-6,

            # ===== weight decay =====
            use_weight_decay=0,
            weight_decay=0.01,

            # ===== autoregressive configs =====
            use_ims=False,
            output_len=96,
            output_len_list=None,

            # ===== train_test =====
            train_test=0,
            valid_ratio=0.2,
            is_finetuning=1,
            test_dir="test_results",
            test_version="predict",  # 可选 "test", "predict", "prune", "visualize"
            prune_ratio=0.2,
            remove_mask=False,

            # ===== forecasting task =====
            seq_len=672,
            label_len=576,
            pred_len=96,
            input_len=96,

            # ===== imputation task =====
            mask_rate=0.25,

            # ===== anomaly detection task =====
            loss_threshold=10.0,

            # ===== opacus options =====
            use_opacus=False,
            noise_multiplier=1.1,
            max_grad_norm=1.0,

            # ===== training info visualize configs =====
            record_info=False,

            # ===== version info =====
            model_version=1,
        )

        for key, value in params.items():
            if key in args.__dict__:
                args.__dict__[key] = value
            else:
                print(f"警告: 未知参数 '{key}'，将被忽略。")

        fix_seed = args.seed
        random.seed(fix_seed)
        torch.manual_seed(fix_seed)
        np.random.seed(fix_seed)
        args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
        if args.use_multi_gpu:
            ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
            port = os.environ.get("MASTER_PORT", "64209")
            hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
            rank = int(os.environ.get("RANK", "0"))  # node id
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            gpus = torch.cuda.device_count()  # gpus per node
            args.local_rank = local_rank
            print(
                'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                            gpus))
            torch.dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
            print('init_process_group finished')
            torch.cuda.set_device(local_rank)
        with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
            # setting record of experiments
            setting = f"{args.model}_{args.task_name}_{args.data}_d{args.d_model}_n{args.n_heads}_l{args.e_layers}_v{args.model_version}"
            exp = Exp_Forecast(args) 
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
        # train_data_path = os.path.join(params['train_dataset_dir'], params['train_data_filename'])
        # train_dataset_dict = torch.load(train_data_path)

        # combined_dataset = train_dataset_dict["combined_dataset"]
        # temp_cols = train_dataset_dict['Y_ID_num']
        # total_features = train_dataset_dict['X_ID_num'] + train_dataset_dict['Y_ID_num']
        
        # print("子进程：数据加载成功。")

        # train_TFdecoder(
        #     training_style=params['training_style'],
        #     combined_dataset=combined_dataset,
        #     batch_size=params['batch_size'],
        #     total_features=total_features,
        #     lr=params['lr'],
        #     temp_cols=temp_cols,
        #     d_model=params['d_model'],
        #     nhead=params['nhead'],
        #     num_layers=params['num_layers'],
        #     num_epochs=params['num_epochs'], 
        #     device=torch.device(params['device']),
        #     save_path=params['save_path'],
        #     model_name=params['model_name'],
        #     version=params['version'],
        #     log_every_n_steps=params['log_every_n_steps'],
        #     checkpoint_path=params.get('checkpoint_path', None) 
        # )
        print("子进程：测试完成。")
    except Exception as e:
        print(f"子进程发生错误: {e}")

# -----------------------------------------------------------------------------
# Streamlit UI 界面
# -----------------------------------------------------------------------------

st.set_page_config(layout="wide", page_title="通用模型测试控制与监控面板")

st.title("🛰️🌡️  通用模型测试控制与监控面板")

# --- Session State 初始化 ---
if 'testing_process' not in st.session_state:
    st.session_state.testing_process = None
if 'is_testing' not in st.session_state:
    st.session_state.is_testing = False
if 'current_run_name' not in st.session_state:
    st.session_state.current_run_name = ""
if 'params' not in st.session_state:
    st.session_state.params = {}


ckpt_base_dir = "./checkpoints"
ckpt_default_folder_name = "original"
dataset_base_dir = "./dataset/xw/"
dataset_default_folder_name = "board"
# --- 文件管理与选择 ---
st.sidebar.header("📂 数据集目录创建")
text = st.sidebar.text_input("输入数据集名称", key="dataset_name", value="board", help="请输入您想创建的数据集名称，例如 'my_dataset'。")

if st.sidebar.button("提交生成"):
    # 把 st.session_state["user_text"] 作为后端读入值
    dataset_folder = st.session_state["dataset_name"]
    try:
        test_files_dir = os.path.join(dataset_base_dir, dataset_folder, "test")
        os.makedirs(os.path.join(dataset_base_dir, dataset_folder, "test"), exist_ok=True)
    except Exception as e:
        st.sidebar.error(f"数据集目录创建失败: {e}")


# --- 文件管理与选择 ---
st.sidebar.header("📂 文件管理")
if 'dataset_name' in st.session_state and st.session_state['dataset_name']:
    dataset_folder = st.session_state['dataset_name']
    test_files_dir = os.path.join(dataset_base_dir, dataset_folder, "test")
else:
    test_files_dir = os.path.join(dataset_base_dir, dataset_default_folder_name, "test")
os.makedirs(test_files_dir, exist_ok=True)
uploaded_file = st.sidebar.file_uploader("上传新的测试文件 (csv)", type=['csv'], key="general_test_data_uploader")

default_test_filename = 'train.csv'

if uploaded_file is not None:
    save_path = os.path.join(test_files_dir, uploaded_file.name)
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.sidebar.success(f"文件 '{uploaded_file.name}' 上传成功！")
        default_test_filename = uploaded_file.name
    except Exception as e:
        st.sidebar.error(f"文件保存失败: {e}")

try:
    dataset_available_files = sorted([f for f in os.listdir(dataset_base_dir) if os.path.isdir(os.path.join(dataset_base_dir, f))])
    if not dataset_available_files:
        st.sidebar.warning("测试数据目录下没有找到任何文件夹。")
        st.stop()

    dataset_default_index = dataset_available_files.index(dataset_default_folder_name) if dataset_default_folder_name in dataset_available_files else 0
    
    dataset_folder = st.sidebar.selectbox(
        "选择测试文件夹",
        dataset_available_files,
        index=dataset_default_index,
        help="选择用于本次模型测试的数据集文件夹(测试包含文件夹内test子目录下的所有csv文件，需要保证列相同否则会报错)。"
    )
except FileNotFoundError:
    st.sidebar.error(f"测试数据目录未找到: {test_files_dir}")
    st.stop()


# --- 侧边栏：超参数配置 ---
st.sidebar.title("🛠️ 参数配置")

available_devices = get_available_devices()

try:
    ckpt_available_files = sorted([f for f in os.listdir(ckpt_base_dir) if os.path.isdir(os.path.join(ckpt_base_dir, f))])
    if not ckpt_available_files:
        st.sidebar.warning("模型权重目录下没有找到任何文件夹。")
        st.stop()

    ckpt_default_index = ckpt_available_files.index(ckpt_default_folder_name) if ckpt_default_folder_name in ckpt_available_files else 0
    
    ckpt_folder = st.sidebar.selectbox(
        "选择模型文件夹",
        ckpt_available_files,
        index=ckpt_default_index,
        help="选择用于本次模型测试的模型权重文件夹。"
    )
    if ckpt_folder == ckpt_default_folder_name:
        checkpoint_path_input = os.path.join(ckpt_base_dir, ckpt_folder, "Timer_forecast_1.0.ckpt")
    else:
        checkpoint_path_input = os.path.join(ckpt_base_dir, ckpt_folder, "checkpoint.pth")
except FileNotFoundError:
    st.sidebar.error(f"测试数据目录未找到: {os.path.join(ckpt_base_dir, ckpt_folder)}")
    st.stop()

with st.sidebar.form(key='general_params_form'):
    st.header("运行设置")
    model_name = st.text_input("模型名称 (Model Name)", value="Timer_multivariate")
    version = st.number_input("版本号 (Version)", min_value=1, value=1)
    device = st.selectbox("测试设备 (Device)", options=available_devices, index=1 if len(available_devices) > 1 else 0)

    st.header("测试超参数")
    test_version = st.select_slider("测试方式 (test: 预测后一个patch, predict: 预测对应长度的patch)", options=["test", "predict"], value="predict")
    output_len = st.number_input("预测长度 (Output Length)", min_value=1, max_value=512, value=256)
    batch_size = st.number_input("批次大小 (Batch Size)", min_value=1, max_value=256, value=32)

    st.header("模型架构 (Transformer-Decoder)")
    d_model = st.select_slider("模型维度 (d_model)", options=[128, 256, 512, 1024], value=1024)
    nhead = st.select_slider("注意力头数 (nhead)", options=[4, 8, 16], value=8)
    num_layers = st.number_input("层数 (Num Layers)", min_value=1, max_value=12, value=8)
    
    submit_button = st.form_submit_button(label='锁定参数并准备测试')


# --- 主面板：控制与监控 ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("▶️ 测试控制")
    control_placeholder = st.empty()
    control_placeholder.info("如果不需要后台测试模型，刷新或关闭浏览器前请停止测试，防止出现未受控进程。")
    start_button = st.button("启动测试", type="primary", disabled=st.session_state.is_testing)
    stop_button = st.button("停止测试", disabled=not st.session_state.is_testing)

with col2:
    st.subheader("📈 状态信息")
    status_placeholder = st.empty()
    if st.session_state.is_testing:
        device_info = st.session_state.params.get('device', 'N/A')
        model_info = st.session_state.params.get('model_type', 'N/A')
        status_placeholder.success(
            f"✅ **{model_info}** 模型测试中...\n\n"
            f"设备: `{device_info}`\n\n"
            f"日志目录: `runs/{st.session_state.current_run_name}`\n\n"
            f"测试数据: `{st.session_state.params.get('test_data_filename', 'N/A')}`"
        )
    else:
        status_placeholder.info("⏹️ 测试已停止或未开始。")

# --- 按钮逻辑处理 ---
if start_button:
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    st.session_state.current_run_name = f"{model_name}_v{version}"

    params = {
        "model_id": model_name,
        'batch_size': batch_size,
        'd_model': d_model,
        'n_heads': nhead,
        'e_layers': num_layers,
        'ckpt_path': checkpoint_path_input,
        'root_path': os.path.join(dataset_base_dir, dataset_folder) if dataset_folder in os.listdir(dataset_base_dir) else os.path.join(dataset_base_dir, 'board'),
        'test_version': test_version,
        'output_len': output_len,
        'gpu': device,
        'model_version': version,
    }
    st.session_state.params = params

    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)
    
    p = mp.Process(target=preiction_process_wrapper, args=(params,))
    p.start()
    
    st.session_state.testing_process = p
    st.session_state.is_testing = True
    
    st.session_state.testing_process.join()

    st.session_state.testing_process = None
    st.session_state.is_testing = False

    st.rerun()

# 停止测试
if stop_button:
    if st.session_state.testing_process is not None:
        st.session_state.testing_process.terminate()
        st.session_state.testing_process.join()
        st.warning(f"测试进程 {st.session_state.testing_process.pid} 已被终止。")

    st.session_state.is_testing = False
    st.session_state.testing_process = None
    
    st.rerun()

# # --- TensorBoard 集成 ---
# st.divider()
# st.subheader("📊 TensorBoard 监控")

# os.makedirs("runs", exist_ok=True)
# TB_PORT = launch_tensorboard(logdir="runs")

# SERVER_IP = "10.66.51.30"

# TENSORBOARD_URL = f"http://{SERVER_IP}:{TB_PORT}"

# st.info(f"TensorBoard 服务已由应用自动在后台启动。")

# components.iframe(TENSORBOARD_URL, height=800, scrolling=True)