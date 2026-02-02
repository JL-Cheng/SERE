from mmengine.config import read_base

with read_base():
    ################################################################
    # dataset
    ################################################################

    from opencompass.configs.datasets.cmmlu.cmmlu_0shot_cot_gen_305931 import cmmlu_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_cot_gen_1d56df import BoolQ_datasets
    from opencompass.configs.datasets.bbh.bbh_gen_5b92b0 import bbh_datasets

    from opencompass.configs.datasets.math.math_evaluatorv2_gen_cecb31 import math_datasets
    from opencompass.configs.datasets.gsm8k.gsm8k_gen_1d7fe4 import gsm8k_datasets
    from opencompass.configs.datasets.math401.math401_gen_ab5f39 import math401_datasets
    
    from opencompass.configs.datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    from opencompass.configs.datasets.mbpp.deprecated_mbpp_gen_1e1056 import mbpp_datasets

    ################################################################
    # summary
    ################################################################
    from opencompass.configs.summarizers.groups.cmmlu import cmmlu_summary_groups
    from opencompass.configs.summarizers.groups.bbh import bbh_summary_groups

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

other_summary_groups = []
other_summary_groups.append({'name': 'Exam', 'subsets': ['cmmlu','BoolQ', 'bbh']})
other_summary_groups.append({'name': 'Math', 'subsets': ['math', 'gsm8k', 'math401']})
other_summary_groups.append({'name': 'Code', 'subsets': ['openai_humaneval', 'mbpp']})
other_summary_groups.append({'name': 'Overall', 'subsets': ['Exam', 'Math', 'Code']})

summarizer = dict(
    dataset_abbrs=[
        'Overall',
        'Exam',
        'Math',
        'Code',
        '--------- Exam ---------',  # category
        'cmmlu',
        'BoolQ',
        'bbh',
        '--------- Math ---------',  # category
        'math',
        'gsm8k',
        'math401',
        '--------- Code ---------',  # category
        'openai_humaneval',
        'mbpp', 
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith('_summary_groups')], []),
)

## model configs
from opencompass.models import VLLMwithChatTemplate

model_kwargs = dict(
    tensor_parallel_size=1,             # was 1 -> causes full model on a single GPU
    gpu_memory_utilization=0.9,        # slightly lower headroom
    trust_remote_code=True,
)
model_path = 'path/to/your/model'
model_kwargs["hf_overrides"] = {
    "architectures": ["Qwen3MoeForCausalLMSERE"], 
    "select_top_k": 2, 
    "threshold": 0.1,
}

models = [
    dict(
        type=VLLMwithChatTemplate,
        abbr='qwen3-30b-a3b-2507-vllm',
        path=model_path,
        model_kwargs=model_kwargs,
        generation_kwargs=dict(temperature=0.7, top_p=0.8, top_k=20, min_p=0),
        max_out_len=2048,
        batch_size=16,
        run_cfg=dict(num_gpus=1),
    )
]

from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.runners import LocalRunner
from opencompass.tasks import OpenICLEvalTask, OpenICLInferTask

infer = dict(
    partitioner=dict(type=NumWorkerPartitioner, num_worker=8),
    runner=dict(type=LocalRunner,
                max_num_workers=8,
                task=dict(type=OpenICLInferTask)),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner, n=10),
    runner=dict(type=LocalRunner,
                max_num_workers=128,
                task=dict(type=OpenICLEvalTask)),
)