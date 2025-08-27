# 📝 Detailed Documentation

## ⚙️ 1. Data Preparation

Please obtain the [GUI control dataset](https://huggingface.co/datasets/hflqf88888/SWIRL_GUI_data) or the [math reasoning dataset](https://huggingface.co/datasets/hflqf88888/SWIRL_MATH_data) from HuggingFace, and make sure the `./data` directory follows the structure illustrated below:


```
SWIRL
├── data
│   ├── SWIRL_GUI_data
│   │   ├── test
│   │   │   ├── high_level
│   │   │   │   ├── AndroidControl_test.parquet
│   │   │   │   └── GUIOdyssey_test.parquet
│   │   │   └── low_level
│   │   │       ├── AndroidControl_test.parquet
│   │   │       ├── Gui-Act_web_test.parquet
│   │   │       ├── OmniAct_desktop_test.parquet
│   │   │       └── OmniAct_web_test.parquet
│   │   └── train
│   │       ├── sft_images
│   │       │   └── ...
│   │       ├── stage1_Interactor_warmup1500.parquet
│   │       ├── stage1_Navigator_warmup1500.json
│   │       └── stage2_interleaved2000.parquet
│   └── SWIRL_MATH_data
│       ├── test
│       │   ├── CMATH_test.parquet
│       │   ├── gsm8k_test.parquet
│       │   └── MATH500_test.parquet
│       └── train
│           └── MATH_train.parquet
└── ...
```

## 💎 2. SWIRL Training

### 2.1 Training Framework Overview

We provide SLURM-based training scripts for running experiments on a compute cluster.
The overall training pipeline follows these steps:
> 1. Launch the Interactor/Student inference service with vLLM.
> 2. Train the Navigator/Teacher with GRPO.
> 3. Run inference with the Navigator/Teacher to generate intermediate instructions, which are then used as training data for the Interactor/Student.
> 4. Train the Interactor/Student using the generated data with GRPO.

The training scripts for GUI Control and Math Reasoning follow a similar structure:

```
swirl
├── admm.slurm
├── generation_for_B_training.sh
├── running_config
│   ├── planner_generation_default.yaml
│   └── ppo_trainer.yaml
├── start_vllm.sh
├── tool_config
│   ├── B_config.json
│   └── B_config.yaml
├── train_A.sh
└── train_B.sh
```

File Descriptions：
* `admm.slurm`: The main SLURM entry point that coordinates the entire training process.
* `running_config/`: Contains configuration files for training and inference (e.g., PPO trainer, rollout settings).
* `tool_config/`: Contains coordination settings between Agent A and Agent B (e.g., tool definitions, execution configs).
* `start_vllm.sh`: Script for deploying the Interactor/Student inference service with vLLM (Step 1).
* `train_A.sh / train_B.sh`: Training scripts for Agent A (Navigator/Teacher) and Agent B (Interactor/Student), corresponding to Step 2 and Step 4.
* `generation_for_B_training.sh`: Inference script for Agent A, used to generate training data for Agent B (Step 3).


### 2.2 GUI Control

In our multi-agent design for GUI Control, the Navigator and Interactor communicate using the tool-based interface style of SGLang. The training process therefore consists of two stages:
> 1. Warm-up, which enables the agents to quickly learn the communication protocol.
> 2. Interleaved updates, which further unlocks and consolidates the potential of the multi-agent system.


#### 2.2.1 Stage 1: Warm-up

##### 2.2.1.1 Navigator Warm-up (SFT)
For the Navigator, we adopt supervised fine-tuning (SFT).   
First, configure the root directory of the annotations and images in `SWIRL/qwen-vl-finetune/qwenvl/data/__init__.py`. Specifically, set the paths to match the following training data on your local machine:
* `SWIRL/data/decouple/release/SWIRL_GUI_data/train/stage1_Navigator_warmup1500.json`
* `SWIRL/data/decouple/release/SWIRL_GUI_data/train/sft_images`

For further details, please refer to the [official Qwen2.5-VL instructions](https://github.com/QwenLM/Qwen2.5-VL/tree/main/qwen-vl-finetune).

Launch training with:
``` shell
cd SWIRL/qwen-vl-finetune/
bash scripts/warm_up/navigator_s1_training.sh
```


##### 2.2.1.2 Interactor Warm-up (RL)
For the Interactor, we adopt reinforcement learning.
``` shell
cd SWIRL/
bash examples/warm_up/Interactor_s1_training.sh
```

#### 2.2.2 Stage 2: Interleaved Updates

The interleaved update stage alternates training between Navigator and Interactor to strengthen coordination.

The training scripts are provided in `SWIRL/examples/swirl`. Before launching, you need to specify the cluster partition in `SWIRL/examples/swirl/admm.slurm` and adjust other parameters (e.g., CPU resources) according to your cluster setup.

Launch training with:

``` shell
cd SWIRL/
sbatch examples/swirl/admm.slurm 
```



### 2.3 Math Reasoning

The training process for Math Reasoning follows a simplified workflow that only involves alternating updates. Communication between the Teacher and Student is also streamlined: we use the *<teacher_response>* tag to mark and extract the responses generated by the Teacher.

The training scripts are located in `SWIRL/examples/swirl_math`. Similar to GUI Control training, you need to specify the cluster partition in `SWIRL/examples/swirl_math/admm.slurm` and adjust the configuration according to your cluster resources. Training can then be launched with the following command:

``` shell
cd SWIRL/
sbatch examples/swirl_math/admm.slurm 
```

## 🛠️ 3. Inference and Evaluation

### 3.1 Inference

Inference in SWIRL is a multi-stage process:
> 1. Agent A first performs inference to generate intermediate reasoning and instructions.
> 2. Agent B then takes these outputs to complete the final inference.

You can run inference using the following scripts:

#### GUI High-Level Tasks
``` shell
cd SWIRL/
bash examples/inference/inference_gui_high_level.sh
```

#### GUI Low-Level Tasks
``` shell
cd SWIRL/
bash examples/inference/inference_gui_low_level.sh
```


#### Math Reasoning Tasks
``` shell
cd SWIRL/
bash examples/inference/inference_math.sh
```

### 3.2 Evaluation
To evaluate model performance, we provide separate scripts for GUI and Math tasks:

#### GUI Evaluation
``` shell
cd SWIRL/
bash examples/inference/evalutate_gui.sh
```

#### Math Evaluation
``` shell
cd SWIRL/
bash examples/inference/evalutate_math.sh
```


## 📕 4. Your Customized SWIRL Training


This project provides multi-agent training pipelines for GUI control and math reasoning. The same paradigm can be easily extended to other domains and scaled up to support more agents. The key principle is simple: *train one agent, deploy the other, and alternate updates via lightweight communication*, ensuring efficient coordination and stable training.


**Step 1: Define Agent Interaction Protocols**

The first step is to clearly define the interaction format between agents, including both the training prompts and the message-passing protocol used during communication. To achieve this, you need to:
> 1. Write preprocessing scripts to prepare the training dataset.
> 2. Implement data conversion scripts that transform training samples into the formats required by each agent (see `SWIRL/multi_agent/data_preprocess/plan2exec.py` for reference).

**Step 2: Configure Interaction and Reward Functions**

During each round, one agent is optimized while the other remains fixed. Their interactions must be clearly defined, along with custom reward functions. In our implementation, we deploy one agent as an inference service on a separate node and integrate its responses into the reward calculation through HTTP requests. You will need to:
> 1. Customize reward functions for each agent (e.g., see `SWIRL/verl/utils/reward_score/gui.py`).
> 2. Implement a reward manager that wraps multi-threaded requests to deployed agents (e.g., `SWIRL/verl/workers/reward_manager/planner_specific.py`).

**Step 3. Set Agent-Specific Training Configurations**

Each agent requires its own training configuration. For reference, see the script in `SWIRL/examples/swirl/train_navigator.sh`.

**Step 4. Build the Alternating Training Pipeline**

Finally, construct the overall alternating training pipeline. A complete example is provided in `SWIRL/examples/swirl/admm.slurm`.


## 💫 5. Tips
1. In the project codebase, the `Navigator` is sometimes referred to as the *Planner*, and the `Interactor` as the *Executor*. These terms are used interchangeably and denote the same roles.
2. During training, you may encounter situations where a proxy must be set specifically for wandb. Please refer to [this](https://verl.readthedocs.io/en/latest/faq/faq.html#how-to-set-proxy-only-for-wandb) on proxy configuration. To apply this setting, update the training scripts such as: `SWIRL/examples/swirl/train_navigator.sh` and `SWIRL/examples/swirl/train_interactor.sh`.
Similarly, for math reasoning tasks, you should configure the proxy in the corresponding training scripts as well.
