# GAI Service Environment Documentation (v3)

## Overview

The GAI Service Environment (`GAIServiceEnv_v1`) is a reinforcement learning environment that simulates a Generative AI service provider managing multiple users with different resource requirements. The environment models real-world constraints including computational resources, memory limits, latency requirements, and quality of service (QoS) considerations.

## Environment Components

### 1. Service Provider
- **Location**: Fixed at coordinates (0, 0, 50) in 3D space
- **Resources**: Limited computational power (FLOPS), memory, and bandwidth
- **Objective**: Maximize revenue while satisfying user constraints

### 2. Users
- **Count**: Configurable (default: 10 users)
- **Mobility**: Users move randomly within the environment
- **Requests**: Each user requests GAI services with specific requirements

### 3. System Constraints
- **Memory Limit (Mmax)**: Maximum available memory (default: 48 GB)
- **Computational Limit (Gmax)**: Maximum FLOPS (default: 1e10)
- **Latency Limit (sys_tau)**: Maximum allowed latency per user (default: 3 seconds)

## State Space

The observation space is a continuous box with dimension `6 × num_users`. For each user, the state includes:

1. **Position X**: Normalized x-coordinate [-1, 1]
2. **Position Y**: Normalized y-coordinate [-1, 1] 
3. **Image Size**: Normalized image data size [0, 0.1]
4. **Prompt Size**: Normalized prompt data size [0, 1]
5. **Direction**: Normalized movement direction [0, 1]
6. **QoS Required**: Normalized quality requirement [0, 0.6]

```python
state_dim = 6 * num_users
observation_space = Box(low=-inf, high=inf, shape=(state_dim,), dtype=float32)
```

## Action Space

The action space is continuous with dimension `2 × num_users`. For each user, actions include:

1. **Serve Decision**: Continuous value [-1, 1] → converted to binary serve/not-serve
2. **Denoise Steps**: Continuous value [-1, 1] → mapped to [1, max_denoise_steps]

```python
action_dim = 2 * num_users  
action_space = Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=float32)
```

### Action Processing
- **Serve Decision**: `serve = 1 if (action + 1)/2 >= 0.5 else 0`
- **Denoise Steps**: `steps = 1 + normalized_action * (max_steps - 1)`

## Reward Function

The reward function balances revenue generation with constraint satisfaction:

### Revenue Components
```python
price = 1e-9 * memory + 1e-12 * flops + 1e-8 * communication_data
total_reward += price  # For each served user
```

### Penalty Components
```python
# QoS penalty for each served user
qos_penalty = max(0, actual_qos - required_qos) * lambda_qos

# System constraint violations
if total_memory > Mmax:
    penalty += lambda_mem * (total_memory - Mmax)
if total_flops > Gmax:
    penalty += lambda_flops * (total_flops - Gmax)  
if total_latency > latency_limit:
    penalty += lambda_latency * (total_latency - latency_limit)

# Rejection penalty
rejection_penalty = 0.01 * psi  # Per rejected user
```

### Bonus Components
```python
# System compliance bonus
if all_constraints_satisfied:
    bonus += psi * 0.1

# Service ratio bonus  
service_bonus = (served_users / total_users) * psi * 0.2

# Resource efficiency bonus
efficiency_bonus = (memory_util + flops_util) * psi * 0.05
```

### Final Reward
```python
final_reward = total_revenue - total_penalty + total_bonus
```

## Resource Calculations

### 1. Computational Requirements (FLOPS)
```python
rho = image_size / base_image_size
flops = rho * (GE0 + GD0 + denoise_steps * G_eps + G_prompt)
```

### 2. Memory Requirements
```python
memory = (c1 * image_size + c2) * 1.3  # GB
```

### 3. Latency Calculation
```python
distance = ||user_position - service_provider_position||
channel_rate = bandwidth * log2(1 + SNR) / 8  # bytes/sec

t_upload = (image_size + prompt_size) / upload_rate
t_memory = (image_size + prompt_size) / memory_rate  
t_compute = flops / computational_power
t_download = image_size / download_rate

total_latency = t_upload + t_memory + t_compute + t_download
```

### 4. Quality of Service (QoS)
QoS is modeled as BRISQUE score (lower is better):
- **< 5 steps**: 40-50 (poor quality)
- **5-10 steps**: 30-40 (fair quality)
- **10-15 steps**: 20-30 (good quality)
- **15+ steps**: 10-20 (excellent quality)

## Configuration Parameters

### System Parameters
```python
{
    "num_users": 10,                    # Number of users
    "T": 10,                           # Episode length
    "sys_tau": 3,                      # Max latency per user (seconds)
    "Gmax": 1e10,                      # Max FLOPS
    "Mmax": 48,                        # Max memory (GB)
    "max_denoise_steps": 35,           # Max diffusion steps
}
```

### Network Parameters
```python
{
    "bandwidth": 1e6,                  # Total bandwidth (Hz)
    "noise_power": 4.0e-21,           # Noise power (W)
    "upload_power": 0.0501,           # Upload power (W)
    "download_power": 0.5012,         # Download power (W)
    "h0": 1.42e-4,                    # Channel gain
    "path_loss": 2.0,                 # Path loss exponent
}
```

### Cost Parameters
```python
{
    "lambda_qos": 0.75,               # QoS penalty weight
    "lambda_latency": 0.5,            # Latency penalty weight  
    "lambda_mem": 1.0,                # Memory penalty weight
    "lambda_flops": 1.0,              # FLOPS penalty weight
    "psi": 100,                       # Bonus/penalty scale
}
```

### Computational Parameters
```python
{
    "PVM": 1e12,                      # Computational power (FLOPS/sec)
    "Rmem": 2.304e12,                 # Memory bandwidth (bytes/sec)
    "c1": 3.81e-6,                    # Memory coefficient 1
    "c2": 4.86,                       # Memory coefficient 2
    "GE0": 1e8,                       # Base encoding FLOPS
    "GD0": 1e8,                       # Base decoding FLOPS
    "G_eps": 1e8,                     # FLOPS per denoising step
    "G_prompt": 1e7,                  # Prompt processing FLOPS
}
```

## Episode Dynamics

### 1. Initialization
- Users are randomly positioned in [-500, 500] × [-500, 500]
- Each user gets random image size, prompt size, and QoS requirements
- Time step counter is reset to 0

### 2. Step Execution
1. Agent selects actions for all users
2. Environment processes serve/reject decisions sequentially by distance
3. For each served user:
   - Calculate resource requirements
   - Check constraint feasibility  
   - Update cumulative resource usage
   - Compute QoS and pricing
4. Calculate total reward and penalties
5. Move all users to new positions
6. Increment time step

### 3. Termination
- Episode ends when `time_step >= T`
- No early termination conditions

## Constraint Handling

The environment implements a **sequential serving strategy**:

1. **Prioritization**: Users sorted by distance (closer users served first)
2. **Feasibility Check**: For each user in priority order:
   ```python
   memory_ok = (current_memory + required_memory) <= Mmax
   flops_ok = (current_flops + required_flops) <= Gmax  
   latency_ok = required_latency <= sys_tau
   ```
3. **Decision**: Serve if all constraints satisfied, reject otherwise
4. **Resource Update**: Update cumulative resources for served users

## Performance Metrics

The environment tracks comprehensive metrics:

### System Metrics
- **Total Served**: Number of users served
- **Total Rejected**: Number of users rejected  
- **Service Ratio**: served_users / total_users
- **Total Revenue**: Sum of prices for served users
- **Total Penalty**: Sum of all penalties

### Resource Utilization
- **Memory Utilization**: total_memory / Mmax
- **FLOPS Utilization**: total_flops / Gmax
- **Latency Utilization**: total_latency / latency_limit

### Constraint Satisfaction
- **Memory Constraint**: total_memory ≤ Mmax
- **FLOPS Constraint**: total_flops ≤ Gmax
- **Latency Constraint**: total_latency ≤ sys_tau × num_users

## Usage Example

```python
from env.env_v3 import GAIServiceEnv_v1, EnvConfig_v1

# Create environment
config = EnvConfig_v1("GAIServiceEnv")
config["num_users"] = 5  # Customize parameters
env = GAIServiceEnv_v1(config)

# Run episode
state = env.reset()
for step in range(config["T"]):
    action = agent.select_action(state)  # Your RL agent
    next_state, reward, done, info = env.step(action)
    
    print(f"Step {step}:")
    print(f"  Served: {info['served']}")
    print(f"  Reward: {reward:.4f}")
    print(f"  Memory Usage: {info['resource_utilization']['memory']:.2%}")
    
    state = next_state
    if done:
        break
```

## Key Features

### 1. **Realistic Resource Modeling**
- Accurate FLOPS and memory calculations based on diffusion model requirements
- Network latency modeling with distance-dependent channel rates
- QoS modeling based on denoising steps

### 2. **Multi-Objective Optimization**
- Revenue maximization vs. constraint satisfaction
- Service coverage vs. resource efficiency
- Quality vs. computational cost trade-offs

### 3. **Dynamic Environment**
- User mobility creates time-varying channel conditions
- Stochastic user requests with varying requirements
- Real-time resource allocation decisions

### 4. **Comprehensive Monitoring**
- Detailed metrics for analysis and debugging
- Resource utilization tracking
- Constraint violation detection

## Common Issues and Solutions

### 1. **No Users Served**
- **Cause**: Resource constraints too tight or poor action selection
- **Solution**: Adjust Mmax/Gmax parameters or improve agent training

### 2. **Constraint Violations**
- **Cause**: Environment logic errors or extreme parameter values
- **Solution**: Verify constraint checking logic and parameter ranges

### 3. **Unstable Training**
- **Cause**: Reward scale issues or poor state normalization
- **Solution**: Adjust penalty weights (lambda values) and ensure proper normalization

### 4. **Poor Convergence**
- **Cause**: Complex reward function or insufficient exploration
- **Solution**: Simplify reward function initially, use curriculum learning

## Extension Points

1. **Multi-Service Provider**: Extend to competitive scenarios
2. **Heterogeneous Users**: Different service types and requirements  
3. **Dynamic Pricing**: Adaptive pricing based on demand/supply
4. **Energy Modeling**: Add power consumption constraints
5. **Federated Learning**: Incorporate collaborative training scenarios

This environment provides a realistic testbed for developing RL algorithms for resource allocation in edge computing and GAI service provisioning scenarios.
