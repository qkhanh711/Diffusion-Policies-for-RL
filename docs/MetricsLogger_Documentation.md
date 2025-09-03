# MetricsLogger Documentation

## Tổng quan

`MetricsLogger` là một class được thiết kế để theo dõi và ghi lại metrics chi tiết cho GAI Service Environment (`env_v6_baseline.py`). Class này cung cấp khả năng tracking performance, phân tích xu hướng, và xuất báo cáo chi tiết.

## Tính năng chính

### 1. Episode-level Metrics
- **Episode rewards**: Theo dõi reward tổng của mỗi episode
- **Episode lengths**: Độ dài của mỗi episode
- **Service ratios**: Tỷ lệ users được phục vụ trong mỗi episode
- **Served counts**: Số lượng users được phục vụ

### 2. Step-level Metrics
Theo dõi metrics chi tiết cho từng step:
- `rewards`: Reward tại mỗi step
- `served_users`: Số users được phục vụ
- `time_spent`: Thời gian sử dụng trong batch processing
- `peak_memory`: Peak memory usage
- `total_flops`: Tổng FLOPs computation
- `num_batches`: Số lượng batches được tạo
- `peak_users_per_batch`: Peak số users per batch
- `memory_utilization`: Tỷ lệ sử dụng memory
- `penalties`: Penalties áp dụng
- `bonuses`: Bonuses nhận được
- `prices`: Revenue/price từ services

**QoS Metrics:**
- `avg_qos_achieved`: QoS trung bình đạt được
- `avg_qos_required`: QoS trung bình yêu cầu
- `qos_satisfaction_rate`: Tỷ lệ đáp ứng QoS requirements

**Diffusion Steps Metrics:**
- `avg_diffusion_steps`: Số bước diffusion trung bình
- `min_diffusion_steps`: Số bước diffusion nhỏ nhất  
- `max_diffusion_steps`: Số bước diffusion lớn nhất

### 3. Constraint Violations Tracking
- **Memory violations**: Vi phạm giới hạn memory
- **Time violations**: Vi phạm time budget
- **FLOPs violations**: Vi phạm computation limit
- **QoS violations**: Vi phạm Quality of Service requirements

### 4. Export & Logging
- **JSON export**: Xuất detailed logs cho analysis
- **File logging**: Ghi episode summaries vào file
- **Real-time monitoring**: Theo dõi metrics trong thời gian thực

## Cách sử dụng

### 1. Khởi tạo Environment với MetricsLogger

```python
from env_v6_baseline import GAIServiceEnv_v1_baseline, EnvConfig_v1_baseline

config = EnvConfig_v1_baseline("MyEnvironment")

# Khởi tạo với metrics enabled
env = GAIServiceEnv_v1_baseline(
    config, 
    enable_metrics=True,           # Bật metrics logging
    metrics_window_size=100,       # Track 100 episodes gần nhất
    log_file="metrics.log"         # File để ghi episode logs
)
```

### 2. Sử dụng trong Training Loop

```python
# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    
    for step in range(config["T"]):
        action = get_action(state)  # Your policy
        next_state, reward, done, info = env.step(action)
        episode_reward += reward
        
        if done:
            break
    
    # Metrics được tự động ghi lại trong env.step() và env.reset()

# Lấy summary metrics
env.get_metrics_summary(detailed=True)
```

### 3. Phân tích Performance

```python
# Lấy thống kê episodes
episode_stats = env.get_episode_statistics()
print(f"Average reward: {episode_stats['avg_episode_reward']:.2f}")
print(f"Service ratio: {episode_stats['avg_service_ratio']:.3f}")

# Phân tích xu hướng
trends = env.get_performance_trends()
print(f"Recent rewards: {trends['reward_trend'][-5:]}")

# Kiểm tra constraint violations
violations = env.get_violation_stats()
print(f"Memory violation rate: {violations['memory_violation_rate']:.2%}")
```

### 4. Export Data cho External Analysis

```python
# Export detailed logs
env.export_metrics("detailed_metrics.json")

# File sẽ chứa tất cả step-level data để phân tích ngoài
```

### 5. So sánh Policies

```python
policies = ["PolicyA", "PolicyB", "PolicyC"]
results = {}

for policy_name in policies:
    env.reset_metrics()  # Reset metrics cho policy mới
    
    # Run episodes với policy
    for episode in range(num_eval_episodes):
        # ... run episodes ...
    
    # Lấy results
    stats = env.get_episode_statistics()
    results[policy_name] = stats['avg_episode_reward']

# So sánh results
best_policy = max(results.keys(), key=lambda p: results[p])
print(f"Best policy: {best_policy}")
```

## API Reference

### MetricsLogger Class

#### Constructor
```python
MetricsLogger(window_size=100, log_file=None)
```
- `window_size`: Số episodes/steps gần nhất để track (cho moving averages)
- `log_file`: File path để ghi episode logs (optional)

#### Main Methods
- `log_step(step_info, reward, timestep)`: Ghi metrics cho 1 step
- `log_episode(episode_reward, episode_length, total_served, num_users)`: Ghi metrics cho 1 episode
- `get_episode_statistics()`: Lấy thống kê episode-level
- `get_step_statistics()`: Lấy thống kê step-level  
- `get_violation_statistics()`: Lấy thống kê constraint violations
- `get_performance_trends()`: Lấy xu hướng performance
- `print_summary(detailed=False)`: In tóm tắt metrics
- `export_detailed_logs(filename)`: Xuất detailed logs ra JSON
- `reset_metrics()`: Reset tất cả metrics

### GAIServiceEnv_v1_baseline Integration

#### Constructor Updates
```python
GAIServiceEnv_v1_baseline(config, enable_metrics=True, metrics_window_size=100, log_file=None)
```

#### New Methods
- `get_episode_statistics()`: Wrapper cho metrics logger
- `get_step_statistics()`: Wrapper cho metrics logger
- `get_metrics_summary(detailed=False)`: Print và return summary
- `get_performance_trends()`: Lấy performance trends
- `get_violation_stats()`: Lấy violation statistics
- `export_metrics(filename)`: Export detailed metrics
- `reset_metrics()`: Reset metrics counter

## Example Use Cases

### 1. Training Monitoring
```python
# Theo dõi quá trình training
env = GAIServiceEnv_v1_baseline(config, enable_metrics=True)

for episode in range(1000):
    # ... training ...
    
    if episode % 100 == 0:
        env.get_metrics_summary()  # Print progress every 100 episodes
```

### 2. Hyperparameter Tuning
```python
hyperparams = [{"lambda_qos": 0.1}, {"lambda_qos": 0.3}, {"lambda_qos": 0.5}]
results = {}

for params in hyperparams:
    config.update(params)
    env = GAIServiceEnv_v1_baseline(config, enable_metrics=True)
    
    # Run evaluation
    # ...
    
    results[str(params)] = env.get_episode_statistics()['avg_episode_reward']
```

### 3. A/B Testing
```python
# So sánh 2 versions của environment
env_v1 = GAIServiceEnv_v1_baseline(config_v1, enable_metrics=True)
env_v2 = GAIServiceEnv_v1_baseline(config_v2, enable_metrics=True)

# Run same episodes on both
for env, name in [(env_v1, "V1"), (env_v2, "V2")]:
    # ... run episodes ...
    stats = env.get_episode_statistics()
    print(f"{name}: Avg reward = {stats['avg_episode_reward']:.2f}")
```

### 4. Real-time Dashboard
```python
# Continuous monitoring
env = GAIServiceEnv_v1_baseline(config, enable_metrics=True)

while training:
    # ... training steps ...
    
    if step % 1000 == 0:
        # Update dashboard
        stats = env.get_episode_statistics()
        violations = env.get_violation_stats()
        
        dashboard.update({
            'reward': stats['avg_episode_reward'],
            'service_ratio': stats['avg_service_ratio'],
            'violations': violations['total_violations']
        })
```

## Output Files

### Episode Log File (JSON Lines)
Mỗi dòng chứa một episode summary:
```json
{"episode": 1, "timestamp": 1234567890, "reward": 2500.5, "length": 10, "served": 45, "service_ratio": 0.45}
```

### Detailed Metrics Export (JSON)
Array of detailed step-by-step logs:
```json
[
  {
    "episode": 1,
    "timestep": 0,
    "total_step": 1,
    "reward": 250.5,
    "timestamp": 0.123,
    "served": 5,
    "time_spent": 0.45,
    "peak_mem": 32.1,
    "total_flops": 150000000,
    "avg_qos_achieved": 38.5,
    "avg_qos_required": 30.0,
    "qos_satisfaction_rate": 0.8,
    "qos_violations": 1,
    "served_qos_achieved": [38.5, 42.1, 28.3, 35.0, 41.2],
    "served_qos_required": [30.0, 30.0, 30.0, 30.0, 30.0],
    "avg_diffusion_steps": 15.2,
    "min_diffusion_steps": 8,
    "max_diffusion_steps": 25,
    "served_diffusion_steps": [12, 25, 8, 15, 16],
    ...
  }
]
```

## Performance Considerations

- **Memory Usage**: MetricsLogger sử dụng `deque` với `maxlen` để tránh memory overflow
- **File I/O**: Episode logs được append incrementally, detailed export chỉ khi cần
- **CPU Overhead**: Minimal overhead (~1-2% trong testing)
- **Storage**: Episode logs ~100-200 bytes/episode, detailed logs ~1-2KB/step

## Troubleshooting

### Common Issues

1. **"Metrics logging is disabled"**
   - Ensure `enable_metrics=True` khi khởi tạo environment

2. **File permission errors**
   - Check write permissions cho log_file directory

3. **Memory usage cao**
   - Giảm `metrics_window_size` nếu training rất lâu
   - Định kỳ gọi `reset_metrics()` nếu cần

4. **Performance degradation**
   - Disable detailed logging cho production inference
   - Set `log_file=None` nếu không cần file logging

### Best Practices

1. **Development**: Enable tất cả metrics với detailed logging
2. **Training**: Enable metrics với reasonable window_size (50-200)
3. **Production**: Disable metrics hoặc chỉ enable basic tracking
4. **Evaluation**: Enable metrics với smaller window_size (10-50)

## Integration với Existing Code

MetricsLogger được thiết kế để tích hợp seamlessly:

- **Backward Compatible**: Existing code vẫn hoạt động bình thường
- **Optional**: Có thể disable hoàn toàn mà không ảnh hưởng functionality
- **Non-intrusive**: Không thay đổi core environment logic
- **Extensible**: Dễ dàng thêm metrics mới khi cần
