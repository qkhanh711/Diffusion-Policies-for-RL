# QoS Metrics Enhancement for Environment v5

## Overview
This update replaces the TFLOPS plot in `environment_metrics.png` with comprehensive QoS (Quality of Service) metrics to provide better insights into service quality and user satisfaction.

## Changes Made

### 1. Environment Configuration (`src/env/env_v5.py`)
- **Balanced Penalty Weights**: Reduced harsh penalties for more balanced learning
  - `lambda_latency`: 0.5 → 0.3 (less harsh latency penalty)
  - `lambda_mem`: 1.0 → 0.8 (slightly less harsh memory penalty)
  - `lambda_flops`: 1.0 → 0.6 (less harsh FLOPS penalty)

- **Enhanced Bonus System**: Increased bonuses to encourage better resource utilization
  - `user_service_bonus`: 2 → 8 (encourage serving more users)
  - `memory_utilization_bonus`: 5 → 15 (encourage memory usage)
  - `fairness_bonus`: 3 → 10 (encourage fair service)

- **Progressive Memory Bonus**: New bonus system for balanced memory usage
  - Rewards for 30%-90% utilization (sweet spot)
  - Extra rewards for high utilization (90%+)

- **User Service Diversity Bonus**: New bonus for serving different users

### 2. Environment Metrics Plot (`src/t_main_v5.py`)
**Plot 3 (Previously FLOPS)**: Now shows **QoS Achieved**
- Displays average QoS score achieved per episode
- Lower BRISQUE scores = Better quality
- Includes target QoS line (30) for reference
- Color: Purple with markers

**Plot 6 (Previously Denoise Steps)**: Now shows **QoS Violations**
- Counts users where achieved QoS > required QoS
- Helps identify quality compliance issues
- Color: Orange with square markers

### 3. Enhanced User Positions Plot
- **Dual-panel layout**: User positions + QoS analysis
- **QoS Comparison**: Required vs Achieved QoS for each served user
- **Statistics**: Average achieved QoS and violation count
- **Visual indicators**: Target QoS line and compliance metrics

### 4. QoS Metrics Tracking
- **Real-time calculation**: QoS achieved based on denoise steps
- **Violation tracking**: Counts QoS requirement violations
- **Compliance rate**: Percentage of users meeting QoS requirements
- **Historical data**: Stored in environment metrics for analysis

## QoS Quality Mapping

| Denoise Steps | BRISQUE Score | Quality Level | Description |
|---------------|---------------|---------------|-------------|
| 1-4           | 40-50         | Poor          | Low quality, fast processing |
| 5-9           | 30-40         | Fair          | Acceptable quality |
| 10-14         | 20-30         | Good          | High quality |
| 15+           | 10-20         | Excellent     | Premium quality, slower processing |

## Benefits of Changes

### 1. **Better Resource Balance**
- Less focus on minimizing latency at all costs
- Encourages optimal memory utilization
- Balanced penalties prevent overfitting to single metric

### 2. **Quality-Focused Metrics**
- Direct visibility into service quality
- Easy identification of QoS violations
- Better understanding of user satisfaction

### 3. **Improved Learning**
- More achievable bonus thresholds
- Progressive rewards encourage exploration
- Better balance between different objectives

### 4. **Enhanced Monitoring**
- Real-time QoS tracking
- Historical quality trends
- Performance vs quality trade-offs

## Usage

### Training
The enhanced metrics will automatically appear during training:
```python
# Metrics are logged automatically
metrics_logger.log_env_metrics(episode, env_info, action, current_state, config)

# Plots are saved every 50 episodes
if episode % 50 == 0:
    metrics_logger.save_env_metrics_plot()
    metrics_logger.save_user_positions_plot()
```

### Analysis
- **`environment_metrics.png`**: 6-panel overview including QoS metrics
- **`user_positions.png`**: Enhanced user analysis with QoS comparison
- **CSV/JSON exports**: Include QoS metrics for detailed analysis

## Testing

Run the test script to verify functionality:
```bash
python test_qos_metrics.py
```

This will test:
- QoS calculation accuracy
- Metrics plotting functionality
- Data consistency across episodes

## Future Enhancements

1. **Dynamic QoS thresholds**: Adaptive quality requirements based on user preferences
2. **Quality-cost trade-offs**: Explicit modeling of quality vs resource usage
3. **User satisfaction metrics**: More sophisticated quality assessment
4. **Real-time QoS monitoring**: Live quality tracking during training

## Compatibility

- **Backward compatible**: Existing training runs will work
- **Fallback calculations**: QoS metrics calculated from denoise steps if not available
- **Gradual migration**: Can be enabled/disabled via configuration

---

**Note**: These changes are designed to work with both `gdql` and `dql` algorithms, providing better balance between latency optimization and quality maximization. 