# How to use
1. install 
```bash
git clone https://github.com/pantheon5100/sr_metrics_cal.git
cd sr_metrics_cal
pip install -e .
```

2. integrete into your code.
```python
from sr_metrics import CalMetrics
# Give the path to save the result table
calculate_metrics = CalMetrics(f'results/result.csv')

for lr, hr in dataloader:
    sr = ...
    results_dict = calculate_metrics(sample, gt_img, crop_border= scale_factor, input_type='tensor_rgb_01', file_name=file_name, test_y_channel=True)

calculate_metrics.average_results()
```

3. use to test results when given lr directory and hr directory
```python
from sr_metrics import CalMetrics
# Give the path to save the result table
calculate_metrics = CalMetrics(f'results/result.csv')

calculate_metrics.metrics_dir(sr_dir, hr_dir, filename_tmpl='{}', crop_border=scale_factor, test_y_channel=True, pbar=True)

```