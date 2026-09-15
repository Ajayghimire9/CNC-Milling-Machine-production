select
    model_version,
    processing_time,
    processing_time_std,
    average_power_consumption,
    average_power_consumption_std,
    anomaly_score,
    anomaly,
    risk_level,
    current_timestamp as loaded_at
from {{ ref('stg_machine_predictions') }}
