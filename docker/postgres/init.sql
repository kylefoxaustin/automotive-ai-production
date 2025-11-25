CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    epochs INTEGER,
    final_loss FLOAT,
    cost_profile VARCHAR(50),
    model_path VARCHAR(255)
);

CREATE TABLE IF NOT EXISTS metrics (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES training_runs(id),
    epoch INTEGER,
    loss FLOAT,
    accuracy FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS can_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    can_id VARCHAR(8),
    data BYTEA,
    channel INTEGER
);
