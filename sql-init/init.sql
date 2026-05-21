CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS employees (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  department TEXT NOT NULL,
  role TEXT NOT NULL,
  location TEXT NOT NULL,
  salary INTEGER NOT NULL,
  skills TEXT[] NOT NULL,
  created_at TIMESTAMP DEFAULT NOW()
);

INSERT INTO employees (name, department, role, location, salary, skills) VALUES
('Alice Martin', 'Infra', 'DevOps Engineer', 'Reims', 42000, ARRAY['docker', 'linux', 'ansible']),
('Bruno Leroy', 'Data', 'Data Engineer', 'Paris', 47000, ARRAY['python', 'sql', 'airflow']),
('Claire Dubois', 'Security', 'Cybersecurity Analyst', 'Lyon', 45000, ARRAY['siem', 'soc', 'python']),
('David Petit', 'Infra', 'System Administrator', 'Reims', 39000, ARRAY['windows', 'vmware', 'powershell']),
('Emma Bernard', 'Product', 'Business Analyst', 'Remote', 41000, ARRAY['sql', 'powerbi', 'jira']);