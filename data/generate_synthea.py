import pandas as pd
import numpy as np
from faker import Faker
import datetime
import os

fake = Faker()

def generate_mock_data(num_patients=5000):
    np.random.seed(42)
    Faker.seed(42)
    
    departments = ['Cardiology', 'Neurology', 'Orthopedics', 'General Surgery', 'ICU', 'Emergency']
    conditions = ['Heart Failure', 'Stroke', 'Hip Fracture', 'Appendicitis', 'Sepsis', 'Trauma']
    
    data = []
    
    for _ in range(num_patients):
        patient_id = fake.uuid4()
        age = int(np.random.normal(65, 15))
        age = max(18, min(100, age))
        
        department = np.random.choice(departments)
        condition = np.random.choice(conditions)
        
        # Base ALOS depends on department and age
        base_alos = {
            'Cardiology': 5,
            'Neurology': 7,
            'Orthopedics': 4,
            'General Surgery': 3,
            'ICU': 10,
            'Emergency': 1
        }[department]
        
        alos_modifier = (age - 65) * 0.1
        actual_alos = max(1.0, np.random.normal(base_alos + alos_modifier, 2))
        
        admission_date = fake.date_between(start_date='-1y', end_date='today')
        discharge_date = admission_date + datetime.timedelta(days=int(actual_alos))
        
        # Readmission probability increases with age and certain conditions
        readmission_prob = 0.05
        if age > 75:
            readmission_prob += 0.05
        if condition in ['Heart Failure', 'Sepsis']:
            readmission_prob += 0.10
            
        readmitted = np.random.rand() < readmission_prob
        
        data.append({
            'patient_id': patient_id,
            'age': age,
            'department': department,
            'condition': condition,
            'admission_date': admission_date,
            'discharge_date': discharge_date,
            'length_of_stay': round(actual_alos, 1),
            'readmitted_30d': int(readmitted)
        })
        
    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/patients.csv', index=False)
    print(f"Generated {num_patients} mock patient records at data/patients.csv")

if __name__ == '__main__':
    generate_mock_data()
