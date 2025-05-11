CREATE TABLE DiabetesPatients (
    PatientID INT PRIMARY KEY,
    Name VARCHAR(100),
    Age INT,
    Gender VARCHAR(10),
    Weight DECIMAL(5,2),        
    HbA1c DECIMAL(4,2),         
    DiagnosisDate DATE,
    LastCheckupDate DATE,
	DiabetesType VARCHAR(25)
);