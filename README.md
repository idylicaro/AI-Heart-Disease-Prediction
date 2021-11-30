# AI-Heart-Disease-Prediction
This is a project developed in a discipline at the university.

## How to execute in localhost
    !! I'm can't run in Windows OS !!
    - First, install requiriments with ```pip3 install -r .\requirements.txt```
    - Second, run ```pip install flask-restful``` and ```pip install flask-cors```
    - Third, if don't exist 'jsons' folder in application folder, create.
    - Execute src/server.py
    - Send an Json(in format on ./test.json) in this endpoint http://localhost:5444/disease (maybe your change localhost for you IP)
    
    - Test: 

---

### Production Link is:
- https://ia-projeto-interface.kevennykeke.repl.co/

### Attributes Information

1. Age: age of the patient [years]
2. Sex: sex of the patient [M: Male, F: Female]
3. ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. RestingBP: resting blood pressure [mm Hg]
5. Cholesterol: serum cholesterol [mm/dl]
6. FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
7. RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
8. MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
9. ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
10. Oldpeak: oldpeak = ST [Numeric value measured in depression]
11. ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
12. HeartDisease: output class [1: heart disease, 0: Normal]

---
#### What is: 
- **Angina** - Angina, also called angina pectoris, is often described as squeezing, pressure, heaviness, tightness or pain in your chest. Some people with angina symptoms say angina feels like a vise squeezing their chest or a heavy weight lying on their chest. Angina may be a new pain that needs to be checked by a doctor, or recurring pain that goes away with treatment.
- **Typical (classic) angina chest pain** consists of (1) Substernal chest pain or discomfort that is (2) Provoked by exertion or emotional stress and (3) relieved by rest or nitroglycerine (or both).
- **Atypical (probable) angina chest pain** applies when 2 out of 3 criteria of classic angina are present.
- **Blood pressure** is the force of your blood pushing against the walls of your arteries. Each time your heart beats, it pumps blood into the arteries. Your blood pressure is highest when your heart beats, pumping the blood. This is called systolic pressure. When your heart is at rest, between beats, your blood pressure falls. This is called diastolic pressure.
- **Serum cholesterol** - TODO
- **Fasting blood sugar test** - A blood sample will be taken after an overnight fast. A fasting blood sugar level less than 100 mg/dL (5.6 mmol/L) is normal. A fasting blood sugar level from 100 to 125 mg/dL (5.6 to 6.9 mmol/L) is considered prediabetes. If it's 126 mg/dL (7 mmol/L) or higher on two separate tests, you have diabetes.
- **Resting electrocardiogram** - records the electrical activity of your heart at rest. It provides information about your heart rate and rhythm.
- **ST depression** - refers to a finding on an electrocardiogram, wherein the trace in the ST segment is abnormally low below the baseline.
- **ST Slope** - The ST segment shift relative to exercise-induced increments in heart rate, the ST/heart rate slope (ST/HR slope).
