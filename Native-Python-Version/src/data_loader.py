import os
import pandas as pd

CLINICAL_FAQS = [
    {
        "id": "MED-001",
        "category": "Infectious Diseases",
        "condition": "Dengue Fever",
        "source": "WHO / NIH Clinical Guidelines",
        "question": "What are the warning signs of severe Dengue fever and which painkillers must be avoided?",
        "answer": "High fever (40°C/104°F) is accompanied by severe headache, pain behind the eyes, muscle and joint pains, and rash. WARNING SIGNS of Severe Dengue (occurring 3–7 days after illness onset) include severe abdominal pain, persistent vomiting, rapid breathing, bleeding gums or nose, fatigue, and blood in vomit or stool. CRITICAL MEDICATION RULE: Patients must take ONLY Paracetamol (Acetaminophen) for fever and pain. Strictly AVOID Aspirin, Ibuprofen, Naproxen, and other NSAIDs, as they thin the blood and significantly increase the risk of fatal internal bleeding (Dengue Hemorrhagic Fever)."
    },
    {
        "id": "MED-002",
        "category": "Infectious Diseases",
        "condition": "Dengue Fever",
        "source": "WHO / NIH Clinical Guidelines",
        "question": "How is Dengue managed at home and when is hospitalization required?",
        "answer": "Mild dengue is managed at home with strict bed rest, Oral Rehydration Salts (ORS), coconut water, fresh juices, and Paracetamol (maximum 4 grams per day for adults, spaced every 6 hours). Hospitalization is immediately required if platelet counts drop rapidly alongside warning signs, if hematocrit rises sharply (indicating plasma leakage), or if the patient cannot tolerate oral fluids."
    },
    {
        "id": "MED-003",
        "category": "Endocrinology & Metabolic",
        "condition": "Type 2 Diabetes Mellitus",
        "source": "NIH National Institute of Diabetes (NIDDK)",
        "question": "What are the diagnostic blood sugar thresholds for Prediabetes and Diabetes (HbA1c and Fasting)?",
        "answer": "According to standard ADA and NIH criteria: 1) Normal: Fasting Plasma Glucose (FPG) less than 100 mg/dL and HbA1c less than 5.7%. 2) Prediabetes: FPG between 100 mg/dL and 125 mg/dL, OR HbA1c between 5.7% and 6.4%. 3) Diabetes: FPG of 126 mg/dL or higher (after 8 hours of fasting), OR HbA1c of 6.5% or higher, OR a 2-hour Oral Glucose Tolerance Test (OGTT) level of 200 mg/dL or higher."
    },
    {
        "id": "MED-004",
        "category": "Endocrinology & Metabolic",
        "condition": "Hypoglycemia (Low Blood Sugar)",
        "source": "NIH National Institute of Diabetes (NIDDK)",
        "question": "How do you treat acute low blood sugar (Hypoglycemia) using the 15-15 Rule?",
        "answer": "Hypoglycemia occurs when blood glucose drops below 70 mg/dL. Symptoms include shakiness, sweating, dizziness, fast heartbeat, and confusion. Use the '15-15 Rule': Consume 15 grams of fast-acting carbohydrates (such as 4 ounces / half a cup of regular fruit juice, 1 tablespoon of sugar or honey, or 3–4 glucose tablets), wait 15 minutes, and recheck blood sugar. If it is still below 70 mg/dL, repeat another 15 grams. Do NOT give food or liquid if the person is unconscious; administer injectable or nasal Glucagon and call emergency services."
    },
    {
        "id": "MED-005",
        "category": "Cardiology",
        "condition": "Hypertension (High Blood Pressure)",
        "source": "American Heart Association / NIH NHLBI",
        "question": "What are the blood pressure categories and what constitutes a Hypertensive Crisis?",
        "answer": "Blood pressure categories for adults: 1) Normal: Less than 120/80 mmHg. 2) Elevated: Systolic 120–129 AND Diastolic less than 80 mmHg. 3) Hypertension Stage 1: Systolic 130–139 OR Diastolic 80–89 mmHg. 4) Hypertension Stage 2: Systolic 140 or higher OR Diastolic 90 or higher mmHg. 5) HYPERTENSIVE CRISIS (Medical Emergency): Blood pressure higher than 180/120 mmHg. If accompanied by chest pain, shortness of breath, back pain, numbness, or difficulty speaking, seek emergency hospital care immediately."
    },
    {
        "id": "MED-006",
        "category": "Gastroenterology",
        "condition": "Acute Gastroenteritis & Dehydration",
        "source": "WHO / CDC Clinical Guidance",
        "question": "When are antibiotics needed for acute diarrhea and how should ORS be prepared?",
        "answer": "Most cases of acute watery diarrhea are viral (such as Rotavirus or Norovirus) and do NOT require antibiotics. Inappropriate antibiotic use can worsen diarrhea and cause resistance. Antibiotics are only indicated for bloody diarrhea (dysentery), suspected Cholera with severe dehydration, or laboratory-confirmed Giardiasis. Primary treatment is Low-Osmolarity Oral Rehydration Salts (ORS) dissolved in 1 liter of clean boiled/cooled water, plus Zinc supplementation (20 mg daily for 10–14 days for children over 6 months)."
    },
    {
        "id": "MED-007",
        "category": "Respiratory Medicine",
        "condition": "Bronchial Asthma",
        "source": "NIH National Heart, Lung, and Blood Institute (NHLBI)",
        "question": "What is the difference between a Rescue Inhaler and a Controller Inhaler in Asthma?",
        "answer": "1) Rescue Inhalers (Short-Acting Beta Agonists like Salbutamol / Albuterol): Used for rapid relief during an acute asthma attack to relax airway muscles within minutes. They do not reduce long-term airway inflammation. 2) Controller Inhalers (Inhaled Corticosteroids like Budesonide or Fluticasone): Taken daily even when feeling well to prevent inflammation and mucus buildup. If a patient needs their rescue inhaler more than twice a week, their asthma is considered poorly controlled and requires physician review."
    },
    {
        "id": "MED-008",
        "category": "Emergency & Environmental",
        "condition": "Heat Exhaustion vs. Heatstroke",
        "source": "CDC / NIH Emergency Medicine",
        "question": "How do you differentiate between Heat Exhaustion and Heatstroke, and what is the first aid?",
        "answer": "Heat Exhaustion presents with heavy sweating, cold/pale/clammy skin, fast weak pulse, nausea, and dizziness; move the person to a cool room, loosen clothing, and give sips of water. HEATSTROKE is a life-threatening emergency: core body temperature rises above 103°F (39.4°C), skin becomes hot, red, dry or damp, pulse is fast and strong, accompanied by confusion or unconsciousness. First Aid for Heatstroke: Call emergency services immediately, cool the person rapidly with cold water immersion or ice packs on the neck, armpits, and groin, and do NOT give fluids to drink."
    }
]

def load_or_create_medical_df(csv_path="data/real_world_medical_faqs.csv") -> pd.DataFrame:
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    if not os.path.exists(csv_path):
        df = pd.DataFrame(CLINICAL_FAQS)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
    return df