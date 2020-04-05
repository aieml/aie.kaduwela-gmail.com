from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from keras.layers import Dropout
import keras.backend as K
import joblib

from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

scaler_diabetic_complications = MinMaxScaler()
scaler_diabetic_risk = MinMaxScaler()

data_diabetic_complications=np.load('data_diabetic_complications.npy')
data_diabetic_risk=np.load('data_diabetic_risk.npy')

scaler_diabetic_complications.fit(data_diabetic_complications)
scaler_diabetic_risk.fit(data_diabetic_risk)

def load_model(weight_file,input_size):
    
    K.clear_session()
    
    model = models.Sequential()
    model.add(layers.Dense(128, input_dim=input_size, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    model.load_weights(weight_file)
    #model.summary()
    return model

def find_label():

	pass

@app.route('/diabetic_complications',methods = ['POST', 'GET'])	
def diabetic_complications():

    User_json = request.json

    sex = int(User_json['sex'])
    age = int(User_json['age'])
    edu_level = int(User_json['edu_level'])
    edu_stand = int(User_json['edu_stand'])
    edu_advance = int(User_json['edu_advance'])
    bmi = float(User_json['bmi'])
    dia_type = int(User_json['dia_type'])
    dmt1 = int(User_json['dmt1'])
    dia_duration = int(User_json['dia_duration'])
    insulin = int(User_json['insulin'])
    medi_treatment = int(User_json['medi_treatment'])
    hba1c = float(User_json['hba1c'])
    hba1cmm = float(User_json['hba1cmm'])
    
    test_data=[sex,age,edu_level,edu_stand,edu_advance,bmi,dia_type,dmt1,dia_duration,insulin,medi_treatment,hba1c,hba1cmm]

    print(test_data)
    
    test_data=scaler_diabetic_complications.transform([test_data])
    
    model_nephropathy=load_model('P1_compliactions_nephropathy.h5',13)
    result_nephropathy=round(model_nephropathy.predict(test_data)[0][0]*100,2)

    model_retinopathy=load_model('P1_compliactions_retinopathy.h5',13)
    result_retinopathy=round(model_retinopathy.predict(test_data)[0][0]*100,2)
    
    model_neuropathy=load_model('P1_compliactions_neuropathy.h5',13)
    result_neuropathy=round(model_neuropathy.predict(test_data)[0][0]*100,2)

    model_foot_ulcer=load_model('P1_compliactions_foot_ulcer.h5',13)
    result_foot_ulcer=round(model_foot_ulcer.predict(test_data)[0][0]*100,2)

    print(result_nephropathy,result_retinopathy,result_neuropathy,result_foot_ulcer)

    results=[{'result_nephropathy':result_nephropathy,'result_retinopathy':result_retinopathy,'result_neuropathy':result_neuropathy,'result_foot_ulcer':result_foot_ulcer}]

    return jsonify(results=results)

@app.route('/diabetic_risk',methods = ['POST', 'GET'])	
def diabetic_risk():

    User_json = request.json
    
    age = int(User_json['age'])
    sex = int(User_json['sex'])
    height = float(User_json['height'])
    weight = float(User_json['weight'])
    bmi = float(User_json['bmi'])
    sbp = float(User_json['sbp'])
    dbp = float(User_json['dbp'])
    fpg = float(User_json['fpg'])
    chol = float(User_json['chol'])
    trigl = float(User_json['trigl'])
    hdl = float(User_json['hdl'])
    ldl = float(User_json['ldl'])
    alt = float(User_json['alt'])
    ast = float(User_json['ast'])
    bun = float(User_json['bun'])
    ccr = float(User_json['ccr'])
    fpg = float(User_json['fpg'])
    year = float(User_json['year'])
    smoke = int(User_json['smoke'])
    drink = float(User_json['drink'])
    family = float(User_json['family'])
    
    test_data=[age,sex,height,weight,bmi,sbp,dbp,fpg,chol,trigl,hdl,ldl,alt,ast,bun,ccr,fpg,year,smoke,drink,family]

    print(test_data)
    
    test_data=scaler_diabetic_risk.transform([test_data])
    
    model=load_model('P1_Diabetic_Risk.h5',21)
    result=round(model.predict(test_data)[0][0]*100,2)

    print(result)

    results=[{'result':result}]

    return jsonify(results=results)

@app.route('/test',methods = ['POST', 'GET'])	
def test():
	User_json = request.json
	name=str(User_json['name'])
	results=[{'result':'Hi '+name}]
	return jsonify(results=results)

@app.route('/meal_plan',methods = ['POST', 'GET'])	
def meal_plan():
    
	User_json = request.json
    
	sex = int(User_json['sex'])
	age = int(User_json['age'])
	bmi = float(User_json['bmi'])
	risk = float(User_json['risk'])

	test_data=np.array([sex,age,bmi,risk]).reshape(1,-1)

	model=joblib.load('2_meal_plan.sav')
	print(model.predict(test_data))
	result=model.predict(test_data)[0]
	results=[{'result':int(result)}]
	return jsonify(results=results)

@app.route('/activity_suggestion',methods = ['POST', 'GET'])	
def activity_suggestion():
    
	User_json = request.json

	sex = int(User_json['sex'])
	age = int(User_json['age'])
	employ = float(User_json['employ'])
	workHours = float(User_json['workHours'])
	freeHours = float(User_json['freeHours'])
	sleepHours = float(User_json['sleepHours'])
	famHours = float(User_json['famHours'])
	ill = float(User_json['ill'])
	emotion = float(User_json['emotion'])
	score = float(User_json['score'])

	test_data=np.array([sex,age,employ,workHours,freeHours,sleepHours,famHours,ill,emotion,score]).reshape(1,-1)

	model=joblib.load('4_activity_suggestion.sav')
	print(model.predict(test_data))
	result=model.predict(test_data)[0]
	results=[{'result':int(result)}]
	return jsonify(results=results)

app.run(debug=True)



