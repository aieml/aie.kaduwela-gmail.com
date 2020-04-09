from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
import keras.models as models
import keras.layers as layers
import keras.optimizers as optimizers
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Convolution2D, ZeroPadding2D, AveragePooling2D
import keras.backend as K
import joblib
import base64
import cv2
from keras.layers.advanced_activations import PReLU

from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

scaler_diabetic_complications = MinMaxScaler()
scaler_diabetic_risk = MinMaxScaler()

data_diabetic_complications=np.load('data_diabetic_complications.npy')
data_diabetic_risk=np.load('data_diabetic_risk.npy')

scaler_diabetic_complications.fit(data_diabetic_complications)
scaler_diabetic_risk.fit(data_diabetic_risk)

def load_stress_model():

	img_rows, img_cols = 48, 48
	model = Sequential()
	
	model.add(Convolution2D(64, 5, 5, border_mode='valid',input_shape=(img_rows, img_cols, 1)))
	model.add(PReLU(init='zero', weights=None))
	model.add(ZeroPadding2D(padding=(2, 2), dim_ordering='tf'))
	model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
	
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(64, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
	
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(128, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(Convolution2D(128, 3, 3))
	model.add(PReLU(init='zero', weights=None))
	
	model.add(ZeroPadding2D(padding=(1, 1), dim_ordering='tf'))
	model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
	
	model.add(Flatten())
	model.add(Dense(1024))
	model.add(PReLU(init='zero', weights=None))
	model.add(Dropout(0.2))
	model.add(Dense(1024))
	model.add(PReLU(init='zero', weights=None))
	model.add(Dropout(0.2))
	
	model.add(Dense(6))

	model.add(Activation('softmax'))

	ada = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',optimizer=ada,metrics=['accuracy'])
	
	model.load_weights('emotion_recognition.h5')
	
	return model


def food_model():
    
    print('hit load model')
    model = models.Sequential()
    
    model.add(Conv2D(256, (3, 3), input_shape=(50,50,1)))
    print('middle model')
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten()) 
    
    model.add(Dense(64))
    
    model.add(Dense(9))
    model.add(Activation('sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.load_weights('Food_V1.h5')
    
    print('last model')
    return model
 

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
    
    print("model hit")
    print(User_json)
    
    age = int(User_json['age'])
    sex = int(User_json['sex'])
    height = float(User_json['height'])
    weight = float(User_json['weight'])
    bmi = float(User_json['bmi'])
    sbp = float(User_json['sbp'])
    dbp = float(User_json['dbp'])
    fpg1 = float(User_json['fpg1'])
    chol = float(User_json['chol'])
    trigl = float(User_json['trigl'])
    hdl = float(User_json['hdl'])
    ldl = float(User_json['ldl'])
    alt = float(User_json['alt'])
    ast = float(User_json['ast'])
    bun = float(User_json['bun'])
    ccr = float(User_json['ccr'])
    fpg2 = float(User_json['fpg2'])
    year = float(User_json['year'])
    smoke = int(User_json['smoke'])
    drink = float(User_json['drink'])
    family = float(User_json['family'])
    
    test_data=[age,sex,height,weight,bmi,sbp,dbp,fpg1,chol,trigl,hdl,ldl,alt,ast,bun,ccr,fpg2,year,smoke,drink,family]
    
    test_data=scaler_diabetic_risk.transform([test_data])
    
    print(test_data)
    
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
    
@app.route('/food',methods = ['POST', 'GET'])
def food():
    
    print('came here')
    model = food_model()
    
    print('after model')
    
    target_dict={0:'Burger', 1: 'Hoppers', 2: 'Koththu', 3: 'Noodles', 4: 'Pizza', 5: 'Rice', 6: 'Rolls', 7: 'Rottie', 8: 'Samosa'}
    User_json = request.json
    
    encrypted_string =  User_json['base_string']
    
    imgdata = base64.b64decode(encrypted_string)
    filename = 'test_image.jpg' 
    with open(filename, 'wb') as f:
        f.write(imgdata)
    
    img = cv2.imread('test_image.jpg')
    test_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (height, width) = img.shape[:2]
    
    #imgdata[0:50,0:width] = [0,0,255]
    #imgdata[50:80,0:100] = [0,255,255]
    
    
    test_img=cv2.resize(test_img,(50,50))
    test_img=test_img/255.0
    test_img=test_img.reshape(-1,50,50,1)
    result=model.predict([test_img])
        
    K.clear_session()
    label=target_dict[np.argmax(result)]
        
    
        
    results = [
    {
        "predictedResult": label
    }
    ]
    return jsonify(results=results)
    
@app.route('/stress',methods = ['POST', 'GET'])	
def stress():

	model=load_stress_model()
	emotions = {0:'Angry',1:'Fear',2:'Happy',3:'Sad',4:'Surprised',5:'Neutral'}
	User_json = request.json
	
	encrypted_string =  User_json['base_string']
	
	imgdata = base64.b64decode(encrypted_string)
	filename = 'emotion_image.jpg' 
    with open(filename, 'wb') as f:
        f.write(imgdata)
	
	faceCascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	frame=cv2.imread('emotion_image.jpg')
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=7,minSize=(100, 100))
	(height,width)=frame.shape[:2]
	
	frame[0:50,0:230]=[100]
	frame[0:50,230:width]=[100]
	frame[50:80,0:100]=[50]
	
	try:
		if (len(faces)) > 0:
			for x, y, width, height in faces:
				cropped_face = gray[y:y + height,x:x + width]
				test_image = cv2.resize(cropped_face, (48, 48))
				test_image = test_image.reshape([-1,48,48,1])

				test_image = np.multiply(test_image, 1.0 / 255.0)

				probab = model.predict(test_image)[0]
				
				label = np.argmax(probab)
				probab_predicted = int(probab[label])
				predicted_emotion = emotions[label]
				print(predicted_emotion)
				#predicted_emotion should be sent
				
				acc=np.max(probab)
				cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255,0),2)
				cv2.putText(frame,predicted_emotion,(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
				#cv2.putText(imge,cords,(250,35),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
				cv2.putText(frame,'acc:'+str(round(acc,2)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
				cv2.imwrite('emotion_result.jpg',frame)
				#predicted_emotion should be sent
			K.clear_session()
			
	results = [
    {
        "predictedResult": predicted_emotion
    }
    ]
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



