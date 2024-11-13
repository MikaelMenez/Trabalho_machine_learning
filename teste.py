import numpy as np
import os
from flask import Flask, request, render_template, make_response 
import joblib
app = Flask(__name__, static_url_path='/static')
model= joblib.load('./model_KNN.pkl')
@app.route('/')
def display_gui():
    result=''
    return render_template('template.html',**locals())
@app.route('/predict',methods=['POST','GET'])
def verificar():
    raio = float(request.form['mean_radius'])
    textura = float(request.form['mean_texture'])
    perímetro = float(request.form['mean_perimeter'])
    area = float(request.form['mean_area'])
    suavidade = float(request.form['mean_smoothness'])
    teste = np.array([[raio,textura,perímetro,area,suavidade]])

    print(":::::: Dados de Teste ::::::")
    print("raio: {}".format(raio))
    print("textura: {}".format(textura))
    print("perímetro: {}".format(perímetro))
    print("area: {}".format(area))
    print("suavidade: {}".format(suavidade))
    print("\n")

    result = model.predict(teste)[0]
    print("Classe Predita: {}".format(str(result)))

    return render_template('template.html',**locals())

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5500))
    app.run(host='0.0.0.0',port=port,debug=True)