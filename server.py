from flask import Flask, render_template,request,jsonify
from main_utils import return_df_with_similarities_and_sim_score
app = Flask(__name__)

@app.route('/')
def hellobello():
    return render_template('index.html')

# @app.route('/front_end')
# def STS():
#     return render_template('index.html')

@app.route('/front_end', methods=['POST', 'GET'])
def STS():
    if request.method == 'POST':
        text1 = request.form['text1']
        text2 = request.form['text2']
        
        request_body = jsonify({'text1': text1, 'text2': text2})
        df_with_similarities,sim_score = return_df_with_similarities_and_sim_score(text1,text2)
        # response_body = len(text1) + len(text2)
        return jsonify({'similarity score': sim_score})
    else :
        return "Gettyyy"
    
