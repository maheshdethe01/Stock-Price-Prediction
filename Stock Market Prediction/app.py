from flask import Flask,render_template,request
import datetime
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
main = pd.read_csv(r"C:\Users\MAHESH DETHE\Downloads\stock_price_data")
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict',methods = ['POST'])
def predict():
    name = request.form['Company']             # company name
    quantity = request.form['Quantity']        # quantity
    buy_date = request.form['Buying Date']     # buying date
    # sell_date = request.form['Selling Date']   # sellig date
    y_buy  = [int(i) for i in buy_date]
    b_date = datetime.date(y_buy[0],y_buy[1],y_buy[2])
    
    #if date is in the data,take the same stock price as in the data else predict
    
    sub = main[main['Name'] == name]
    x = buy_date
    if x in main['Date'].values:
        val_x = sub.loc[sub['Date'] == x,'Open'].values[0]
        return 
    elif b_date >= datetime.date(2022,11,5):
        diff = (b_date - datetime.date(2022,11,4)).days
        #predict
        #Getting the last 100 days records
        m = sub['Open'].tail(100)
        fut_inp = sub['Open'].tail(100).values.reshape(-1,1).reshape(1,-1)
        tmp_inp = list(fut_inp)
        #Creating list of the last 100 data
        tmp_inp = tmp_inp[0].tolist()
        list_output=[]
        n_steps=100
        i=0
        while(i < diff ):
            if(len(tmp_inp)>100):
                fut_inp = np.array(tmp_inp[1:])
                fut_inp=fut_inp.reshape(1,-1)
                fut_inp = fut_inp.reshape((1, n_steps, 1))
                y_pred = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(y_pred[0].tolist())
                tmp_inp = tmp_inp[1:]
                list_output.extend(y_pred.tolist())
                i = i + 1
            else:
                fut_inp = fut_inp.reshape((1, n_steps,1))
                y_pred = model.predict(fut_inp, verbose=0)
                tmp_inp.extend(y_pred[0].tolist())
                list_output.extend(y_pred.tolist())
                i = i + 1
        from sklearn.preprocessing import StandardScaler
        # Using StandardScaler for normalizing data 
        MM1 = StandardScaler()
        scaled_df_1 = MM1.fit_transform(np.array(m).reshape(-1,1))  #df is list of open stock prices of the company
        final = MM1.inverse_transform([[list_output[-1][-1]]])[0][0]
        buy_stock = final*int(quantity)
    return render_template('home.html',price = buy_stock,c = name,q = quantity,b = buy_date)

if __name__ == '__main__':
    app.run(debug=True)
