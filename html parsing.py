pip install flask

from flask import Flask, render_template, redirect, request, url_for
app = Flask(__name__)

@app.route('/')
@app.route('/<string:url>')
def inputTest(url=None):
    return render_template('home.html', url=url)

@app.route('/create',methods=['GET'])
def create(url=None):
    if request.method == 'GET':
        temp = request.form['url']
    else:
        temp = None
    return redirect(url_for('testing html parsing',url=temp))

if __name__ == '__main__':
    app.run()
