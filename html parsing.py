pip install flask
pip install requests


from flask import Flask, render_template, redirect, request, url_for
from flask import request

def create()
    if request.method == 'POST':
        url = request.form['url']
        if url == None:
            print ("no url is given yet.")
        else:
            print ("url is given.")


creat()
