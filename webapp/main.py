# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template, request

from scraper import scrapeBBC

# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.


@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return render_template('index.html')

@app.post('/parse')
# ‘/’ URL is bound with hello_world() function.
def do_parse():
    urls = request.form['urls'].split('\r\n')
    sents = [scrapeBBC(url) for url in urls]
    sents = [item for sublist in sents for item in sublist]

    return render_template('index.html',
                        urls=urls,
                        num_sents=len(sents),
                        num_urls=len(urls))

# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
