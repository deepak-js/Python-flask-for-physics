from flask import Flask, redirect, url_for, render_template, request,make_response,send_file,Markup
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import sympy as smp
from sympy import symbols
from sympy import Symbol
from math import *
import io
import base64
from flask import Response
import seaborn as sns
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import pygal as pygal
from numpy.linalg import eig
from numpy import linalg as la
import os
import pandas as pd



app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", title='home')


@app.route("/layout")
def layout():
    return render_template("layout.html", title='layout')


@app.route("/vectors-home")
def vectors_home():
    return render_template("vectors-home.html", title='Vectors Home')


@app.route("/matrix")
def matrix_home():
    return render_template("Matrix.html", title='Matrix Home')


@app.route("/optics")
def optics():
    return render_template("optics.html", title='Optics Home')


@app.route("/xrd",methods=["GET","POST"])
def xrd():
    if request.method == "POST":
        file = request.files["csvfile"]
        if not os.path.isdir('static'):
            os.mkdir('static')
        filepath = os.path.join('static',file.filename)
        file.save(filepath)
        #data = "The file name of the upload file is: {}".format(file.filename)
        return render_template('xrd-analysis.html')
    else:
        return render_template('xrd.html')


plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
@app.route('/print-plot')
def plot_png():
   fig = Figure()
   axis = fig.add_subplot(1, 1, 1)
   data = np.loadtxt("static/data.txt")
   xs = data.transpose()[0]
   ys = data.transpose()[1]
   axis.plot(xs, ys)
   output = io.BytesIO()
   FigureCanvas(fig).print_png(output)
   return Response(output.getvalue(), mimetype='image/png')


@app.route('/average-grain-size')
def grain_size():
    data = np.loadtxt("static/data.txt")
    k = 0.89
    l = 1.54e-10
    B = []
    theta = []
    for i in range(0, len(data)):
        B.append(data[i][0] * np.pi / 180)
        theta.append(1 / 2 * data[i][0] * np.pi / 180)
    D = (k * l) / (B * np.cos(theta))
    e = (B * np.cos(theta)) / 4
    dd = 1 / D ** 2
    user = "Average Grain size = ", np.average(D)
    return redirect(url_for("user", usr=user))


@app.route('/average-dislocation-density')
def dislocation_density():
    data = np.loadtxt("static/data.txt")
    k = 0.89
    l = 1.54e-10
    B = []
    theta = []
    for i in range(0, len(data)):
        B.append(data[i][0] * np.pi / 180)
        theta.append(1 / 2 * data[i][0] * np.pi / 180)
    D = (k * l) / (B * np.cos(theta))
    e = (B * np.cos(theta)) / 4
    dd = 1 / D ** 2
    user = "Average Dislocation Denisty = ", np.average(dd)
    return redirect(url_for("user", usr=user))

@app.route('/average-micro-strain-term')
def micro_strain_term():
    data = np.loadtxt("static/data.txt")
    k = 0.89
    l = 1.54e-10
    B = []
    theta = []
    for i in range(0, len(data)):
        B.append(data[i][0] * np.pi / 180)
        theta.append(1 / 2 * data[i][0] * np.pi / 180)
    D = (k * l) / (B * np.cos(theta))
    e = (B * np.cos(theta)) / 4
    dd = 1 / D ** 2
    user = "AVerage Micro strain term = ", np.average(e)
    return redirect(url_for("user", usr=user))


#Vectors
from numpy.linalg import norm
@app.route("/vectors",methods=["POST", "GET"])
def get_vector():
    if request.method == "POST":
        u1 = float(request.form["u1"])
        u2 = float(request.form["u2"])
        u3 = float(request.form["u3"])

        v1 = float(request.form["v1"])
        v2 = float(request.form["v2"])
        v3 = float(request.form["v3"])
        vector_row = np.array([[u1,u2,u3]])
        vector_column = np.array([[v1,v2,v3]])
        print(vector_row.shape)
        print(vector_column.shape)
        user = np.cross(vector_row,vector_column)
        return redirect(url_for("user", usr=user))
    else:
        return render_template("vectors.html")


@app.route("/vector-normalisation",methods=["POST", "GET"])
def vector_norm():
    if request.method == "POST":
        u1 = float(request.form["u1"])
        u2 = float(request.form["u2"])
        u3 = float(request.form["u3"])
        u4 = float(request.form["u4"])
        vector_row = np.array([[u1,u2,u3,u4]])
        print(vector_row.shape)
        user = norm(vector_row)
        return redirect(url_for("user", usr=user))
    else:
        return render_template("vector-norm.html")


from numpy import arccos, dot
@app.route("/vector-angle",methods=["POST", "GET"])
def vector_angle():
    if request.method == "POST":
        u1 = float(request.form["u1"])
        u2 = float(request.form["u2"])
        u3 = float(request.form["u3"])

        v1 = float(request.form["v1"])
        v2 = float(request.form["v2"])
        v3 = float(request.form["v3"])
        v = np.array([[u1,u2,u3]])
        w = np.array([[v1,v2,v3]])
        theta = arccos(dot(v, w.T)/(norm(v)*norm(w)))
        user = theta
        return redirect(url_for("user", usr=user))
    else:
        return render_template("vector-angle.html")


@app.route("/vector-plot",methods=["POST", "GET"])
def vector_plot():
    if request.method == "POST":
        u1 = float(request.form["u1"])
        u2 = float(request.form["u2"])
        u3 = float(request.form["u3"])

        v1 = float(request.form["v1"])
        v2 = float(request.form["v2"])
        v3 = float(request.form["v3"])
        plt.rcParams["figure.figsize"] = [7.00, 3.50]
        plt.rcParams["figure.autolayout"] = True
        data = np.array([[u1, v1], [u2, v2], [u3, v3]])
        origin = np.array([[0, 0, 0], [0, 0, 0]])
        plt.quiver(*origin, data[:, 0], data[:, 1], color=['black', 'red', 'green'], scale=15)
        plt.show()
        canvas = FigureCanvas(fig)
        img = io.BytesIO()
        fig.savefig(img)
        img.seek(0)
        return send_file(img, mimetype='img/png')
        return redirect(url_for("visualize"))
    else:
        return render_template("vector-plot.html")


@app.route("/addition", methods=["POST", "GET"])
def addition():
    if request.method == "POST":
        user1 = int(request.form["nm"])
        user2 = int(request.form["mn"])
        user = user1 + user2
        return redirect(url_for("user", usr=user))
    else:
        return render_template("addition.html")


@app.route("/subtraction", methods=["POST", "GET"])
def subtraction():
    if request.method == "POST":
        user1 = int(request.form["nm"])
        user2 = int(request.form["mn"])
        user = user1 - user2
        return redirect(url_for("user", usr=user))
    else:
        return render_template("subtraction.html")


@app.route("/projectile-motion", methods=["POST", "GET"])
def projectile():
    g=9.8
    if request.method == "POST":
        velocity = float(request.form["velocity"])
        theta = float(request.form["theta"])
        v_x = float(velocity * np.cos(theta * np.pi / 180))
        v_y = float(velocity * np.sin(theta * np.pi / 180))
        R = float(velocity ** 2 * np.sin(2 * theta * np.pi / 180) / g)
        H = float(velocity ** 2 * np.sin(theta * np.pi / 180) ** 2 / (2 * g))
        T = float(2 * velocity * np.sin(theta * np.pi / 180) / g)
        first= "Velocity along the x-axis: ",v_x,"velocity along the y-axis: ",v_y, "Range: ",R
        second= "Maximum Height: ",H,"Time of Flight: ",T
        final = first+second
        t= 1
        x= v_x*t
        y= v_y*t - 4.9*t**2
        return redirect(url_for("user", usr=final))
    else:
        return render_template("projectile-motion.html")


@app.route("/integration" , methods=["POST", "GET"])
def integrate():
    if request.method == "POST":
        x = smp.symbols("x")
        y = request.form["int"]
        print(y)
        integrate = smp.integrate(y,x)
        print(integrate)
        return redirect(url_for("user", usr=integrate))
    else:
        return render_template("integration.html")


from scipy.misc import derivative


@app.route("/numerical-differentiation" , methods=["POST", "GET"])
def numerical_differentation():
    if request.method == "POST":
        x = smp.symbols("x")
        y = request.form["int"]
        z = request.form["z"]
        print(y)
        diff = derivative(y,z)
        print(diff)
        return redirect(url_for("user", usr=integrate))
    else:
        return render_template("numerical-differentiation.html")


#Vectors
@app.route("/vector-addition", methods=["POST", "GET"])
def vector():
    if request.method == "POST":
        u1 = int(request.form["u1"])
        u2 = int(request.form["u2"])
        u3 = int(request.form["u3"])
        u4 = int(request.form["u4"])
        u5 = int(request.form["u5"])

        v1 = int(request.form["v1"])
        v2 = int(request.form["v2"])
        v3 = int(request.form["v3"])
        v4 = int(request.form["v4"])
        v5 = int(request.form["v5"])



        return redirect(url_for("user", usr=user))
    else:
        return render_template("subtraction.html")



from numpy import exp, cos, linspace
import matplotlib.pyplot as plt
import os, time, glob


def damped_vibrations(t, A, b, w):
    return A*exp(-b*t)*cos(w*t)

def compute(A, b, w, T, resolution=500):
#    Return filename of plot of the damped_vibration function.
    t = linspace(0, T, resolution+1)
    u = damped_vibrations(t, A, b, w)
    plt.figure()  # needed to avoid adding curves in plot
    plt.plot(t, u)
    plt.title('A=%g, b=%g, w=%g' % (A, b, w))
    if not os.path.isdir('static'):
        os.mkdir('static')
    else:
        # Remove old plot files
        for filename in glob.glob(os.path.join('static', '*.png')):
            os.remove(filename)
    # Use time since Jan 1, 1970 in filename in order make
    # a unique filename that the browser has not chached
    plotfile = os.path.join('static', str(time.time()) + '.png')
    plt.savefig(plotfile)
    return plotfile

from wtforms import Form, FloatField, validators
from math import pi

class InputForm(Form):
    A = FloatField(
        label='amplitude (m)', default=1.0,
        validators=[validators.InputRequired()])
    b = FloatField(
        label='damping factor (kg/s)', default=0,
        validators=[validators.InputRequired()])
    w = FloatField(
        label='frequency (1/s)', default=2*pi,
        validators=[validators.InputRequired()])
    T = FloatField(
        label='time interval (s)', default=18,
        validators=[validators.InputRequired()])

from model import InputForm
from flask import Flask, render_template, request


@app.route('/vib1', methods=['GET', 'POST'])
def index_plot():
    form = InputForm(request.form)
    if request.method == 'POST' and form.validate():
        result = compute(form.A.data, form.b.data,
                         form.w.data, form.T.data)
    else:
        result = None

    return render_template('view.html', form=form, result=result)


#Calculating the Area and Perimeter of a Rectangle

@app.route("/rectangle", methods=["POST", "GET"])
def rectangle():
    if request.method == "POST":
        a = int(request.form["a"])
        b = int(request.form["b"])
        area = a*b
        perimeter = 2 * (a+b)
        user = "Area : ",area,"Perimeter: ",perimeter
        return redirect(url_for("user", usr=user))
    else:
        return render_template("rectangle.html")


#Matrix

@app.route("/eigen-values", methods=["POST", "GET"])
def eigen_values():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        A=np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        w,v = np.linalg.eig(A)
        user = "the eigen values: ",w,"AND eigen vectors: ",v
        return redirect(url_for("user", usr=user))
    else:
        return render_template("eigen-values.html")

@app.route("/eigen2d", methods=["POST", "GET"])
def eigen2d():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        A =np.array([[a11,a12],[a21,a22]])
        w,v = np.linalg.eig(A)
        user = "the eigen values: ",w,"AND eigen vectors: ",v
        return redirect(url_for("user", usr=user))
    else:
        return render_template("eigen2d.html")

@app.route("/eigen4d", methods=["POST", "GET"])
def eigen4d():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a14 = float(request.form["a14"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a24 = float(request.form["a24"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        a34 = float(request.form["a34"])
        a41 = float(request.form["a41"])
        a42 = float(request.form["a42"])
        a43 = float(request.form["a43"])
        a44 = float(request.form["a44"])
        A =np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]])
        w,v = np.linalg.eig(A)
        user = "the eigen values: ",w,"AND eigen vectors: ",v
        return redirect(url_for("user", usr=user))
    else:
        return render_template("eigen4d.html")


@app.route("/inverse-matrix", methods=["POST", "GET"])
def inverse_matrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        A=np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        w = np.linalg.inv(A)
        user = "the inverse matrix: ",w
        return redirect(url_for("user", usr=user))
    else:
        return render_template("inverse-matrix.html")

@app.route("/inverse2d", methods=["POST", "GET"])
def inverse2d():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        A =np.array([[a11,a12],[a21,a22]])
        w = np.linalg.inv(A)
        user = "the inverse matrix: ",w
        return redirect(url_for("user", usr=user))
    else:
        return render_template("inverse2d.html")

@app.route("/inverse4d", methods=["POST", "GET"])
def inverse4d():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a14 = float(request.form["a14"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a24 = float(request.form["a24"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        a34 = float(request.form["a34"])
        a41 = float(request.form["a41"])
        a42 = float(request.form["a42"])
        a43 = float(request.form["a43"])
        a44 = float(request.form["a44"])
        A =np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]])
        w = np.linalg.inv(A)
        user = "the inverse matrix: ",w
        return redirect(url_for("user", usr=user))
    else:
        return render_template("inverse4d.html")


@app.route("/matrix-multiplication", methods=["POST", "GET"])
def matrix_multi():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        b11 = float(request.form["b11"])
        b12 = float(request.form["b12"])
        b13 = float(request.form["b13"])
        b21 = float(request.form["b21"])
        b22 = float(request.form["b22"])
        b23 = float(request.form["b23"])
        b31 = float(request.form["b31"])
        b32 = float(request.form["b32"])
        b33 = float(request.form["b33"])
        A=np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        B = np.array([[b11, b12, b13], [b21, b22, b23], [b31, b32, b33]])
        w = np.matmul(A,B)
        user = "Multiplication matrix: ",w
        return redirect(url_for("user", usr=user))
    else:
        return render_template("matrix-multiplication.html")

@app.route("/matrixmulti2d", methods=["POST", "GET"])
def matrixmulti2d():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        b11 = float(request.form["b11"])
        b12 = float(request.form["b12"])
        b21 = float(request.form["b21"])
        b22 = float(request.form["b22"])
        A =np.array([[a11,a12],[a21,a22]])
        B = np.array([[b11, b12], [b21, b22]])
        w = np.matmul(A,B)
        user = "matrix multiplication: ",w
        return redirect(url_for("user", usr=user))
    else:
        return render_template("matrixmulti2d.html")

@app.route("/matrixmulti4d", methods=["POST", "GET"])
def matrixmulti4d():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a14 = float(request.form["a14"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a24 = float(request.form["a24"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        a34 = float(request.form["a34"])
        a41 = float(request.form["a41"])
        a42 = float(request.form["a42"])
        a43 = float(request.form["a43"])
        a44 = float(request.form["a44"])
        b11 = float(request.form["b11"])
        b12 = float(request.form["b12"])
        b13 = float(request.form["b13"])
        b14 = float(request.form["b14"])
        b21 = float(request.form["b21"])
        b22 = float(request.form["b22"])
        b23 = float(request.form["b23"])
        b24 = float(request.form["b24"])
        b31 = float(request.form["b31"])
        b32 = float(request.form["b32"])
        b33 = float(request.form["b33"])
        b34 = float(request.form["b34"])
        b41 = float(request.form["b41"])
        b42 = float(request.form["b42"])
        b43 = float(request.form["b43"])
        b44 = float(request.form["b44"])
        A =np.array([[a11,a12,a13,a14],[a21,a22,a23,a24],[a31,a32,a33,a34],[a41,a42,a43,a44]])
        B = np.array([[b11, b12, b13, b14], [b21, b22, b23, b24], [b31, b32, b33, b34], [b41, b42, b43, b44]])
        w = np.matmul(A,B)
        user = "matrix multiplication: ",w
        return redirect(url_for("user", usr=user))
    else:
        return render_template("matrixmulti4d.html")


@app.route("/det-matrix", methods=["POST", "GET"])
def det_matrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        A=np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        w = np.linalg.det(A)
        user = "The Determinant of The Matrix is ",w

        return redirect(url_for("user", usr=user))
    else:
        return render_template("det-matrix.html")

@app.route("/det-2d-matrix", methods=["POST", "GET"])
def det_2dmatrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])

        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])

        A=np.array([[a11,a12],[a21,a22]])
        w = np.linalg.det(A)
        user = "The Determinant of The Matrix is ",w

        return redirect(url_for("user", usr=user))
    else:
        return render_template("det-2d-matrix.html")

@app.route("/det-4d-matrix", methods=["POST", "GET"])
def det_4dmatrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a14 = float(request.form["a14"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a24 = float(request.form["a24"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        a34 = float(request.form["a34"])
        a41 = float(request.form["a41"])
        a42 = float(request.form["a42"])
        a43 = float(request.form["a43"])
        a44 = float(request.form["a44"])
        A = np.array([[a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]])
        w = np.linalg.det(A)
        user = "The Determinant of The Matrix is ",w

        return redirect(url_for("user", usr=user))
    else:
        return render_template("det-4d-matrix.html")


@app.route("/rank-matrix", methods=["POST", "GET"])
def rank_matrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        A=np.array([[a11,a12,a13],[a21,a22,a23],[a31,a32,a33]])
        I = np.linalg.matrix_rank(A)

        user = "The Rank of The Matrix is ",I

        return redirect(url_for("user", usr=user))
    else:
        return render_template("rank-matrix.html")

@app.route("/rank-2d-matrix", methods=["POST", "GET"])
def rank_2dmatrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])

        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])

        A=np.array([[a11,a12],[a21,a22]])
        I = np.linalg.matrix_rank(A)
        user = "The Rank of The Matrix is ",I

        return redirect(url_for("user", usr=user))
    else:
        return render_template("rank-2d-matrix.html")

@app.route("/rank-4d-matrix", methods=["POST", "GET"])
def rank_4dmatrix():
    if request.method == "POST":
        a11 = float(request.form["a11"])
        a12 = float(request.form["a12"])
        a13 = float(request.form["a13"])
        a14 = float(request.form["a14"])
        a21 = float(request.form["a21"])
        a22 = float(request.form["a22"])
        a23 = float(request.form["a23"])
        a24 = float(request.form["a24"])
        a31 = float(request.form["a31"])
        a32 = float(request.form["a32"])
        a33 = float(request.form["a33"])
        a34 = float(request.form["a34"])
        a41 = float(request.form["a41"])
        a42 = float(request.form["a42"])
        a43 = float(request.form["a43"])
        a44 = float(request.form["a44"])
        A = np.array([[a11, a12, a13, a14], [a21, a22, a23, a24], [a31, a32, a33, a34], [a41, a42, a43, a44]])
        I = np.linalg.matrix_rank(A)
        user = "The Rank of The Matrix is ",I
        return redirect(url_for("user", usr=user))
    else:
        return render_template("rank-4d-matrix.html")

"""Reflection and Refraction"""
from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_masks_X import Scalar_mask_X
from diffractio.scalar_masks_XZ import Scalar_mask_XZ
from diffractio.scalar_sources_X import Scalar_source_X
from matplotlib import rcParams

@app.route("/gauss-beam",methods=["POST", "GET"])
def gauss_beam():
    if request.method == "POST":
        plt.rcParams['figure.dpi'] = 125
        a = float(request.form["a"])
        b = float(request.form["b"])
        c = float(request.form["c"])
        d = float(request.form["d"])
        e = float(request.form["e"])
        f = float(request.form["f"])

        x0 = np.linspace(a * um, b * um, 1024)
        z0 = np.linspace(c * um, d * um, 1024)

        wavelength = e * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=0 * um, z0=100 * um, w0= f * um, theta=0 * degrees)
        u0.draw(filename="static/vacuum-1.png")

        x0 = np.linspace((-50+a) * um, (50+b) * um, 512)
        z0 = np.linspace(c * um, d * um, 512)

        wavelength = e * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=0 * um, z0=100 * um, w0=(f+5) * um, theta=0 * degrees)
        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)
        u1.incident_field(u0)
        u1.WPM(verbose=True)
        u1.cut_resample(x_limits=(-50, 50), num_points=(512, 512))
        u1.draw(kind='phase', scale='scaled',filename="static/propogation.png")
        u1.draw(kind='intensity', logarithm=1, scale='scaled',filename="static/light-1.png");
        #u1.draw(filename="static/gauss.png")
        return render_template("gauss-beam-image.html")
    else:
        return render_template("gauss-beam.html")

"""2. Reflection and refraction """

@app.route("/ref-refr",methods=["POST", "GET"])
def reflection_refraction():
    if request.method == "POST":
        rcParams['figure.figsize'] = (7, 5)
        rcParams['figure.dpi'] = 125
        a = float(request.form["a"])
        b = float(request.form["b"])

        x0 = np.linspace(-100 * um, 100 * um, 2048)
        z0 = np.linspace(-100 * um, 100 * um, 2048)

        wavelength = a * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=0 * um, z0=100 * um, w0=15 * um, theta=0 * degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1)
        u1.incident_field(u0)
        u1.semi_plane(r0=(0, 0), refraction_index=b, angle=60 * degrees)

        u1.draw_refraction_index(filename="static/refractive-index-2.png")

        u1.WPM(verbose=False)

        u1.draw(kind='phase', draw_borders=True, filename="static/propogation-2.png");

        u1.draw(kind='intensity', logarithm=True, draw_borders=True, filename="static/refractive-2.png");


        #u1.draw(filename="static/gauss.png")
        return render_template("reflection-refraction-image.html")
    else:
        return render_template("reflection-refraction.html")


@app.route("/refraction",methods=["POST", "GET"])
def refraction():
    if request.method == "POST":
        rcParams['figure.figsize'] = (7, 5)
        rcParams['figure.dpi'] = 125
        a = float(request.form["a"])
        b = float(request.form["b"])

        x0 = np.linspace(-110 * um, 110 * um, 2048)
        z0 = np.linspace(-110 * um, 110 * um, 2048)
        wavelength = a * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=25 * um, z0=150 * um, w0=40 * um, theta=0 * degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.layer(r0=(50, 0), depth=80 * um, refraction_index=b, angle=45 * degrees)

        u1.draw_refraction_index(filename="static/refractive-index-3.png")

        u1.WPM(verbose=False)
        u1.draw(kind='intensity', draw_borders=True, filename="static/propogation-3.png");

        u1.draw_profiles_interactive(kind='intensity', logarithm=False, normalize=False);

        return render_template("refraction-image.html")
    else:
        return render_template("refraction.html")

@app.route("/total-refraction",methods=["POST", "GET"])
def total_refraction():
    if request.method == "POST":
        rcParams['figure.figsize'] = (7, 5)
        rcParams['figure.dpi'] = 125
        a = float(request.form["a"])
        b = float(request.form["b"])

        x0 = np.linspace(-100 * um, 100 * um, 2048)
        z0 = np.linspace(-100 * um, 100 * um, 2048)

        wavelength = a * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=0 * um, z0=100 * um, w0=15 * um, theta=0 * degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength, n_background=1.5)
        u1.incident_field(u0)
        u1.semi_plane(r0=(0, 0), refraction_index=b, angle=60 * degrees)

        u1.draw_refraction_index(filename="static/refractive-index-4.png")

        u1.WPM(verbose=False)

        u1.draw(kind='phase', draw_borders=True, filename="static/propogation-4.png")
        plt.ylim(ymin=-60)
        plt.xlim(xmax=75);

        u1.draw(kind='intensity', logarithm=True, draw_borders=True, filename="static/light-4.png");
        plt.xlim(-50, 50)
        plt.ylim(-30, 30)


        return render_template("total-refraction-image.html")
    else:
        return render_template("total-refraction.html")


@app.route("/total-refraction-layer", methods=["POST", "GET"])
def total_refraction_layer():
    if request.method == "POST":
        rcParams['figure.figsize'] = (7, 5)
        rcParams['figure.dpi'] = 125
        a = float(request.form["a"])
        b = float(request.form["b"])

        x0 = np.linspace(-150 * um, 300 * um, 2048)
        z0 = np.linspace(-150 * um, 500 * um, 2048)
        wavelength = a * um
        u0 = Scalar_source_X(x=x0, wavelength=wavelength)
        u0.gauss_beam(A=1, x0=-100 * um, z0=150 * um, w0=15 * um, theta=0 * degrees)

        u1 = Scalar_mask_XZ(x=x0, z=z0, wavelength=wavelength)
        u1.incident_field(u0)
        u1.layer(r0=(50, 0), depth=100 * um, refraction_index=b, angle=60 * degrees)
        u1.draw_refraction_index(filename="static/refractive-index-5.png")
        u1.filter_refraction_index(type_filter=2, pixels_filtering=3);

        u1.BPM(verbose=False)
        u1.draw(kind='intensity', logarithm=True, draw_borders=True, filename="static/propogation-5.png");

        return render_template("total-refraction-layer-image.html")
    else:
        return render_template("total-refraction-layer.html")


"""Diffraction by different objects"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.utils_drawing import draw_several_fields
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY

rcParams['figure.figsize']=(7,5)
rcParams['figure.dpi']=125

@app.route("/single-slit",methods=["POST","GET"])
def single_slit():
    if request.method == "POST":

        a = float(request.form["a"])
        b = float(request.form["b"])
        num_pixels = 512

        length = 100 * um
        x0 = np.linspace(-length / 2, length / 2, num_pixels)
        y0 = np.linspace(-length / 2, length / 2, num_pixels)
        wavelength = a * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
        # u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.slit(x0=0, size=b * um, angle=0 * degrees)

        u2 = u1 * t1

        u3 = u2.RS(z=25 * um, new_field=True)

        u4 = u2.RS(z=100 * um, new_field=True)

        draw_several_fields((u2, u3, u4), titles=('mask', '25 um', '100 um'))

        u2.draw(filename="static/u2-single-slit.png")
        u3.draw(filename="static/u3-single-slit.png")
        u4.draw(filename="static/u4-single-slit.png")

        return render_template("single-slit-image.html")
    else:
        return render_template("single-slit.html")

@app.route("/double-slit",methods=["POST","GET"])
def double_slit():
    if request.method == "POST":

        a = float(request.form["a"])
        b = float(request.form["b"])
        num_pixels = 512

        length = 100 * um
        x0 = np.linspace(-length / 2, length / 2, num_pixels)
        y0 = np.linspace(-length / 2, length / 2, num_pixels)
        wavelength = a * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
        # u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.double_slit(x0=0, size=b * um, separation=10 * um, angle=0 * degrees)

        u2 = u1 * t1
        u3 = u2.RS(z=100 * um, new_field=True)

        u4 = u2.RS(z=200 * um, new_field=True)

        draw_several_fields((u2, u3, u4), titles=('mask', '100 um', '200 um'))
        u2.draw(filename="static/u2-double-slit.png")
        u3.draw(filename="static/u3-double-slit.png")
        u4.draw(filename="static/u4-double-slit.png")
        return render_template("double-slit-image.html")
    else:
        return render_template("double-slit.html")


@app.route("/circular-slit",methods=["POST","GET"])
def circular_slit():
    if request.method == "POST":
        a = float(request.form["a"])
        b = float(request.form["b"])
        num_pixels = 512

        length = 100 * um
        x0 = np.linspace(-length / 2, length / 2, num_pixels)
        y0 = np.linspace(-length / 2, length / 2, num_pixels)
        wavelength = a * um

        u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
        u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)

        t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
        t1.circle(
            r0=(0 * um, 0 * um), radius=(b * um, b * um), angle=0 * degrees)

        u2 = u1 * t1

        u3 = u2.RS(z=100 * um, new_field=True)

        u4 = u2.RS(z=500 * um, new_field=True)

        draw_several_fields((u2, u3, u4), titles=('mask', '100 um', '500 um'), logarithm=True)
        u2.draw(filename="static/u2-circular-slit.png")
        u3.draw(filename="static/u3-circular-slit.png")
        u4.draw(filename="static/u4-circular-slit.png")

        return render_template("circular-slit-image.html")
    else:
        return render_template("circular-slit.html")



@app.route("/<usr>")
def user(usr):
    return f"<h1>{usr}</h1>"









if __name__ == "__main__":
    app.run(debug=True)
