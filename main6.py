""" NN to solve 1st order ODE problems
     1 step method with convergence property
"""

# Import the required modules
from __future__ import division

import numpy as np
import os, sys
import matplotlib.pyplot as plt
import tensorflow as tf
from utils import colored_hook

from scipy.integrate import odeint

# Define a function which calculates the derivative
def func(y, x):
    return x/10 - y
    #return -.5*y
    #return x -.7*y*y
    #return 1.01-y*y

n = 20
xs = np.linspace(0.0,5,n)
dt = 5/(n-1)
y0 = 1.0  # the initial condition

# scipy solver
ys = odeint(func, y0, xs)
y_scipy = np.array(ys).flatten()

# (forward) Euler's Method
y_euler = np.zeros(np.shape(xs))
y_euler[0] = y0
#y_euler2 = np.zeros(np.shape(xs))
#y_euler2[0] = y0
for i in range(1,n):
    x = xs[i-1]
    y = y_euler[i-1]
    y_euler[i] = y + dt*func(y,x)
    #y = y_euler2[i-1]
    #y_euler2[i] = y + np.exp(np.log(dt)+np.log(np.maximum( func(y,x),1e-7)))
    #y_euler2[i] -=    np.exp(np.log(dt)+np.log(np.maximum(-func(y,x),1e-7)))

def _build_solver( x, reuse=False):
    with tf.variable_scope('solver') as scope:
        if reuse:
            scope.reuse_variables()
        nin = x.get_shape()[-1].value -1
        y, t, f, dt = x[:,0], x[:,1], x[:,2], x[:,3]
        xx = x[:,0:3]

        with tf.variable_scope('1log-Relu'):
            nh1 = 3
            scale_int = np.zeros((nin,nh1))
            #scale_int[0][0] = 1.0
            scale_int[2][1] = 1.0
            scale_int[2][2] = -1.0
            #scale_int[3][3] = 1.0
            w1 = tf.get_variable("w1", [nin, nh1], initializer=tf.constant_initializer(scale_int), trainable=True)
            b1 = tf.get_variable("b1", [nh1], initializer=tf.constant_initializer(-1.0), trainable=True)
            h1 = tf.matmul(xx, w1)+b1
            h1 = tf.math.log(tf.math.maximum(tf.nn.relu( h1 ),1e-9), name = 'h1')

        with tf.variable_scope('2expo'):
            nh2 = 2
            scale_int = np.zeros((nh1+nin,nh2))
            scale_int[1][0] = 1.0
            #scale_int[3][0] = 1.0
            scale_int[2][1] = 1.0
            #scale_int[3][1] = 1.0
            #scale_int[0][2] = 1.0
            #scale_int[0][3] = -1.0
            w2 = tf.get_variable("w2", [nh1+nin, nh2], initializer=tf.constant_initializer(scale_int), trainable=True)
            b2 = tf.get_variable("b2", [nh2], initializer=tf.constant_initializer(0.0), trainable=True)
            pp = tf.concat([h1,xx],1)
            h2 = tf.math.minimum(tf.matmul(pp, w2)+b2,10)
            h2 = tf.math.exp( h2, name='h2')

        with tf.variable_scope('3final'):
            nh3 = 1
            scale_int = np.zeros((nh2+nin,nh3))
            w3 = tf.get_variable("w3", [nh2+nin, 1], initializer=tf.constant_initializer(scale_int), trainable=True)
            b3 = tf.get_variable("b3", [1], initializer=tf.constant_initializer(-.8), trainable=True)
            pp = tf.concat([h2*0,xx*0],1)
            h3 = tf.add(tf.matmul(pp, w3), b3, name='h3')
        #h3 = tf.get_variable("h3", [1], initializer=tf.constant_initializer(-1.0), trainable=True)
        #h3=h1

        with tf.variable_scope('4model'):#tf.math.maximum(h3,5e-4)
            #print("M", M.shape)
            with tf.variable_scope('NN-Var'):
                M = tf.add(h3,0,name="M")
                #M  = tf.get_variable("M", [1], initializer=tf.constant_initializer(-1.0), trainable=True)
                N0 = tf.get_variable("N0", [1], initializer=tf.constant_initializer(0.0), trainable=True)
                N1 = tf.get_variable("N1", [1], initializer=tf.constant_initializer(0.1), trainable=True)
                N2 = tf.get_variable("N2", [1], initializer=tf.constant_initializer(0.0), trainable=True)
            Mt = M*t
            Mdt = M*(t+dt)
            phi0 = N0 + N1*Mt + N2/2*Mt
            phi1 = N1 + N2*Mt
            #phi2 = 2*N2
            pht0 =  N0 + N1*Mdt + N2/2*Mdt
            pht1 = N1 + N2*Mdt
            #pht2 = tf.zeros_like(h3) + N2
            #print("pht0", pht0.shape)
            FF1 = phi0 + phi1 + N2
            FF2 = pht0 + pht1 + N2
            z  = -FF2 +tf.math.exp(M*dt)*(y+FF1+dt*(f-M*(y+phi0)))
            #print("z", z.shape)
    return z

def fun2():
    input_layer = tf.placeholder('float32', shape=[None, 4], name = "input")
    yn0, x0, fn0, dx = input_layer[:, 0], input_layer[:, 1], input_layer[:, 2], input_layer[:, 3]

    with tf.variable_scope('Step-y1'):
        yn1 = _build_solver(input_layer)
        yn1 = yn1[:,0]
        fn1 = func(yn1,x0+1.0*dx)
        #print("Hey")

#    with tf.variable_scope('Step-y05'):
        g05 = tf.stack([yn0,x0,fn0,dx/2],axis=1)
        yn05 = _build_solver(g05, reuse = True)
        yn05 = yn05[:,0]
        fn05= func(yn05,x0+0.5*dx)

#    with tf.variable_scope('Step-y10'):
        g10 = tf.stack([yn05,x0+0.5*dx,fn05,dx/2],axis=1)
        yn10 = _build_solver(g10, reuse = True)
        yn10 = yn10[:,0]
        fn10= func(yn10,x0+1.0*dx)

    #g15 = tf.stack([yn1[:,0],x0+dx,fn1[:,0],dx/2],axis=1)
    #yn15 = _build_solver(g15, reuse = True)
    #fn15= func(yn15,x0+1.5*dx)
    #g20 = tf.stack([yn1[:,0],x0+1.0*dx,fn1[:,0],dx],axis=1)
    #yn20 = _build_solver(g20, reuse = True)
    #fn20= func(yn20,x0+2.0*dx)

    with tf.variable_scope('loss'):
        ode_loss = 0.00*tf.reduce_mean(tf.square(yn1 -yn0 -dx*fn0))             # forward Euler
        ode_loss += tf.reduce_mean(tf.square(yn1 -yn10))
        #ode_loss += tf.reduce_mean(tf.square(yn1 - yn0-dx*(fn0+fn1)/2))     # backward Euler
        #ode_loss = tf.reduce_mean(tf.square(yn1 - yn0-dx*(fn0+2*fn05+fn1)/4))  #1/2 Simpson
        #ode_loss += tf.reduce_mean(tf.square(yn1 - yn0-dx*(fn0+4*fn05+fn10)/6))  #1/3 Simpson
        #ode_loss = tf.reduce_mean(tf.square(yn20 - yn0-dx*(fn0+4*fn05+2*fn1+4*fn15+fn20)/6)) #1/3 Simpson
        #ode_loss = tf.reduce_mean(tf.square(yn20-4/3*yn1 +1/3*yn0-dx*(fn20)*2/3)) #BDF2

        err_est = tf.reduce_mean(tf.square(yn1-yn10))

    with tf.variable_scope('Print-Variable'):
        t_vars = tf.all_variables()
        nn = [var for var in t_vars if 'b3' in var.name]
        print_var1 = tf.Print(nn,nn, message = "M(b3) = ")
        nn = [var for var in t_vars if 'NN-Var' in var.name]
        print_var2 = tf.Print(nn,nn, message = "Var = ")
        print(nn)

    learn_rate = 10
    #optimizer = tf.train.AdamOptimizer(learn_rate).minimize(ode_loss)
    optimizer = tf.train.GradientDescentOptimizer(learn_rate).minimize(ode_loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    writer = tf.summary.FileWriter("log")
    writer.add_graph(sess.graph)
    summary = tf.summary.merge([
                    tf.summary.scalar("loss", (ode_loss))
                    ])

    print("Train Step")
    batch_size = 5000
    hm_epochs = 5001
    for index in range(hm_epochs):
        y_center = 1#np.random.rand(1)[0]*2
        x_center = 3#np.random.rand()*6
        dt_center = 0# np.random.rand(1)[0]*dt#*0.5
        input = []
        x = 0.1
        for i in range(batch_size):
           y = np.random.randn()*2 + y_center
           x = np.random.randn()*3 + x_center
           dt_train = max(np.random.rand()*dt*1.5+dt_center,1e-10)
           input.append([y,x,func(y,x),dt_train])
        if index%10==0:
            _,lloss = sess.run([optimizer,ode_loss],feed_dict={input_layer:input})
            #lloss = sess.run(ode_loss,feed_dict={input_layer:input})
            summ = sess.run(summary, feed_dict={input_layer:input})
            writer.add_summary(summ,index)
        if index%200==0:
            _,_ = sess.run([print_var1,print_var2],feed_dict={input_layer:input})
            print(index, lloss)

    _,_ = sess.run([print_var1,print_var2],feed_dict={})

    print("Test Step")
    x = 0.0
    y_nn= np.zeros(np.shape(xs))
    y_nn[0] = y0
    y_nn2= np.zeros(np.shape(xs))
    y_nn2[0] = y0
    y = y0
    lloss = 0.0
    dt_train = dt
    for i in range(1,n):
        y = y_nn[i-1]
        input = [[y,x,func(y,x),dt_train]]
        ynn,loss = sess.run([yn1,ode_loss],feed_dict={input_layer:input})
        y_nn[i] = ynn[0]

        y = y_nn2[i-1]
        input = [[y,x,func(y,x),dt_train/2]]
        ynn,loss = sess.run([yn10,ode_loss],feed_dict={input_layer:input})
        input = [[ynn,x+dt_train/2,func(ynn,x+dt_train/2),dt_train/2]]
        ynn2,loss = sess.run([yn10,ode_loss],feed_dict={input_layer:input})
        y_nn2[i] = ynn2[0]

        x += dt_train
        print(i,"Err",loss)
        lloss += loss
    #print('Test Step:', lloss)


    # Plot the numerical solution
    #plt.rcParams.update({'font.size': 14})  # increase the font size
    #plt.xlabel("x")
    #plt.ylabel("y")

    #y_exact = xs - 1 + 2*np.exp(-xs)
    #plt.plot(xs, y_scipy);
    #plt.plot(xs, y_euler, ".");
    #plt.plot(xs, y_nn, "+");
    #plt.plot(xs, y_nn2, "+");
    #plt.show()

    def recur(y,x,dt,err_max=1e-4):
        input = [[y,x,func(y,x),dt_train]]
        ynn,loss = sess.run([yn10,err_est],feed_dict={input_layer:input})
        if loss<err_max or dt<5e-3:
            return ynn
        else:
            print("x=",x,", dt=",dt,", err =", loss)
            return recur(recur(y,x,dt/2,err_max),x+dt/2,dt/2,err_max)
    """
    print("Test loop")
    x = 0.0
    y_lp= np.zeros(np.shape(xs))
    y_lp[0] = y0
    y = y0
    dt_train = dt
    for i in range(1,n):
        y = y_nn[i-1]
        ynn = recur(y,x,dt,5e-6)
        y_lp[i] = ynn[0]
        x += dt_train
    """

    plt.figure()
    ax = plt.subplot(2,1,1)
    y_exact = xs - 1 + 2*np.exp(-xs)
    plt.plot(xs,abs(y_scipy-y_scipy))
    plt.plot(xs,abs(y_scipy-y_euler),"+")
    plt.plot(xs,abs(y_scipy-y_nn),".")
    plt.plot(xs,abs(y_scipy-y_nn2),".")
    #plt.plot(xs,abs(y_exact-y_lp),"^")
    #print((y_exact))
    #print((y_nn))
    #print((y_nn2))
    ax = plt.subplot(2,1,2)
    plt.plot(xs,y_scipy)
    plt.plot(xs,y_euler,"+")
    plt.plot(xs,y_nn,".")
    plt.plot(xs,y_nn2,".")
    #plt.plot(xs,y_lp,"^")
    plt.show()

def main(arv):
    fun2()

if __name__ == "__main__":
    sys.excepthook = colored_hook(
        os.path.dirname(os.path.realpath(__file__)))
    tf.app.run()

#y_diff = np.abs(y_exact - y_scipy)
#plt.semilogy(xs, y_diff)
#plt.ylabel("Error")
#plt.xlabel("x")
#plt.title("Error in numerical integration");
# Note the logarithmic scale on the y-axis.
