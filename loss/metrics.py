import numpy as np


def Pearson_np(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    if np.sqrt(np.sum(vx ** 2)) == 0 and np.sqrt(np.sum(vy ** 2)) == 0:
        r = 1
    elif min(np.sqrt(np.sum(vx ** 2)), np.sqrt(np.sum(vy ** 2))) == 0:
        r = 0
    else:
        r = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return r


def rmse(x,y):
    error = 0.0
    for i in range(len(x)):
        error += (x[i] - y[i])**2
    return (error / len(x))**0.5
def mae(x,y):
    error = 0.0
    for i in range(len(x)):
        error += abs(x[i] - y[i])
    return error / len(x)
def bpm(mae_val,interval):
    
    if mae_val <= interval:
        return 1
    else:
    	return 0
 
def bpm2(x,y,interval):
    output = list()
    for i in range(len(x)):
        diff = abs(x[i]-y[i])
        if diff <= interval:
            output.append(1)
        else:
            output.append(0)
    return sum(output)/len(output)
def bpm3(x,interval):
    output = 0.0
    for i in range(len(x)):
        if x[i] <= interval:
            output += 1
    return output/len(x)