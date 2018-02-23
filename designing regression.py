from statistics import mean
import matplotlib.pyplot as plt
import numpy as np

 

xs = np.array([100,75,56,87,44,55,66,77])
ys= np.array([ 94,69,65,90,38,50,76,77])


def linear_regression(xs,ys):
    first = mean(xs) * mean(ys)
    second= mean(xs * ys)
    third= mean(xs) * mean(xs)
    fourth = mean(xs * xs)

    m= ((first - second)) / ((third - fourth))
    b= mean(ys) - m * mean(xs) 


    return m,b


m,b= linear_regression(xs,ys)
print(m,b)

predict_ys=[]

predict_xs = [35,34,45,67,87,98]

for x in predict_xs:
    predict_y = (m*x) + b
    predict_ys.append(predict_y)

    
print(predict_ys)

#plt.scatter(predict_x,predict_y,color='g')
plt.scatter(xs,ys)

best_fit_line= [(m * x)+b for x in xs] 

plt.plot(xs, best_fit_line)
plt.show()
