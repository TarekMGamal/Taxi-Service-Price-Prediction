import preprocessing as pre
import model1
import model2

if __name__ == '__main__':
    x, y = pre.pre_process()
    model1.polynomial_regression(x, y)
    model2.polynomial_regression(x, y)

