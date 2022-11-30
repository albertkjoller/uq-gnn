#from synthetic_data_generation import U
import matplotlib.pyplot as plt


def plot_gg_mstd(x,y):
    plt.style.use('ggplot')
    plt.plot(x, y, c='firebrick', label=r'$U(x) = \frac{500000}{x^{12}} - \frac{4000}{x^6}$')
    plt.axvspan(0.44061295280879714,0.9793612848709183,alpha=0.5, color='b')
    plt.vlines(0.7099871188398578,-10,10,linestyles='--',colors='b',label='(-0.5,0.5)')

    plt.axvspan(1.6637949084106864,3.6331468138407432,alpha=0.5, color='g')
    plt.vlines(2.6484708611257155,-10,10,linestyles='--',colors='g',label='(-2,2)')

    plt.axvspan(8.239987144268305,17.996905131788758,alpha=0.5, color='yellow')
    plt.vlines(13.118446138028531,-10,10,linestyles='--',colors='yellow',label='(-10,10)')

    plt.hlines(0, 0, 10, linestyles='--', colors='black')
    plt.xlabel('distance between atoms')
    plt.title('Force Between Atoms')
    plt.ylabel('$U(x)$')
    plt.legend()
    plt.ylim(-10, 10)
    plt.show()

def plot_gg(x,y):
    plt.style.use('ggplot')
    plt.plot(x,y,c='firebrick', label=r'$U(x) = \frac{500000}{x^{12}} - \frac{4000}{x^6}$')
    plt.hlines(0,0,10, linestyles='--', colors='black')
    plt.xlabel('distance between atoms')
    plt.title('Force Between Atoms')
    plt.ylabel('$U(x)$')
    plt.legend()
    plt.ylim(-10,10)
    plt.show()
