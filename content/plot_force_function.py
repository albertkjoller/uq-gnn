#from synthetic_data_generation import U
import matplotlib.pyplot as plt


def plot_gg_mstd(x,y):
    plt.style.use('ggplot')
    plt.plot(x, y, c='firebrick', label=r'$U(x) = \frac{500000}{x^{12}} - \frac{4000}{x^6}$')
    plt.axvspan(4.541109878943798,9.29665518639443,alpha=0.2, color='r')
    plt.vlines(6.918882532669114,-10,10,linestyles='--',colors='r',label='ID')

    plt.axvspan(3.147881957605556,5.790439446565278,alpha=0.2, color='b')
    plt.vlines(4.469160702085417,-10,10,linestyles='--',colors='b',label='OOD')

    plt.hlines(0, 0, 10, linestyles='--', colors='black')
    plt.xlabel('distance between atoms')
    plt.title('Force Between Atoms')
    plt.ylabel('$U(x)$')
    plt.legend()
    plt.xlim(2.30,10)
    plt.ylim(-10, 4)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('figures/synthetic_range.png')
    plt.show()

def plot_gg(x,y):
    plt.style.use('ggplot')
    plt.plot(x,y,c='firebrick', label=r'$U(x) = \frac{500000}{x^{12}} - \frac{4000}{x^6}$')
    plt.hlines(0,0,10, linestyles='--', colors='black')
    plt.xlabel('distance between atoms')
    plt.title('Force Between Atoms')
    plt.ylabel('$U(x)$')
    plt.legend()
    plt.xlim(2.30,10)
    plt.ylim(-10,4)
    plt.show()
