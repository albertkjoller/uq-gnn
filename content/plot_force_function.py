#from synthetic_data_generation import U
import matplotlib.pyplot as plt


def plot_gg_mstd(x,y):
    plt.style.use('ggplot')
    plt.plot(x, y, c='firebrick', label=r'$U(x) = \frac{500000}{x^{12}} - \frac{4000}{x^6}$')
    plt.axvspan(2.601193297843941,4.058931030865776,alpha=0.2, color='b')
    plt.vlines(3.3300621643548585,-10,10,linestyles='--',colors='b',label='OOD')

    plt.axvspan(3.0866713169349715,5.686421181700695,alpha=0.2, color='r')
    plt.vlines(4.386546249317833,-10,10,linestyles='--',colors='r',label='ID')

    plt.hlines(0, 0, 6, linestyles='--', colors='black')
    plt.xlabel('distance between atoms')
    plt.ylabel('$U(x)$')
    plt.legend()
    plt.xlim(2.30,6)
    plt.ylim(-10, 4)
    plt.legend(loc='lower right', fontsize=15)
    plt.tight_layout()
    plt.savefig('figures/synthetic_range.png')
    plt.show()

def plot_gg(x,y):
    plt.style.use('ggplot')
    plt.plot(x,y,c='firebrick', label=r'$U(x) = \frac{500000}{x^{12}} - \frac{4000}{x^6}$')
    plt.hlines(0,0,10, linestyles='--', colors='black')
    plt.xlabel('distance between atoms')
    plt.ylabel('$U(x)$')
    plt.legend(loc='lower right', fontsize=20)
    plt.xlim(0,10)
    plt.ylim(-10,10)
    plt.tight_layout()
    plt.savefig('figures/synthetic_function.png')
    plt.show()
