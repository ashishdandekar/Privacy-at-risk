from formulae import *
from compositions import *

'''
    sudo apt-get install texlive-latex-recommended
    sudo apt install texlive-latex-extra
    sudo apt install dvipng
    sudo apt install texlive-fonts-recommended texlive-fonts-extra cm-super

    despite this you need to install type1cm.sty separately
'''

plt.rcParams['text.usetex'] = True #Let TeX do the typsetting
plt.rcParams['text.latex.preamble'] = [r'\usepackage{sansmath}', r'\sansmath'] #Force sans-serif math mode (for axes labels)
plt.rcParams['font.family'] = 'sans-serif' # ... for regular text
plt.rcParams['font.sans-serif'] = 'Helvetica' # Choose a nice font here

eps = np.linspace(0, 1)

# plt.clf()
# for i in [1.0, 1.5, 2.0, 2.5]:
#     plt.plot([1 - bound(e, i, 1.0) for e in eps], eps, label = "$\epsilon_0$=%.1f"%i)
# plt.legend()
# plt.xlabel("Privacy at Risk ($\gamma_1$)")
# plt.ylabel("Privacy level ($\epsilon$)")
# plt.savefig("single.eps", bbox_inches = "tight", figsize=(15,15))

# plt.clf()
# for k in [1., 2., 3.]:
#     plt.plot([1 - bound(e, 1.0, k) for e in eps], eps, label = "$k$=%d"%k)
# plt.legend()
# plt.xlabel("Privacy at Risk ($\gamma_1$)")
# plt.ylabel("Privacy level ($\epsilon$)")
# plt.savefig("general.eps", bbox_inches = "tight", figsize=(15,15))

# gammas = np.linspace(0, 1)
# plt.clf()
# for i in [0.0001, 0.001, 0.01, 0.1]:
#     plt.plot(gammas, [calculate_nsamples(i, 1 - g) for g in gammas], label = "$\\rho=%.4f$"%i)
# plt.yscale('log')
# plt.legend(loc="upper left")
# plt.xlabel("Privacy at Risk ($\gamma_2$)")
# plt.ylabel("Sample size ($n$)")
# plt.savefig("sample_size.eps", bbox_inches = "tight", figsize=(15,15))

# plt.clf()
# for r in [0.01, 0.012, 0.015, 0.02]:
#     alpha = 1. - (2 * np.exp(-2 * (r**2) * 10000))
#     plt.plot([1 - calculate_gamma(e, 1.0, 0.01, 0.15, alpha, 1) for e in eps], eps, label =  "$\\rho$=%.3f"%r)
# plt.legend()
# plt.xlabel("Privacy at Risk ($\gamma_3$)")
# plt.ylabel("Privacy level ($\epsilon$)")
# plt.savefig("rho_dep.eps", bbox_inches = "tight", figsize=(15,15))
# 
# plt.clf()
# for n in [10000, 15000, 20000, 25000]:
#     alpha = 1. - (2 * np.exp(-2 * (0.01**2) * n))
#     plt.plot([1- calculate_gamma(e, 1.0, 0.01, 0.15, alpha, 1) for e in eps], eps, label =  "$n$=%d"%n)
# plt.legend()
# plt.xlabel("Privacy at Risk ($\gamma_3$)")
# plt.ylabel("Privacy level ($\epsilon$)")
# plt.savefig("sample_dep.eps", bbox_inches = "tight", figsize=(15,15))

# B = 5500 * 100
# dp_bud = lambda e: B * np.exp(-1. / e)
# 
# def dp_par(e, e0):
#     gamma = bound(e, e0, 1) if e <= e0 else 0
#     return (gamma * dp_bud(e) + (1 - gamma)*dp_bud(e0))
# 
# eps = np.linspace(0, 0.7)
# for e0 in [0.7, 0.6, 0.5]:
#     plt.plot(eps, [dp_par(e, e0) for e in eps], label="$\epsilon_0$=%.1f" % e0)
# plt.legend()
# plt.xlabel("Privacy level ($\epsilon$)")
# plt.ylabel("$B_{par}$ (in dollars)")
# plt.savefig("b_par.eps", bbox_inches = "tight", figsize=(15,15))

ks = range(1, 300)
eps0s = [0.1, 0.5, 1.0]
opt_eps = []
for eps0 in eps0s:
    opt_eps.append(optimize.newton(best_eps_cost, 0.002, args = (eps0,)))
for i, eps_0 in enumerate(eps0s):
    plt.clf()
    plt.plot(ks, [eps_0 * k for k in ks], label = "Basic Composition[10]")
    plt.plot(ks, [composition(eps_0, k, "azuma") for k in ks], label = "Advanced Composition[10]")
    # plt.plot(ks, [composition(eps_0, k, "bernstein") for k in ks], label = "advanced_bernstein")
    # plt.plot(ks, [composition(eps_0, k, "hoeffding") for k in ks], label = "advanced_hoeffding")
    plt.plot(ks, [composition_par(eps_0, opt_eps[i], k) for k in ks], label = "Composition with Privacy at Risk")
    plt.legend()
    plt.xlabel("Number of compositions ($n$)")
    plt.ylabel("Privacy level after composition ($\epsilon'$)")
    plt.title("Advanced composition for $\epsilon_0 = %0.2f, \delta=10^{-5}$" % eps_0)
    plt.savefig("advanced_%d.eps" % i, bbox_inches = "tight", figsize=(15,15))
    # plt.show()
