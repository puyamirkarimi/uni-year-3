import numpy as np
from matplotlib import pyplot

x = [3, 10, 100, 1000, 10000]
x_smooth = np.arange(3,10000,1)

# barnes hut data
y1 = np.array([
    [0.000488042831421, 0.000306129455566, 0.000297784805298, 0.000322103500366, 0.000326156616211],
    [0.00184512138367, 0.00210499763489, 0.00185799598694, 0.00190591812134, 0.00286602973938],
    [0.091826915741, 0.0935060977936, 0.0940790176392, 0.0929820537567, 0.0914828777313],
    [4.22250103951, 4.24157190323, 4.37925410271, 4.17192482948, 4.15701293945],
    [88.0697581768, 91.2797408104, 91.239814043, 89.0220680237, 87.7484481335]
])

# brute force data
y2 = np.array([
    [0.000200986862183, 0.000205039978027, 0.000213861465454, 0.000205039978027, 0.000195980072021],
    [0.00111508369446, 0.00127601623535, 0.00110697746277, 0.00109195709229, 0.00109696388245],
    [0.0919270515442, 0.0923490524292, 0.0923042297363, 0.0920779705048, 0.0910761356354],
    [9.40417122841, 10.2120091915, 9.6259329319, 9.63111209869, 9.67636203766],
    [835.929067135, 845.845757008, 830.099977016, 845.752336979, 841.425284147]
])

N = len(y1[0,:])
y1_av = np.zeros(N)
y2_av = np.zeros(N)
y1_std_error = np.zeros(N)
y2_std_error = np.zeros(N)
weights1 = []
weights2 = []

for i in range(len(y1[:,0])):
    y1_av[i] = np.mean(y1[i, :])
    y2_av[i] = np.mean(y2[i, :])
    y1_std_error[i] = 7*np.std(y1[i, :])/np.sqrt(len(y1[i, :]))
    y2_std_error[i] = 7*np.std(y2[i, :])/np.sqrt(len(y2[i, :]))
    weights1.append(y1_std_error[i]**(-2))
    weights2.append(y2_std_error[i]**(-2))
fit1_coeffs = np.polyfit(x, y1_av, 5, w=weights1)
fit2_coeffs = np.polyfit(x, y2_av, 2, w=weights2)
fit1 = np.polyval(fit1_coeffs, x_smooth)
fit2 = np.polyval(fit2_coeffs, x_smooth)


fig = pyplot.figure(1)

'''#frame1=fig1.add_axes((.12,.3,.8,.6))

for i in range(len(phaseError)):
    weights.append(phaseError[i]**(-2))

coefs = np.lib.polyfit(dx, phaseShift, 1, w=weights)
fit_y = np.lib.polyval(coefs, dx)
print (60000000*2 * np.pi) / coefs[0]
line = pyplot.plot(dx, fit_y, 'b--')
pyplot.setp(line, color='0.5')'''
pyplot.scatter(x, y1_av, color='red')
pyplot.errorbar(x, y1_av, y1_std_error, linestyle="None", color='red')
pyplot.scatter(x, y2_av, color='green')
pyplot.errorbar(x, y2_av, y2_std_error, linestyle="None", color='green')
#pyplot.plot(x_smooth, fit1, 'r-')
#pyplot.plot(x_smooth, fit2, 'b-')
pyplot.xlabel("N")
pyplot.ylabel("Run time per timestep (s)")
pyplot.xscale('log')
pyplot.yscale('log')
pyplot.xlim((1, 1000))
'''
difference = phaseShift - fit_y
normResidual = [x/y for x, y in zip(difference, phaseError)]
frame2 = fig1.add_axes((.12,.1,.8,.2))
frame2.set_ylabel("Normalised residuals")
frame2.set_xlabel("Change in LED position/m")
frame2.set_autoscaley_on(False)
frame2.set_ylim([-4, 4])
frame2.set_autoscalex_on(False)
frame2.set_xlim([0, 2])
frame2.yaxis.set_ticks(np.array([-2,0,2]))
pyplot.plot(dx, normResidual, 'ro', markersize = 4.5)'''

#pyplot.show()
pyplot.savefig("graph.png")
