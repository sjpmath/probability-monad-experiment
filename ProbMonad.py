import scipy.integrate as integrate
import scipy.special as special
import numpy as np
import matplotlib.pyplot as plt

class Function: # a function with its inverse image provided
	def __init__(self, f, f_inverse):
		self.func = f
		self.inverse = f_inverse

class ProbDist:
	def __init__(self, density):
		self.density = density

	def apply(self, func):
		dens = lambda x : self.density(func.inverse(x))
		return ProbDist(dens)

class ProbMonad:
	def pure(self, x):
		pass
	def bind(self, xm, f):
		pass


"""
class DiscreteProbMonad(ProbMonad):
	def pure(self, x):
		return [(x,1.0)]

	def bind(self, xm, f):
		ym = {}
		for x,p in xm:
			for y,q in f(x):
				if y not in ym: ym.update({y:p*q})
			else: ym.update({y:ym[y]+p*q})
		ans = []
		for y in ym:
			ans.append((y,ym[y]))
		return ans
"""

class ContProbMonad(ProbMonad):
	def pure(self, x):
		dens = lambda y : float(int(y==x))
		return ProbDist(dens)

	def bind(self, xm, f):
		dens = lambda x : ((integrate.quad(lambda y : (f(y)).density(x) * xm.density(y), -np.inf, np.inf))[0])
		return ProbDist(dens)


CPM = ContProbMonad()

def Exp_Dist(lam):
	dens = lambda x : (lam*np.exp(-lam*x) if x >= 0 else 0)
	return ProbDist(dens)

def Norm_Dist(m,s):
	dens = lambda x : (1/(s * np.sqrt(2) * np.pi)) * np.exp(-0.5 * ((x-m)/s)**2)
	return ProbDist(dens)

ExpProb = Exp_Dist(1)
NormProb = Norm_Dist(0.0,0.01)

class BayesianReg:
	def __init__(self, model):
		#self.num_params = 0
		#self.f = f # parameterised - it is lambda params -> data
		self.prior_dist = None
		self.model = model # parameterised prob_dist - it is lambda params -> dist
		self.posterior_dist = None
		self.posterior_dist_normalized = None

		self.data = []

	def sample(self, prior_dist):
		#self.num_params = len(prior_dist_list)
		self.prior_dist = prior_dist

	def observe(self, data):
		self.data.append(data)

	def compute_posterior_unnormed(self): # unnormalised
		"""
		likelihood_dens = lambda param : 1.0
		for data in self.data:
			temp = likelihood_dens
			likelihood_dens = lambda param : (((self.model)(param)).density(data) * temp(param))
		"""

		def likelihood_dens(param):
			ans = 1.0
			for data in self.data:
				ans *= ((self.model)(param)).density(data)
			return ans

		def dens(param):
			return likelihood_dens(param) * self.prior_dist.density(param)

		#dens = lambda param : likelihood_dens(param) * self.prior_dist.density(param)
		self.posterior_dist = ProbDist(dens)
		return self.posterior_dist
		#return CPM.bind(self.prior, self.likelihood)

	def norm(self):
		CPM = ContProbMonad()

		def likelihoodmodel(param):
			def likelihoodmodeldens(l):
				a = 1.0
				for data in l:
					a *= ((self.model)(param)).density(data)
				return a
			return ProbDist(likelihoodmodeldens)

		self.likelihoodmodel = likelihoodmodel
		self.model_evidence = CPM.bind(self.prior_dist, likelihoodmodel)
		normalizer = self.model_evidence.density(self.data)

		#model_evidence = CPM.bind(self.prior_dist, self.model)
		#normalizer = 1.0
		#for data in self.data:
		#	normalizer *= model_evidence.density(data)

		#print(normalizer)
		dens = lambda x : (self.posterior_dist.density(x)/normalizer)
		self.posterior_dist_normalized = ProbDist(dens)
		return self.posterior_dist_normalized

	def compute_prediction(self):
		CPM = ContProbMonad()
		return CPM.bind(self.posterior_dist, self.model)

x = np.linspace(0, 100, 100)

prior = Norm_Dist(0.0, 3.0)
data = [4.0, 5.0]
model = lambda p : Norm_Dist(p, 0.5)
BR = BayesianReg(model)
BR.sample(prior)
for d in data:
	BR.observe(d)
posterior = BR.compute_posterior_unnormed()
posterior_norm = BR.norm()
#prediction = BR.compute_prediction()
posterior_unnormed_dens = np.vectorize(posterior.density)
posterior_norm_dens = np.vectorize(posterior_norm.density)
#prediction_dens = np.vectorize(prediction.density)

plt.plot(x, posterior_unnormed_dens(x), color='red')
#plt.plot(x, prediction_dens(x), color='blue')
plt.plot(x, posterior_norm_dens(x), color='green')
plt.show()

print(((integrate.quad(lambda y : posterior_unnormed_dens(y), -np.inf, np.inf))[0]))
print(((integrate.quad(lambda y : posterior_norm_dens(y), -np.inf, np.inf))[0]))
print()

def add_randvars(randx, randy): # adding random variables
	CPM = ContProbMonad()
	f = lambda x : randy.apply(Function(lambda y : x+y, lambda y : y-x))
	return CPM.bind(randx, f)

x = np.linspace(0, 100, 100)
CPM = ContProbMonad()
X = Exp_Dist(1.0)
Y = Exp_Dist(1.0)
exp2 = add_randvars(X,Y)
exp1_dens = np.vectorize(X.density)
exp2_dens = np.vectorize(exp2.density)
plt.plot(x, exp1_dens(x), color='red')
plt.plot(x, exp2_dens(x), color='blue')
plt.show()

x = np.linspace(-50, 50, 100)
X = Norm_Dist(0.0,1.0)
Y = Norm_Dist(1.0,1.0)
Z = Norm_Dist(2.0,0.50)
norm2 = add_randvars(X,Y)
norm3 = add_randvars(norm2,Z)
norm1_dens = np.vectorize(X.density)
norm2_dens = np.vectorize(norm2.density)
norm3_dens = np.vectorize(norm3.density)
plt.plot(x, norm1_dens(x), color='red')
plt.plot(x, norm2_dens(x), color='blue')
plt.plot(x, norm3_dens(x), color='green')
plt.show()


def sum_randvars(list_randvars):
	if list_randvars == []: raise
	X = list_randvars[0]
	for i in range(1,len(list_randvars)):
		X = add_randvars(X, list_randvars[i])
	return X



"""

def expon_density(x):
	if x >= 0 : return np.exp(-x)
	else: return 0

def func(x):
	if x <= 0: dens = lambda y : 0
	else: dens = lambda y : (x * np.exp(-x * y) if y >=0 else 0)
	return ProbDist(dens)

newdist = CPM.bind(NormProb, func)

x = np.linspace(0, 100, 100)

vnewprob = np.vectorize(newdist.density)

plt.plot(x, vnewprob(x), color='red')

plt.show()
"""