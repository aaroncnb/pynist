
#ddir = '/work1/users/aaronb/Codebrary/Python/Projects/pynist/data/raw/'
#ddir = '/home/aaronb/Codebrary/Python/Projects/pynist/data/raw/'
#ddir = '/home/aaronb/Projectbrary/python/pynist/data/raw/'
import costFunction as cf

ddir = '/home/aaronb/Projectbrary/pynist/data/raw/'



theta1, theta2, J, a1, a2, a3, output_label, result, score = cf.costLowerer(ddir, nneurons = 50, lam=1, alpha = 1e-5, num_iters=400, reg=False)
