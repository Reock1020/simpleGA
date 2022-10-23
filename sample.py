import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import numpy as np
import matplotlib.pyplot as plt
from environments import RPD
from agent import RuleAgent
from ga import SimpleGA

ga = SimpleGA()
ga.evolve()



